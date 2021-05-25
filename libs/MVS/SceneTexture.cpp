/*
* SceneTexture.cpp
*
* Copyright (c) 2014-2015 SEACAVE
*
* Author(s):
*
*      cDc <cdc.seacave@gmail.com>
*
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*
* Additional Terms:
*
*      You are required to preserve legal notices and author attributions in
*      that material or in the Appropriate Legal Notices displayed by works
*      containing it.
*/

#include "Common.h"
#include "Scene.h"
#include "RectsBinPack.h"
// connected components
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/connected_components.hpp>

using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define TEXOPT_USE_OPENMP
#endif

// uncomment to use SparseLU for solving the linear systems
// (should be faster, but not working on old Eigen)
#if !defined(EIGEN_DEFAULT_TO_ROW_MAJOR) || EIGEN_WORLD_VERSION>3 || (EIGEN_WORLD_VERSION==3 && EIGEN_MAJOR_VERSION>2)
#define TEXOPT_SOLVER_SPARSELU
#endif

// method used to try to detect outlier face views
// (should enable more consistent textures, but it is not working)
#define TEXOPT_FACEOUTLIER_NA 0
#define TEXOPT_FACEOUTLIER_MEDIAN 1
#define TEXOPT_FACEOUTLIER_GAUSS_DAMPING 2
#define TEXOPT_FACEOUTLIER_GAUSS_CLAMPING 3
#define TEXOPT_FACEOUTLIER TEXOPT_FACEOUTLIER_GAUSS_CLAMPING

// method used to find optimal view per face
#define TEXOPT_INFERENCE_LBP 1
#define TEXOPT_INFERENCE_TRWS 2
#define TEXOPT_INFERENCE TEXOPT_INFERENCE_LBP

// inference algorithm
#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_LBP
#include "../Math/LBP.h"
namespace MVS {
typedef LBPInference::NodeID NodeID;
// Potts model as smoothness function
// 设置平滑cost,如果两个节点标签相同则cost=0,否则为MaxEnergy
// 目的是让相邻face的标签尽可能一致
LBPInference::EnergyType STCALL SmoothnessPotts(LBPInference::NodeID, LBPInference::NodeID, LBPInference::LabelID l1, LBPInference::LabelID l2) {
	return l1 == l2 && l1 != 0 && l2 != 0 ? LBPInference::EnergyType(0) : LBPInference::EnergyType(LBPInference::MaxEnergy);
}
}
#endif
#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_TRWS
#include "../Math/TRWS/MRFEnergy.h"
namespace MVS {
// TRWS MRF energy using Potts model
typedef unsigned NodeID;
typedef unsigned LabelID;
typedef TypePotts::REAL EnergyType;
static const EnergyType MaxEnergy(1);
struct TRWSInference {
	typedef MRFEnergy<TypePotts> MRFEnergyType;
	typedef MRFEnergy<TypePotts>::Options MRFOptions;

	CAutoPtr<MRFEnergyType> mrf;
	CAutoPtrArr<MRFEnergyType::NodeId> nodes;

	inline TRWSInference() {}
	void Init(NodeID nNodes, LabelID nLabels) {
		mrf = new MRFEnergyType(TypePotts::GlobalSize(nLabels));
		nodes = new MRFEnergyType::NodeId[nNodes];
	}
	inline bool IsEmpty() const {
		return mrf == NULL;
	}
	inline void AddNode(NodeID n, const EnergyType* D) {
		nodes[n] = mrf->AddNode(TypePotts::LocalSize(), TypePotts::NodeData(D));
	}
	inline void AddEdge(NodeID n1, NodeID n2) {
		mrf->AddEdge(nodes[n1], nodes[n2], TypePotts::EdgeData(MaxEnergy));
	}
	EnergyType Optimize() {
		MRFOptions options;
		options.m_eps = 0.005;
		options.m_iterMax = 1000;
		#if 1
		EnergyType lowerBound, energy;
		mrf->Minimize_TRW_S(options, lowerBound, energy);
		#else
		EnergyType energy;
		mrf->Minimize_BP(options, energy);
		#endif
		return energy;
	}
	inline LabelID GetLabel(NodeID n) const {
		return mrf->GetSolution(nodes[n]);
	}
};
}
#endif

// S T R U C T S ///////////////////////////////////////////////////

typedef Mesh::Vertex Vertex;
typedef Mesh::VIndex VIndex;
typedef Mesh::Face Face;
typedef Mesh::FIndex FIndex;
typedef Mesh::TexCoord TexCoord;

typedef int MatIdx;
typedef Eigen::Triplet<float,MatIdx> MatEntry;
typedef Eigen::SparseMatrix<float,Eigen::ColMajor,MatIdx> SparseMat;

enum Mask {
	empty = 0,
	border = 128,
	interior = 255
};

struct MeshTexture {
	// used to render the surface to a view camera
	typedef TImage<cuint32_t> FaceMap;
	struct RasterMesh : TRasterMesh<RasterMesh> {
		typedef TRasterMesh<RasterMesh> Base;
		FaceMap& faceMap;
		FIndex idxFace;
		RasterMesh(const Mesh::VertexArr& _vertices, const Camera& _camera, DepthMap& _depthMap, FaceMap& _faceMap)
			: Base(_vertices, _camera, _depthMap), faceMap(_faceMap) {}
		void Clear() {
			Base::Clear();
			faceMap.memset((uint8_t)NO_ID);
		}
		void Raster(const ImageRef& pt) {
			if (!depthMap.isInside(pt))
				return;
			const Depth z((Depth)INVERT(normalPlane.dot(camera.TransformPointI2C(Point2(pt)))));
			ASSERT(z > 0);
			Depth& depth = depthMap(pt);
			if (depth == 0 || depth > z) {
				depth = z;
				faceMap(pt) = idxFace;
			}
		}
	};

	// used to represent a pixel color
	typedef Point3f Color;
	typedef CLISTDEF0(Color) Colors;

	// used to store info about a face (view, quality)
	struct FaceData {
		VIndex idxView;// the view seeing this face
		float quality; // how well the face is seen by this view
		#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
		Color color; // additionally store mean color (used to remove outliers)
		#endif
	};
	typedef cList<FaceData,const FaceData&,0,8,uint32_t> FaceDataArr; // store information about one face seen from several views
	typedef cList<FaceDataArr,const FaceDataArr&,2,1024,FIndex> FaceDataViewArr; // store data for all the faces of the mesh

	// used to assign a view to a face
	typedef uint32_t Label;
	typedef cList<Label,Label,0,1024,FIndex> LabelArr;

	// represents a texture patch
	struct TexturePatch {
		Label label; // view index
		Mesh::FaceIdxArr faces; // indices of the faces contained by the patch
		RectsBinPack::Rect rect; // the bounding box in the view containing the patch
	};
	typedef cList<TexturePatch,const TexturePatch&,1,1024,FIndex> TexturePatchArr;

	// used to optimize texture patches
	// 用来优化纹理patch 
	struct SeamVertex {
		struct Patch {
			struct Edge {
				uint32_t idxSeamVertex; // 这个边的另一个顶点 the other vertex of this edge
				FIndex idxFace; // 在这个patch中包含该边的face id the face containing this edge in this patch

				inline Edge() {}
				inline Edge(uint32_t _idxSeamVertex) : idxSeamVertex(_idxSeamVertex) {}
				inline bool operator == (uint32_t _idxSeamVertex) const {
					return (idxSeamVertex == _idxSeamVertex);
				}
			};
			typedef cList<Edge,const Edge&,0,4,uint32_t> Edges;

			uint32_t idxPatch; // 包含该顶点的patch的id  the patch containing this vertex
			Point2f proj; // 该顶点在这个patch的投影坐标the projection of this vertex in this patch
			Edges edges; // 在这个patch中以这个点为起始的边，the edges starting from this vertex, contained in this patch (exactly two for manifold meshes)

			inline Patch() {}
			inline Patch(uint32_t _idxPatch) : idxPatch(_idxPatch) {}
			inline bool operator == (uint32_t _idxPatch) const {
				return (idxPatch == _idxPatch);
			}
		};
		typedef cList<Patch,const Patch&,1,4,uint32_t> Patches;

		VIndex idxVertex; // 顶点的索引 the index of this vertex
		Patches patches; // 包含该顶点的所有patchthe patches meeting at this vertex (two or more)

		inline SeamVertex() {}
		inline SeamVertex(uint32_t _idxVertex) : idxVertex(_idxVertex) {}
		inline bool operator == (uint32_t _idxVertex) const {
			return (idxVertex == _idxVertex);
		}
		// 取patch
		Patch& GetPatch(uint32_t idxPatch) {
			const uint32_t idx(patches.Find(idxPatch));
			if (idx == NO_ID)
				return patches.AddConstruct(idxPatch);
			return patches[idx];
		}
		// 根据patch id 进行排序
		inline void SortByPatchIndex(IndexArr& indices) const {
			indices.Resize(patches.GetSize());
			std::iota(indices.Begin(), indices.End(), 0);
			std::sort(indices.Begin(), indices.End(), [&](IndexArr::Type i0, IndexArr::Type i1) -> bool {
				return patches[i0].idxPatch < patches[i1].idxPatch;
			});
		}
	};
	typedef cList<SeamVertex,const SeamVertex&,1,256,uint32_t> SeamVertices;

	// used to iterate vertex labels
	struct PatchIndex {
		bool bIndex; // 记录顶点是否在边界
		union {
			uint32_t idxPatch; // 顶点所在patch id
			uint32_t idxSeamVertex; // 顶点若在边界上则对应seamvertex这个list的id是多少
		};
	};
	typedef CLISTDEF0(PatchIndex) PatchIndices;
	struct VertexPatchIterator {
		uint32_t idx; // 记录的是当前处理的第几个patch
		uint32_t idxPatch; // 当前的patch id
		const SeamVertex::Patches* pPatches;
		inline VertexPatchIterator(const PatchIndex& patchIndex, const SeamVertices& seamVertices) : idx(NO_ID) {
			if (patchIndex.bIndex) {
				pPatches = &seamVertices[patchIndex.idxSeamVertex].patches;
			} else {
				idxPatch = patchIndex.idxPatch;
				pPatches = NULL;
			}
		}
		inline operator uint32_t () const {
			return idxPatch;
		}
		inline bool Next() {
			if (pPatches == NULL)
				return (idx++ == NO_ID);
			if (++idx >= pPatches->GetSize())
				return false;
			idxPatch = (*pPatches)[idx].idxPatch;
			return true;
		}
	};

	// used to sample seam edges
	typedef TAccumulator<Color> AccumColor;
	typedef Sampler::Linear<float> Sampler;
	struct SampleImage {
		AccumColor accumColor;
		const Image8U3& image;
		const Sampler sampler;

		inline SampleImage(const Image8U3& _image) : image(_image), sampler() {}
		// sample the edge with linear weights
		void AddEdge(const TexCoord& p0, const TexCoord& p1) {
			const TexCoord p01(p1 - p0);
			const float length(norm(p01));
			ASSERT(length > 0.f);
			const int nSamples(ROUND2INT(MAXF(length, 1.f) * 2.f)-1);
			AccumColor edgeAccumColor;
			for (int s=0; s<nSamples; ++s) {
				const float len(static_cast<float>(s) / nSamples);
				const TexCoord samplePos(p0 + p01 * len);
				const Color color(image.sample<Sampler,Color>(sampler, samplePos));
				edgeAccumColor.Add(RGB2YCBCR(color), 1.f-len);
			}
			accumColor.Add(edgeAccumColor.Normalized(), length);
		}
		// returns accumulated color
		Color GetColor() const {
			return accumColor.Normalized();
		}
	};

	// used to interpolate adjustments color over the whole texture patch
	typedef TImage<Color> ColorMap;
	struct RasterPatchColorData {
		const TexCoord* tri;
		Color colors[3];
		ColorMap& image;

		inline RasterPatchColorData(ColorMap& _image) : image(_image) {}
		inline void operator()(const ImageRef& pt) {
			const Point3f b(BarycentricCoordinates(tri[0], tri[1], tri[2], TexCoord(pt)));
			#if 0
			if (b.x<0 || b.y<0 || b.z<0)
				return; // outside triangle
			#endif
			ASSERT(image.isInside(pt));
			image(pt) = colors[0]*b.x + colors[1]*b.y + colors[2]*b.z;
		}
	};

	// used to compute the coverage of a texture patch
	struct RasterPatchCoverageData {
		const TexCoord* tri;
		Image8U& image;

		inline RasterPatchCoverageData(Image8U& _image) : image(_image) {}
		inline void operator()(const ImageRef& pt) {
			ASSERT(image.isInside(pt));
			image(pt) = interior;
		}
	};

	// used to draw the average edge color of a texture patch
	// 用来计算纹理块的边界颜色
	struct RasterPatchMeanEdgeData {
		Image32F3& image;
		Image8U& mask;
		const Image32F3& image0;
		const Image8U3& image1;
		const TexCoord p0, p0Dir;
		const TexCoord p1, p1Dir;
		const float length;
		const Sampler sampler;

		inline RasterPatchMeanEdgeData(Image32F3& _image, Image8U& _mask, const Image32F3& _image0, const Image8U3& _image1,
									   const TexCoord& _p0, const TexCoord& _p0Adj, const TexCoord& _p1, const TexCoord& _p1Adj)
			: image(_image), mask(_mask), image0(_image0), image1(_image1),
			p0(_p0), p0Dir(_p0Adj-_p0), p1(_p1), p1Dir(_p1Adj-_p1), length((float)norm(p0Dir)), sampler() {}
		inline void operator()(const ImageRef& pt) {
			const float l((float)norm(TexCoord(pt)-p0)/length);
			// compute mean color
			// 计算边界插值的点在两个patch中的颜色均值
			const TexCoord samplePos0(p0 + p0Dir * l);
			AccumColor accumColor(image0.sample<Sampler,Color>(sampler, samplePos0), 1.f);
			const TexCoord samplePos1(p1 + p1Dir * l);
			accumColor.Add(image1.sample<Sampler,Color>(sampler, samplePos1)/255.f, 1.f);
			image(pt) = accumColor.Normalized();
			// set mask edge also
			// 设为边界标记
			mask(pt) = border;
		}
	};


public:
	MeshTexture(Scene& _scene, unsigned _nResolutionLevel=0, unsigned _nMinResolution=640);
	~MeshTexture();

	void ListVertexFaces();

	bool ListCameraFaces(FaceDataViewArr&, float fOutlierThreshold);

	#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
	bool FaceOutlierDetection(FaceDataArr& faceDatas, float fOutlierThreshold) const;
	#endif

	bool FaceViewSelection(float fOutlierThreshold, float fRatioDataSmoothness);

	void CreateSeamVertices();
	void GlobalSeamLeveling();
	void LocalSeamLeveling();
	void GenerateTexture(bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty);

	template <typename PIXEL>
	static inline PIXEL RGB2YCBCR(const PIXEL& v) {
		typedef typename PIXEL::Type T;
		return PIXEL(
			v[0] * T(0.299) + v[1] * T(0.587) + v[2] * T(0.114),
			v[0] * T(-0.168736) + v[1] * T(-0.331264) + v[2] * T(0.5) + T(128),
			v[0] * T(0.5) + v[1] * T(-0.418688) + v[2] * T(-0.081312) + T(128)
		);
	}
	template <typename PIXEL>
	static inline PIXEL YCBCR2RGB(const PIXEL& v) {
		typedef typename PIXEL::Type T;
		const T v1(v[1] - T(128));
		const T v2(v[2] - T(128));
		return PIXEL(
			v[0]/* * T(1) + v1 * T(0)*/ + v2 * T(1.402),
			v[0]/* * T(1)*/ + v1 * T(-0.34414) + v2 * T(-0.71414),
			v[0]/* * T(1)*/ + v1 * T(1.772)/* + v2 * T(0)*/
		);
	}


protected:
	static void ProcessMask(Image8U& mask, int stripWidth);
	static void PoissonBlending(const Image32F3& src, Image32F3& dst, const Image8U& mask, float bias=1.f);


public:
	const unsigned nResolutionLevel; // 多少倍来下采样图像用来贴图 how many times to scale down the images before mesh optimization
	const unsigned nMinResolution; // 用来贴图的图像的最小分辨率阈值

	// store found texture patches
	TexturePatchArr texturePatches;

	// used to compute the seam leveling
	PairIdxArr seamEdges; // 不同纹理patch的相交边，由两个邻接面表示（两个面共享一条边）idthe (face-face) edges connecting different texture patches
	Mesh::FaceIdxArr components; // 存储每个face对应的纹理patch id ;for each face, stores the texture patch index to which belongs
	IndexArr mapIdxPatch; // 无效纹理块被移除后，新的id与旧的映射 remap texture patch indices after invalid patches removal
	SeamVertices seamVertices; // 存储不同patch间的邻接edge。array of vertices on the border between two or more patches

	// valid the entire time
	Mesh::VertexFacesArr& vertexFaces; // 每个顶点包含的所有faces。for each vertex, the list of faces containing it
	BoolArr& vertexBoundary; // 记录每个顶点是否是边界点。for each vertex, stores if it is at the boundary or not
	Mesh::TexCoordArr& faceTexcoords; // 存储每个face的三个顶点的纹理坐标。for each face, the texture-coordinates of the vertices
	Image8U3& textureDiffuse; //纹理图 texture containing the diffuse color

	// constant the entire time
	Mesh::VertexArr& vertices;
	Mesh::FaceArr& faces;
	ImageArr& images;

	Scene& scene; // the mesh vertices and faces
};

MeshTexture::MeshTexture(Scene& _scene, unsigned _nResolutionLevel, unsigned _nMinResolution)
	:
	nResolutionLevel(_nResolutionLevel),
	nMinResolution(_nMinResolution),
	vertexFaces(_scene.mesh.vertexFaces),
	vertexBoundary(_scene.mesh.vertexBoundary),
	faceTexcoords(_scene.mesh.faceTexcoords),
	textureDiffuse(_scene.mesh.textureDiffuse),
	vertices(_scene.mesh.vertices),
	faces(_scene.mesh.faces),
	images(_scene.images),
	scene(_scene)
{
}
MeshTexture::~MeshTexture()
{
	vertexFaces.Release();
	vertexBoundary.Release();
}

// extract array of triangles incident to each vertex
// and check each vertex if it is at the boundary or not
// 提取每个顶点包含的faces,并判断是否在边界
void MeshTexture::ListVertexFaces()
{
	scene.mesh.EmptyExtra();
	scene.mesh.ListIncidenteFaces();
	scene.mesh.ListBoundaryVertices();
}

// extract array of faces viewed by each image
/**
 * @brief  提取被image看到的所有faces，计算每个face能看到的views 相关信息（view id 投影到对应view的梯度幅值加和及颜色均值）
 * 
 * @param[in] facesDatas  存储的是每个face，投影到对应views相关信息（view id,每个投影点对应图像梯度幅值及颜色均值）
 * @param[in] fOutlierThreshold 判断face投影的view是否是外点的阈值
 * @return true 
 * @return false 
 */
bool MeshTexture::ListCameraFaces(FaceDataViewArr& facesDatas, float fOutlierThreshold)
{
	// create vertices octree
	// 构建八叉树
	facesDatas.Resize(faces.GetSize());
	typedef std::unordered_set<FIndex> CameraFaces;
	struct FacesInserter {
		FacesInserter(const Mesh::VertexFacesArr& _vertexFaces, CameraFaces& _cameraFaces)
			: vertexFaces(_vertexFaces), cameraFaces(_cameraFaces) {}
		inline void operator() (IDX idxVertex) {
			const Mesh::FaceIdxArr& vertexTris = vertexFaces[idxVertex];
			FOREACHPTR(pTri, vertexTris)
				cameraFaces.emplace(*pTri);
		}
		inline void operator() (const IDX* idices, size_t size) {
			FOREACHRAWPTR(pIdxVertex, idices, size)
				operator()(*pIdxVertex);
		}
		const Mesh::VertexFacesArr& vertexFaces;
		CameraFaces& cameraFaces;
	};
	// 构建octree数据结构
	typedef TOctree<Mesh::VertexArr,float,3> Octree;
	const Octree octree(vertices);
	#if 0 && !defined(_RELEASE)
	Octree::DEBUGINFO_TYPE info;
	octree.GetDebugInfo(&info);
	Octree::LogDebugInfo(info);
	#endif

	// extract array of faces viewed by each image
	// 提取被每个image看到的faces
	Util::Progress progress(_T("Initialized views"), images.GetSize());
	typedef float real;
	TImage<real> imageGradMag; // 图像梯度幅值
	TImage<real>::EMat mGrad[2]; // 梯度x,y两个方向
	FaceMap faceMap; // 图像像素对应的face id
	DepthMap depthMap; // 图像像素对应的深度
	#ifdef TEXOPT_USE_OPENMP
	bool bAbort(false);
	#pragma omp parallel for private(imageGradMag, mGrad, faceMap, depthMap)
	// 每帧图像逐个处理
	for (int_t idx=0; idx<(int_t)images.GetSize(); ++idx) {
		#pragma omp flush (bAbort)
		if (bAbort) {
			++progress;
			continue;
		}
		const uint32_t idxView((uint32_t)idx);
	#else
	FOREACH(idxView, images) {
	#endif
		Image& imageData = images[idxView];
		if (!imageData.IsValid()) {
			++progress;
			continue;
		}
		// load image 
		// 加载图像，计算用于贴图的图像分辨率
		unsigned level(nResolutionLevel);
		const unsigned imageSize(imageData.RecomputeMaxResolution(level, nMinResolution));
		if ((imageData.image.empty() || MAXF(imageData.width,imageData.height) != imageSize) && !imageData.ReloadImage(imageSize)) {
			#ifdef TEXOPT_USE_OPENMP
			bAbort = true;
			#pragma omp flush (bAbort)
			continue;
			#else
			return false;
			#endif
		}
		// 更新相机参数
		imageData.UpdateCamera(scene.platforms);
		// compute gradient magnitude
		// 计算图像梯度幅值来作为图像质量
		imageData.image.toGray(imageGradMag, cv::COLOR_BGR2GRAY, true);
		cv::Mat grad[2];
		mGrad[0].resize(imageGradMag.rows, imageGradMag.cols);
		grad[0] = cv::Mat(imageGradMag.rows, imageGradMag.cols, cv::DataType<real>::type, (void*)mGrad[0].data());
		mGrad[1].resize(imageGradMag.rows, imageGradMag.cols);
		grad[1] = cv::Mat(imageGradMag.rows, imageGradMag.cols, cv::DataType<real>::type, (void*)mGrad[1].data());
		#if 1
		// 计算梯度
		cv::Sobel(imageGradMag, grad[0], cv::DataType<real>::type, 1, 0, 3, 1.0/8.0);
		cv::Sobel(imageGradMag, grad[1], cv::DataType<real>::type, 0, 1, 3, 1.0/8.0);
		#elif 1
		const TMatrix<real,3,5> kernel(CreateDerivativeKernel3x5());
		cv::filter2D(imageGradMag, grad[0], cv::DataType<real>::type, kernel);
		cv::filter2D(imageGradMag, grad[1], cv::DataType<real>::type, kernel.t());
		#else
		const TMatrix<real,5,7> kernel(CreateDerivativeKernel5x7());
		cv::filter2D(imageGradMag, grad[0], cv::DataType<real>::type, kernel);
		cv::filter2D(imageGradMag, grad[1], cv::DataType<real>::type, kernel.t());
		#endif
		// 计算梯度的幅值mag=sqrt(dx^2+dy^2)
		(TImage<real>::EMatMap)imageGradMag = (mGrad[0].cwiseAbs2()+mGrad[1].cwiseAbs2()).cwiseSqrt();
		// select faces inside view frustum
		// 选择视锥中的faces
		CameraFaces cameraFaces; // 存储了视锥内所有faces
		FacesInserter inserter(vertexFaces, cameraFaces);
		typedef TFrustum<float,5> Frustum;
		const Frustum frustum(Frustum::MATRIX3x4(((PMatrix::CEMatMap)imageData.camera.P).cast<float>()), (float)imageData.width, (float)imageData.height);
		octree.Traverse(frustum, inserter);
		// project all triangles in this view and keep the closest ones
		// 投影在这个view中的faces及对应深度，只保留最近的（去除遮挡点）
		faceMap.create(imageData.height, imageData.width);
		depthMap.create(imageData.height, imageData.width);
		RasterMesh rasterer(vertices, imageData.camera, depthMap, faceMap);
		rasterer.Clear();
		for (auto idxFace : cameraFaces) {
			const Face& facet = faces[idxFace];
			rasterer.idxFace = idxFace;
			rasterer.Project(facet);
		}
		// compute the projection area of visible faces
		// 计算当前帧可见faces的投影面积
		#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
		CLISTDEF0IDX(uint32_t,FIndex) areas(faces.GetSize());
		areas.Memset(0);
		#endif
		#ifdef TEXOPT_USE_OPENMP
		#pragma omp critical
		#endif
		{
		for (int j=0; j<faceMap.rows; ++j) {
			for (int i=0; i<faceMap.cols; ++i) {
				const FIndex& idxFace = faceMap(j,i);
				ASSERT((idxFace == NO_ID && depthMap(j,i) == 0) || (idxFace != NO_ID && depthMap(j,i) > 0));
				if (idxFace == NO_ID)
					continue;
				FaceDataArr& faceDatas = facesDatas[idxFace];
				#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
				uint32_t& area = areas[idxFace];
				if (area++ == 0) {
				#else
				if (faceDatas.IsEmpty() || faceDatas.Last().idxView != idxView) {
				#endif
					// create new face-data
					// 创建新的face-data
					FaceData& faceData = faceDatas.AddEmpty();
					faceData.idxView = idxView;
					faceData.quality = imageGradMag(j,i);
					#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
					faceData.color = imageData.image(j,i);
					#endif
				} else {
					// update face-data
					// 更新face-data
					ASSERT(!faceDatas.IsEmpty());
					FaceData& faceData = faceDatas.Last();
					ASSERT(faceData.idxView == idxView);
					faceData.quality += imageGradMag(j,i);
					#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
					faceData.color += Color(imageData.image(j,i));
					#endif
				}
			}
		}
		#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
		// 计算颜色均值
		FOREACH(idxFace, areas) {
			const uint32_t& area = areas[idxFace];
			if (area > 0) {
				Color& color = facesDatas[idxFace].Last().color;
				color = RGB2YCBCR(Color(color * (1.f/(float)area)));
			}
		}
		#endif
		}
		++progress;
	}
	#ifdef TEXOPT_USE_OPENMP
	if (bAbort)
		return false;
	#endif
	progress.close();

	#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
	if (fOutlierThreshold > 0) {
		// try to detect outlier views for each face 检测外点view（在场景中face的view被动态物体遮挡比如行人）
		// (views for which the face is occluded by a dynamic object in the scene, ex. pedestrians)
		FOREACHPTR(pFaceDatas, facesDatas)
			FaceOutlierDetection(*pFaceDatas, fOutlierThreshold);
	}
	#endif
	return true;
}

#if TEXOPT_FACEOUTLIER == TEXOPT_FACEOUTLIER_MEDIAN

// decrease the quality of / remove all views in which the face's projection
// has a much different color than in the majority of views
// 如果face在该view的投影与大多views中不同，降低view质量或者直接移除。
bool MeshTexture::FaceOutlierDetection(FaceDataArr& faceDatas, float thOutlier) const
{
	// consider as outlier if the absolute difference to the median is outside this threshold
	// 如果颜色值与中值的差的绝对值大于thOutlier则认为是外点
	if (thOutlier <= 0)
		thOutlier = 0.15f*255.f;

	// init colors array
	// 初始化颜色
	if (faceDatas.GetSize() <= 3)
		return false;
	FloatArr channels[3];
	for (int c=0; c<3; ++c)
		channels[c].Resize(faceDatas.GetSize());
	FOREACH(i, faceDatas) {
		const Color& color = faceDatas[i].color;
		for (int c=0; c<3; ++c)
			channels[c][i] = color[c];
	}

	// find median找中值
	for (int c=0; c<3; ++c)
		channels[c].Sort();
	const unsigned idxMedian(faceDatas.GetSize() >> 1);
	Color median;
	for (int c=0; c<3; ++c)
		median[c] = channels[c][idxMedian];

	// abort if there are not at least 3 inliers
	int nInliers(0);
	BoolArr inliers(faceDatas.GetSize());
	FOREACH(i, faceDatas) {
		const Color& color = faceDatas[i].color;
		for (int c=0; c<3; ++c) {
			if (ABS(median[c]-color[c]) > thOutlier) {
				inliers[i] = false;
				goto CONTINUE_LOOP;
			}
		}
		inliers[i] = true;
		++nInliers;
		CONTINUE_LOOP:;
	}
	if (nInliers == faceDatas.GetSize())
		return true;
	if (nInliers < 3)
		return false;

	// remove outliers
	RFOREACH(i, faceDatas)
		if (!inliers[i])
			faceDatas.RemoveAt(i);
	return true;
}

#elif TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA

// A multi-variate normal distribution which is NOT normalized such that the integral is 1
// - centered is the vector for which the function is to be evaluated with the mean subtracted [Nx1]
// - X is the vector for which the function is to be evaluated [Nx1]
// - mu is the mean around which the distribution is centered [Nx1]
// - covarianceInv is the inverse of the covariance matrix [NxN]
// return exp(-1/2 * (X-mu)^T * covariance_inv * (X-mu))
template <typename T, int N>
inline T MultiGaussUnnormalized(const Eigen::Matrix<T,N,1>& centered, const Eigen::Matrix<T,N,N>& covarianceInv) {
	return EXP(T(-0.5) * T(centered.adjoint() * covarianceInv * centered));
}
template <typename T, int N>
inline T MultiGaussUnnormalized(const Eigen::Matrix<T,N,1>& X, const Eigen::Matrix<T,N,1>& mu, const Eigen::Matrix<T,N,N>& covarianceInv) {
	return MultiGaussUnnormalized<T,N>(X - mu, covarianceInv);
}

// decrease the quality of / remove all views in which the face's projection
// has a much different color than in the majority of views
bool MeshTexture::FaceOutlierDetection(FaceDataArr& faceDatas, float thOutlier) const
{
	// reject all views whose gauss value is below this threshold
	if (thOutlier <= 0)
		thOutlier = 6e-2f;

	const float minCovariance(1e-3f); // if all covariances drop below this the outlier detection aborted

	const unsigned maxIterations(10);
	const unsigned minInliers(4);

	// init colors array
	if (faceDatas.GetSize() <= minInliers)
		return false;
	Eigen::Matrix3Xd colorsAll(3, faceDatas.GetSize());
	BoolArr inliers(faceDatas.GetSize());
	FOREACH(i, faceDatas) {
		colorsAll.col(i) = ((const Color::EVec)faceDatas[i].color).cast<double>();
		inliers[i] = true;
	}

	// perform outlier removal; abort if something goes wrong
	// (number of inliers below threshold or can not invert the covariance)
	size_t numInliers(faceDatas.GetSize());
	Eigen::Vector3d mean;
	Eigen::Matrix3d covariance;
	Eigen::Matrix3d covarianceInv;
	for (unsigned iter = 0; iter < maxIterations; ++iter) {
		// compute the mean color and color covariance only for inliers
		const Eigen::Block<Eigen::Matrix3Xd,3,Eigen::Dynamic,!Eigen::Matrix3Xd::IsRowMajor> colors(colorsAll.leftCols(numInliers));
		mean = colors.rowwise().mean();
		const Eigen::Matrix3Xd centered(colors.colwise() - mean);
		covariance = (centered * centered.transpose()) / double(colors.cols() - 1);

		// stop if all covariances gets very small
		if (covariance.array().abs().maxCoeff() < minCovariance) {
			// remove the outliers
			RFOREACH(i, faceDatas)
				if (!inliers[i])
					faceDatas.RemoveAt(i);
			return true;
		}

		// invert the covariance matrix
		// (FullPivLU not the fastest, but gives feedback about numerical stability during inversion)
		const Eigen::FullPivLU<Eigen::Matrix3d> lu(covariance);
		if (!lu.isInvertible())
			return false;
		covarianceInv = lu.inverse();

		// filter inliers
		// (all views with a gauss value above the threshold)
		numInliers = 0;
		bool bChanged(false);
		FOREACH(i, faceDatas) {
			const Eigen::Vector3d color(((const Color::EVec)faceDatas[i].color).cast<double>());
			const double gaussValue(MultiGaussUnnormalized<double,3>(color, mean, covarianceInv));
			bool& inlier = inliers[i];
			if (gaussValue > thOutlier) {
				// set as inlier
				colorsAll.col(numInliers++) = color;
				if (inlier != true) {
					inlier = true;
					bChanged = true;
				}
			} else {
				// set as outlier
				if (inlier != false) {
					inlier = false;
					bChanged = true;
				}
			}
		}
		if (numInliers == faceDatas.GetSize())
			return true;
		if (numInliers < minInliers)
			return false;
		if (!bChanged)
			break;
	}

	#if TEXOPT_FACEOUTLIER == TEXOPT_FACEOUTLIER_GAUSS_DAMPING
	// select the final inliers
	const float factorOutlierRemoval(0.2f);
	covarianceInv *= factorOutlierRemoval;
	RFOREACH(i, faceDatas) {
		const Eigen::Vector3d color(((const Color::EVec)faceDatas[i].color).cast<double>());
		const double gaussValue(MultiGaussUnnormalized<double,3>(color, mean, covarianceInv));
		ASSERT(gaussValue >= 0 && gaussValue <= 1);
		faceDatas[i].quality *= gaussValue;
	}
	#endif
	#if TEXOPT_FACEOUTLIER == TEXOPT_FACEOUTLIER_GAUSS_CLAMPING
	// remove outliers
	RFOREACH(i, faceDatas)
		if (!inliers[i])
			faceDatas.RemoveAt(i);
	#endif
	return true;
}
#endif
/**
 * @brief  给每个face（三角网格）分配最佳视图view 
 * 
 * 如果face在该view的投影与大多views中不同，降低view质量或者直接移除。
 * @param[in] fOutlierThreshold     颜色差异阈值，用于face选择投影的view时，剔除与大多view不同的外点view
 * @param[in] fRatioDataSmoothness  控制平滑程度，越大越平滑
 * @return true 
 * @return false 
 */
bool MeshTexture::FaceViewSelection(float fOutlierThreshold, float fRatioDataSmoothness)
{
	// extract array of triangles incident to each vertex
	// 提取顶点所在的所有faces
	ListVertexFaces();

	// create texture patches
	// 创建纹理块
	{
		// list all views for each face
		// 列出每个face能被看到的所有views,记录看到该face的所有views信息
		FaceDataViewArr facesDatas;
		if (!ListCameraFaces(facesDatas, fOutlierThreshold))
			return false;

		// create faces graph
		// 创建以face为节点的无向图
		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
		typedef boost::graph_traits<Graph>::edge_iterator EdgeIter;
		typedef boost::graph_traits<Graph>::out_edge_iterator EdgeOutIter;
		Graph graph;
		{
			FOREACH(idxFace, faces) {
				const Mesh::FIndex idx((Mesh::FIndex)boost::add_vertex(graph));
				ASSERT(idx == idxFace);
			}
			Mesh::FaceIdxArr afaces;
			FOREACH(idxFace, faces) {
				// 取face的三个相邻faces，如果在边界可能只有1个或者2个
				scene.mesh.GetFaceFaces(idxFace, afaces);
				ASSERT(ISINSIDE((int)afaces.GetSize(), 1, 4));
				FOREACHPTR(pIdxFace, afaces) {
					const FIndex idxFaceAdj = *pIdxFace;
					// 前面face已经处理过
					if (idxFace >= idxFaceAdj)
						continue;
					const bool bInvisibleFace(facesDatas[idxFace].IsEmpty());
					const bool bInvisibleFaceAdj(facesDatas[idxFaceAdj].IsEmpty());
					// 如果当前face和邻域face都没有可见的view则跳过
					if (bInvisibleFace || bInvisibleFaceAdj) {
						if (bInvisibleFace != bInvisibleFaceAdj)
							seamEdges.AddConstruct(idxFace, idxFaceAdj);
						continue;
					}
					boost::add_edge(idxFace, idxFaceAdj, graph);
				}
				afaces.Empty();
			}
			ASSERT((Mesh::FIndex)boost::num_vertices(graph) == faces.GetSize());
		}

		// assign the best view to each face
		// 给每个面找一个最好的view
		LabelArr labels(faces.GetSize());
		components.Resize(faces.GetSize());
		{
			// find connected components
			// 计算连通域个数是nComponents，每个face i所在的连通域是components[i]
			const FIndex nComponents(boost::connected_components(graph, components.Begin()));

			// map face ID from global to component space
			// 将face的id从全局转到单个component空间
			typedef cList<NodeID, NodeID, 0, 128, NodeID> NodeIDs;
			NodeIDs nodeIDs(faces.GetSize()); // 存储节点在component中的新id
			NodeIDs sizes(nComponents);// 记录每个face在对应component中的新id
			sizes.Memset(0);
			FOREACH(c, components)
				nodeIDs[c] = sizes[components[c]]++;

			// normalize quality values
			// 归一化质量值
			float maxQuality(0); // 计算最大质量值
			FOREACHPTR(pFaceDatas, facesDatas) {
				const FaceDataArr& faceDatas = *pFaceDatas;
				FOREACHPTR(pFaceData, faceDatas)
					if (maxQuality < pFaceData->quality)
						maxQuality = pFaceData->quality;
			}
			Histogram32F hist(std::make_pair(0.f, maxQuality), 1000);
			FOREACHPTR(pFaceDatas, facesDatas) {
				const FaceDataArr& faceDatas = *pFaceDatas;
				FOREACHPTR(pFaceData, faceDatas)
					hist.Add(pFaceData->quality);
			}
			// 归一化质量
			const float normQuality(hist.GetApproximatePermille(0.95f));
			// 马尔可夫随机场（概率无向图模型）解决labeling问题，优化方法LBP（loopy belief propagation algorithm）
			// 每个face都有k个标签，要求解的问题是选择一种标签组合使该方案发生的概率最大可以转化为最小能量求解问题：
			// 找到一组标签组合使得最终的cost最小 min(E)=Σc(xs)+Σc(xs,xt) xs是所有face（node）,xt是face的邻域
			// 具体有关马尔可夫介绍见课件
			#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_LBP
			// initialize inference structures
			// Step 1 初始化inference,设置邻域和节点
			CLISTDEFIDX(LBPInference,FIndex) inferences(nComponents);
			{
				FOREACH(s, sizes) {
					const NodeID numNodes(sizes[s]);
					ASSERT(numNodes > 0);
					if (numNodes <= 1)
						continue;
					LBPInference& inference = inferences[s];
					inference.SetNumNodes(numNodes);
					// 设置平滑cost,如果两个节点标签相同则cost=0,否则为MaxEnergy
					// 目的是让相邻face的标签尽可能一致
					inference.SetSmoothCost(SmoothnessPotts);
				}
				EdgeOutIter ei, eie;
				FOREACH(f, faces) {
					LBPInference& inference = inferences[components[f]];
					// 添加edge即每个face的邻域face
					for (boost::tie(ei, eie) = boost::out_edges(f, graph); ei != eie; ++ei) {
						ASSERT(f == (FIndex)ei->m_source);
						const FIndex fAdj((FIndex)ei->m_target);
						ASSERT(components[f] == components[fAdj]);
						// 确保每个edge只添加一次
						if (f < fAdj) // add edges only once
							inference.SetNeighbors(nodeIDs[f], nodeIDs[fAdj]);
					}
				}
			}

			// set data costs
			// Step 2 设置node的每个标签对应的cost 
			{
				const LBPInference::EnergyType MaxEnergy(fRatioDataSmoothness*LBPInference::MaxEnergy);
				// set costs for label 0 (undefined)
				// 如果是未分配label的face设为0 cost为固定值MaxEnergy
				FOREACH(s, inferences) {
					LBPInference& inference = inferences[s];
					if (inference.GetNumNodes() == 0)
						continue;
					const NodeID numNodes(sizes[s]);
					for (NodeID nodeID=0; nodeID<numNodes; ++nodeID)
						inference.SetDataCost((Label)0, nodeID, MaxEnergy);
				}
				// set data costs for all labels (except label 0 - undefined)
				// 设置face 的每个label的cost
				FOREACH(f, facesDatas) {
					LBPInference& inference = inferences[components[f]];
					if (inference.GetNumNodes() == 0)
						continue;
					const FaceDataArr& faceDatas = facesDatas[f];
					const NodeID nodeID(nodeIDs[f]);
					FOREACHPTR(pFaceData, faceDatas) {
						const FaceData& faceData = *pFaceData;
						// 有效标签从1开始，因为0是无标签的标记
						const Label label((Label)faceData.idxView+1);
						// 归一化
						const float normalizedQuality(faceData.quality>=normQuality ? 1.f : faceData.quality/normQuality);
						// cost计算，质量越好代价越小
						const float dataCost((1.f-normalizedQuality)*MaxEnergy);
						inference.SetDataCost(label, nodeID, dataCost);
					}
				}
			}

			// assign the optimal view (label) to each face
			// (label 0 is reserved as undefined)
			// Step 3 调用能量最小优化函数
			FOREACH(s, inferences) {
				LBPInference& inference = inferences[s];
				if (inference.GetNumNodes() == 0)
					continue;
				inference.Optimize();
			}
			// extract resulting labeling
			// Step 4 提取labeling的结果
			labels.Memset(0xFF);
			FOREACH(l, labels) {
				LBPInference& inference = inferences[components[l]];
				if (inference.GetNumNodes() == 0)
					continue;
				const Label label(inference.GetLabel(nodeIDs[l]));
				ASSERT(label < images.GetSize()+1);
				if (label > 0)
				//  注意-1 ，有效是从1开始的
					labels[l] = label-1;
			}
			#endif
			// TRWS与LBP调用类似不再赘述
			#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_TRWS
			// initialize inference structures
			const LabelID numLabels(images.GetSize()+1);
			CLISTDEFIDX(TRWSInference, FIndex) inferences(nComponents);
			FOREACH(s, sizes) {
				const NodeID numNodes(sizes[s]);
				ASSERT(numNodes > 0);
				if (numNodes <= 1)
					continue;
				TRWSInference& inference = inferences[s];
				inference.Init(numNodes, numLabels);
			}

			// set data costs
			{
				// add nodes
				CLISTDEF0(EnergyType) D(numLabels);
				FOREACH(f, facesDatas) {
					TRWSInference& inference = inferences[components[f]];
					if (inference.IsEmpty())
						continue;
					D.MemsetValue(MaxEnergy);
					const FaceDataArr& faceDatas = facesDatas[f];
					FOREACHPTR(pFaceData, faceDatas) {
						const FaceData& faceData = *pFaceData;
						const Label label((Label)faceData.idxView);
						const float normalizedQuality(faceData.quality>=normQuality ? 1.f : faceData.quality/normQuality);
						const EnergyType dataCost(MaxEnergy*(1.f-normalizedQuality));
						D[label] = dataCost;
					}
					const NodeID nodeID(nodeIDs[f]);
					inference.AddNode(nodeID, D.Begin());
				}
				// add edges
				EdgeOutIter ei, eie;
				FOREACH(f, faces) {
					TRWSInference& inference = inferences[components[f]];
					if (inference.IsEmpty())
						continue;
					for (boost::tie(ei, eie) = boost::out_edges(f, graph); ei != eie; ++ei) {
						ASSERT(f == (FIndex)ei->m_source);
						const FIndex fAdj((FIndex)ei->m_target);
						ASSERT(components[f] == components[fAdj]);
						if (f < fAdj) // add edges only once
							inference.AddEdge(nodeIDs[f], nodeIDs[fAdj]);
					}
				}
			}

			// assign the optimal view (label) to each face
			#ifdef TEXOPT_USE_OPENMP
			#pragma omp parallel for schedule(dynamic)
			for (int i=0; i<(int)inferences.GetSize(); ++i) {
			#else
			FOREACH(i, inferences) {
			#endif
				TRWSInference& inference = inferences[i];
				if (inference.IsEmpty())
					continue;
				inference.Optimize();
			}
			// extract resulting labeling
			// 提取label
			labels.Memset(0xFF);
			FOREACH(l, labels) {
				TRWSInference& inference = inferences[components[l]];
				if (inference.IsEmpty())
					continue;
				const Label label(inference.GetLabel(nodeIDs[l]));
				ASSERT(label >= 0 && label < numLabels);
				if (label < images.GetSize())
					labels[l] = label;
			}
			#endif
		}

		// create texture patches
		// 创建纹理块
		{
			// divide graph in sub-graphs of connected faces having the same label
			// 划分有相同label的face
			EdgeIter ei, eie;
			const PairIdxArr::IDX startLabelSeamEdges(seamEdges.GetSize());
			for (boost::tie(ei, eie) = boost::edges(graph); ei != eie; ++ei) {
				const FIndex fSource((FIndex)ei->m_source);
				const FIndex fTarget((FIndex)ei->m_target);
				ASSERT(components[fSource] == components[fTarget]);
				// 统计label不同的edge
				if (labels[fSource] != labels[fTarget])
					seamEdges.AddConstruct(fSource, fTarget);
			}
			// 将graph中label不同的边剔除
			for (const PairIdx *pEdge=seamEdges.Begin()+startLabelSeamEdges, *pEdgeEnd=seamEdges.End(); pEdge!=pEdgeEnd; ++pEdge)
				boost::remove_edge(pEdge->i, pEdge->j, graph);

			// find connected components: texture patches
			// 重新查找连通块即为纹理块
			ASSERT((FIndex)boost::num_vertices(graph) == components.GetSize());
			const FIndex nComponents(boost::connected_components(graph, components.Begin()));

			// create texture patches;
			// last texture patch contains all faces with no texture
			// 构建纹理块，最后一个patch存放所有没有label的faces
			LabelArr sizes(nComponents);
			sizes.Memset(0);
			FOREACH(c, components)
				++sizes[components[c]];
			// +1增加一个Patch存放无纹理(label)的faces
			texturePatches.Resize(nComponents+1);
			texturePatches.Last().label = NO_ID;
			FOREACH(f, faces) {
				const Label label(labels[f]);
				const FIndex c(components[f]);
				TexturePatch& texturePatch = texturePatches[c];
				ASSERT(texturePatch.label == label || texturePatch.faces.IsEmpty());
				if (label == NO_ID) {
					texturePatch.label = NO_ID;
					texturePatches.Last().faces.Insert(f);
				} else {
					if (texturePatch.faces.IsEmpty()) {
						texturePatch.label = label;
						texturePatch.faces.Reserve(sizes[c]);
					}
					texturePatch.faces.Insert(f);
				}
			}
			// remove all patches with invalid label (except the last one)
			// and create the map from the old index to the new one
			// 移除所有patch中无效的label,同时记录顶点的新旧id
			mapIdxPatch.Resize(nComponents);
			// mapIdxPatch:0,1,2...nComponents-1
			std::iota(mapIdxPatch.Begin(), mapIdxPatch.End(), 0);
			for (FIndex t = nComponents; t-- > 0; ) {
				if (texturePatches[t].label == NO_ID) {
					texturePatches.RemoveAtMove(t);
					mapIdxPatch.RemoveAtMove(t);
				}
			}
			const unsigned numPatches(texturePatches.GetSize()-1);
			uint32_t idxPatch(0);
			for (IndexArr::IDX i=0; i<mapIdxPatch.GetSize(); ++i) {
				while (i < mapIdxPatch[i])
					mapIdxPatch.InsertAt(i++, numPatches);
				mapIdxPatch[i] = idxPatch++;
			}
			while (mapIdxPatch.GetSize() <= nComponents)
				mapIdxPatch.Insert(numPatches);
		}
	}
	return true;
}


// create seam vertices and edges
// 创建接缝的顶点和边
void MeshTexture::CreateSeamVertices()
{
	// each vertex will contain the list of patches it separates,
	// except the patch containing invisible faces;
	// each patch contains the list of edges belonging to that texture patch, starting from that vertex
	// (usually there are pairs of edges in each patch, representing the two edges starting from that vertex separating two valid patches)
	// 每个顶点包含由它分割的patches（除去包含不可见faces的patch）
	// 每个patch包含属于纹理patch的以上述顶点起始的edge
	VIndex vs[2];
	uint32_t vs0[2], vs1[2];
	std::unordered_map<VIndex, uint32_t> mapVertexSeam;
	const unsigned numPatches(texturePatches.GetSize()-1);
	FOREACHPTR(pEdge, seamEdges) {
		// store edge for the later seam optimization
		// 存储edge
		ASSERT(pEdge->i < pEdge->j);
		const uint32_t idxPatch0(mapIdxPatch[components[pEdge->i]]);
		const uint32_t idxPatch1(mapIdxPatch[components[pEdge->j]]);
		ASSERT(idxPatch0 != idxPatch1 || idxPatch0 == numPatches);
		if (idxPatch0 == idxPatch1)
			continue;
		seamVertices.ReserveExtra(2);
		scene.mesh.GetEdgeVertices(pEdge->i, pEdge->j, vs0, vs1);
		ASSERT(faces[pEdge->i][vs0[0]] == faces[pEdge->j][vs1[0]]);
		ASSERT(faces[pEdge->i][vs0[1]] == faces[pEdge->j][vs1[1]]);
		vs[0] = faces[pEdge->i][vs0[0]];
		vs[1] = faces[pEdge->i][vs0[1]];

		const auto itSeamVertex0(mapVertexSeam.emplace(std::make_pair(vs[0], seamVertices.GetSize())));
		if (itSeamVertex0.second)
			seamVertices.AddConstruct(vs[0]);
		SeamVertex& seamVertex0 = seamVertices[itSeamVertex0.first->second];

		const auto itSeamVertex1(mapVertexSeam.emplace(std::make_pair(vs[1], seamVertices.GetSize())));
		if (itSeamVertex1.second)
			seamVertices.AddConstruct(vs[1]);
		SeamVertex& seamVertex1 = seamVertices[itSeamVertex1.first->second];

		if (idxPatch0 < numPatches) {
			const TexCoord offset0(texturePatches[idxPatch0].rect.tl());
			SeamVertex::Patch& patch00 = seamVertex0.GetPatch(idxPatch0);
			SeamVertex::Patch& patch10 = seamVertex1.GetPatch(idxPatch0);
			ASSERT(patch00.edges.Find(itSeamVertex1.first->second) == NO_ID);
			patch00.edges.AddConstruct(itSeamVertex1.first->second).idxFace = pEdge->i;
			patch00.proj = faceTexcoords[pEdge->i*3+vs0[0]]+offset0;
			ASSERT(patch10.edges.Find(itSeamVertex0.first->second) == NO_ID);
			patch10.edges.AddConstruct(itSeamVertex0.first->second).idxFace = pEdge->i;
			patch10.proj = faceTexcoords[pEdge->i*3+vs0[1]]+offset0;
		}
		if (idxPatch1 < numPatches) {
			const TexCoord offset1(texturePatches[idxPatch1].rect.tl());
			SeamVertex::Patch& patch01 = seamVertex0.GetPatch(idxPatch1);
			SeamVertex::Patch& patch11 = seamVertex1.GetPatch(idxPatch1);
			ASSERT(patch01.edges.Find(itSeamVertex1.first->second) == NO_ID);
			patch01.edges.AddConstruct(itSeamVertex1.first->second).idxFace = pEdge->j;
			patch01.proj = faceTexcoords[pEdge->j*3+vs1[0]]+offset1;
			ASSERT(patch11.edges.Find(itSeamVertex0.first->second) == NO_ID);
			patch11.edges.AddConstruct(itSeamVertex0.first->second).idxFace = pEdge->j;
			patch11.proj = faceTexcoords[pEdge->j*3+vs1[1]]+offset1;
		}
	}
	seamEdges.Release();
}
/**
 * @brief 全局颜色校正
 * 
 */
void MeshTexture::GlobalSeamLeveling()
{
	ASSERT(!seamVertices.IsEmpty());
	// 减一是最后一个patch是存放无label的faces
	const unsigned numPatches(texturePatches.GetSize()-1);
	// Step 1 数据准备：记录每个顶点的patch id,并标记在patches间边界处的顶点
	// find the patch ID for each vertex
	// 存放每个顶点的patch id,如果顶点在seam上记录其在seamVertices中的id
	PatchIndices patchIndices(vertices.GetSize());
	patchIndices.Memset(0);
	FOREACH(f, faces) {
		const uint32_t idxPatch(mapIdxPatch[components[f]]);
		const Face& face = faces[f];
		for (int v=0; v<3; ++v)
			patchIndices[face[v]].idxPatch = idxPatch;
	}
	// 记录顶点在seamVertex的id
	FOREACH(i, seamVertices) {
		const SeamVertex& seamVertex = seamVertices[i];
		ASSERT(!seamVertex.patches.IsEmpty());
		PatchIndex& patchIndex = patchIndices[seamVertex.idxVertex];
		// 标记该顶点是否在纹理接缝上
		patchIndex.bIndex = true; 
		patchIndex.idxSeamVertex = i;
	}

	// assign a row index within the solution vector x to each vertex/patch
	// Step 2 分配一个行索引给每个顶点 后续构建索引
	ASSERT(vertices.GetSize() < static_cast<VIndex>(std::numeric_limits<MatIdx>::max()));
	MatIdx rowsX(0);
	typedef std::unordered_map<uint32_t,MatIdx> VertexPatch2RowMap;
	cList<VertexPatch2RowMap> vertpatch2rows(vertices.GetSize());
	FOREACH(i, vertices) {
		const PatchIndex& patchIndex = patchIndices[i];
		VertexPatch2RowMap& vertpatch2row = vertpatch2rows[i];
		if (patchIndex.bIndex) {
			// vertex is part of multiple patches
			// 顶点是多个patch的一部分
			const SeamVertex& seamVertex = seamVertices[patchIndex.idxSeamVertex];
			ASSERT(seamVertex.idxVertex == i);
			FOREACHPTR(pPatch, seamVertex.patches) {
				ASSERT(pPatch->idxPatch != numPatches);
				vertpatch2row[pPatch->idxPatch] = rowsX++;
			}
		} else
		if (patchIndex.idxPatch < numPatches) {
			// vertex is part of only one patch
			// 顶点是一个patch的一部分
			vertpatch2row[patchIndex.idxPatch] = rowsX++;
		}
	}
	// 参考论文Let There Be Color! Large-Scale Texturing of 3D Reconstructions公式2，3 min(g_t(A_t*A+Gamma_t*Gamma)g-2coeffB_t*A*g)+coeffB_t*coeffB
	// 求最小值，对g求导令其导数为0则：(A_t*A+Gamma_t*Gamma)g=A_t*coeffB
	// fill Tikhonov's Gamma matrix (regularization constraints)
	const float lambda(0.1f);
	MatIdx rowsGamma(0);
	Mesh::VertexIdxArr adjVerts;
	CLISTDEF0(MatEntry) rows(0, vertices.GetSize()*4);
	FOREACH(v, vertices) {
		adjVerts.Empty();
		scene.mesh.GetAdjVertices(v, adjVerts);
		VertexPatchIterator itV(patchIndices[v], seamVertices);
		while (itV.Next()) {
			const uint32_t idxPatch(itV);
			if (idxPatch == numPatches)
				continue;
			const MatIdx col(vertpatch2rows[v].at(idxPatch));
			FOREACHPTR(pAdjVert, adjVerts) {
				const VIndex vAdj(*pAdjVert);
				if (v >= vAdj)
					continue;
				VertexPatchIterator itVAdj(patchIndices[vAdj], seamVertices);
				while (itVAdj.Next()) {
					const uint32_t idxPatchAdj(itVAdj);
					if (idxPatch == idxPatchAdj) {
						const MatIdx colAdj(vertpatch2rows[vAdj].at(idxPatchAdj));
						rows.AddConstruct(rowsGamma, col, lambda);
						rows.AddConstruct(rowsGamma, colAdj, -lambda);
						++rowsGamma;
					}
				}
			}
		}
	}
	ASSERT(rows.GetSize()/2 < static_cast<IDX>(std::numeric_limits<MatIdx>::max()));

	SparseMat Gamma(rowsGamma, rowsX);
	Gamma.setFromTriplets(rows.Begin(), rows.End());
	rows.Empty();

	// fill the matrix A and the coefficients for the Vector b of the linear equation system
	// (A_t*A+Gamma_t*Gamma)g=A_t*coeffB
	// 计算A矩阵和b构建ax=b
	IndexArr indices;
	Colors vertexColors;
	Colors coeffB;
	FOREACHPTR(pSeamVertex, seamVertices) {
		const SeamVertex& seamVertex = *pSeamVertex;
		if (seamVertex.patches.GetSize() < 2)
			continue;
		seamVertex.SortByPatchIndex(indices);
		vertexColors.Resize(indices.GetSize());
		FOREACH(i, indices) {
			const SeamVertex::Patch& patch0 = seamVertex.patches[indices[i]];
			ASSERT(patch0.idxPatch < numPatches);
			SampleImage sampler(images[texturePatches[patch0.idxPatch].label].image);
			FOREACHPTR(pEdge, patch0.edges) {
				const SeamVertex& seamVertex1 = seamVertices[pEdge->idxSeamVertex];
				const SeamVertex::Patches::IDX idxPatch1(seamVertex1.patches.Find(patch0.idxPatch));
				ASSERT(idxPatch1 != SeamVertex::Patches::NO_INDEX);
				const SeamVertex::Patch& patch1 = seamVertex1.patches[idxPatch1];
				sampler.AddEdge(patch0.proj, patch1.proj);
			}
			vertexColors[i] = sampler.GetColor();
		}
		const VertexPatch2RowMap& vertpatch2row = vertpatch2rows[seamVertex.idxVertex];
		for (IDX i=0; i<indices.GetSize()-1; ++i) {
			const uint32_t idxPatch0(seamVertex.patches[indices[i]].idxPatch);
			const Color& color0 = vertexColors[i];
			const MatIdx col0(vertpatch2row.at(idxPatch0));
			for (IDX j=i+1; j<indices.GetSize(); ++j) {
				const uint32_t idxPatch1(seamVertex.patches[indices[j]].idxPatch);
				const Color& color1 = vertexColors[j];
				const MatIdx col1(vertpatch2row.at(idxPatch1));
				ASSERT(idxPatch0 < idxPatch1);
				const MatIdx rowA((MatIdx)coeffB.GetSize());
				coeffB.Insert(color1 - color0);
				ASSERT(ISFINITE(coeffB.Last()));
				rows.AddConstruct(rowA, col0,  1.f);
				rows.AddConstruct(rowA, col1, -1.f);
			}
		}
	}
	ASSERT(coeffB.GetSize() < static_cast<IDX>(std::numeric_limits<MatIdx>::max()));

	const MatIdx rowsA((MatIdx)coeffB.GetSize());
	SparseMat A(rowsA, rowsX);
	A.setFromTriplets(rows.Begin(), rows.End());
	rows.Release();

	SparseMat Lhs(A.transpose() * A + Gamma.transpose() * Gamma);
	// CG uses only the lower triangle, so prune the rest and compress matrix
	// CG 仅使用下三角数据
	Lhs.prune([](const int& row, const int& col, const float&) -> bool {
		return col <= row;
	});

	// globally solve for the correction colors
	// 求解g
	Eigen::Matrix<float,Eigen::Dynamic,3,Eigen::RowMajor> colorAdjustments(rowsX, 3);
	{
		// init CG 
		// 设置误差容忍度和迭代次数
		Eigen::ConjugateGradient<SparseMat, Eigen::Lower> solver;
		solver.setMaxIterations(1000);
		solver.setTolerance(0.0001f);
		solver.compute(Lhs); 
		ASSERT(solver.info() == Eigen::Success);
		#ifdef TEXOPT_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int channel=0; channel<3; ++channel) {
			// init right hand side vector
			// 初始化向量b
			const Eigen::Map< Eigen::VectorXf, Eigen::Unaligned, Eigen::Stride<0,3> > b(coeffB.Begin()->ptr()+channel, rowsA);
			const Eigen::VectorXf Rhs(SparseMat(A.transpose()) * b);
			// solve for x
			// 求解x
			const Eigen::VectorXf x(solver.solve(Rhs));
			ASSERT(solver.info() == Eigen::Success);
			// subtract mean since the system is under-constrained and
			// we need the solution with minimal adjustments
			// 减去均值是因为系统是无约束的
			Eigen::Map< Eigen::VectorXf, Eigen::Unaligned, Eigen::Stride<0,3> >(colorAdjustments.data()+channel, rowsX) = x.array() - x.mean();
			DEBUG_LEVEL(3, "\tcolor channel %d: %d iterations, %g residual", channel, solver.iterations(), solver.error());
		}
	}

	// adjust texture patches using the correction colors
	// 调整patch
	#ifdef TEXOPT_USE_OPENMP
	#pragma omp parallel for schedule(dynamic)
	for (int i=0; i<(int)numPatches; ++i) {
	#else
	for (unsigned i=0; i<numPatches; ++i) {
	#endif
		const uint32_t idxPatch((uint32_t)i);
		TexturePatch& texturePatch = texturePatches[idxPatch];
		ColorMap imageAdj(texturePatch.rect.size());
		imageAdj.memset(0);
		// interpolate color adjustments over the whole patch
		// 整个patch插值调整颜色
		RasterPatchColorData data(imageAdj);
		FOREACHPTR(pIdxFace, texturePatch.faces) {
			const FIndex idxFace(*pIdxFace);
			const Face& face = faces[idxFace];
			data.tri = faceTexcoords.Begin()+idxFace*3;
			for (int v=0; v<3; ++v)
				data.colors[v] = colorAdjustments.row(vertpatch2rows[face[v]].at(idxPatch));
			// render triangle and for each pixel interpolate the color adjustment
			// from the triangle corners using barycentric coordinates
			// 利用重心坐标插值三角内的像素颜色调整值。
			ColorMap::RasterizeTriangle(data.tri[0], data.tri[1], data.tri[2], data);
		}
		// dilate with one pixel width, in order to make sure patch border smooths out a little
		// 膨胀一个像素，确保patch边界平滑。
		imageAdj.DilateMean<1>(imageAdj, Color::ZERO);
		// apply color correction to the patch image
		cv::Mat image(images[texturePatch.label].image(texturePatch.rect));
		for (int r=0; r<image.rows; ++r) {
			for (int c=0; c<image.cols; ++c) {
				const Color& a = imageAdj(r,c);
				if (a == Color::ZERO)
					continue;
				Pixel8U& v = image.at<Pixel8U>(r,c);
				const Color col(RGB2YCBCR(Color(v)));
				const Color acol(YCBCR2RGB(Color(col+a)));
				for (int p=0; p<3; ++p)
					v[p] = (uint8_t)CLAMP(ROUND2INT(acol[p]), 0, 255);
			}
		}
	}
}

// set to one in order to dilate also on the diagonal of the border
// 设置为1以便也在边界的对角线上扩展
// (normally not needed)
#define DILATE_EXTRA 0
/**
 * @brief 处理mask,局部融合只处理从mask边界往里stripWidth个像素的宽度（即是一个条状mask）
 * 
 * @param[in/out] mask  输入也是输出，是patch对应的mask
 * @param[in] stripWidth mask的宽度
 */
void MeshTexture::ProcessMask(Image8U& mask, int stripWidth)
{
	typedef Image8U::Type Type;

	// dilate and erode around the border,
	// in order to fill all gaps and remove outside pixels
	// 膨胀腐蚀边界。为了填充沟和移除外点像素
	// (due to imperfect overlay of the raster line border and raster faces)
	#define DILATEDIR(rd,cd) { \
		Type& vi = mask(r+(rd),c+(cd)); \
		if (vi != border) \
			vi = interior; \
	}
	const int HalfSize(1);
	const int RowsEnd(mask.rows-HalfSize);
	const int ColsEnd(mask.cols-HalfSize);
	for (int r=HalfSize; r<RowsEnd; ++r) {
		for (int c=HalfSize; c<ColsEnd; ++c) {
			const Type v(mask(r,c));
			if (v != border)
				continue;
			#if DILATE_EXTRA
			for (int i=-HalfSize; i<=HalfSize; ++i) {
				const int rw(r+i);
				for (int j=-HalfSize; j<=HalfSize; ++j) {
					const int cw(c+j);
					Type& vi = mask(rw,cw);
					if (vi != border)
						vi = interior;
				}
			}
			#else
			DILATEDIR(-1, 0);
			DILATEDIR(1, 0);
			DILATEDIR(0, -1);
			DILATEDIR(0, 1);
			#endif
		}
	}
	#undef DILATEDIR
	#define ERODEDIR(rd,cd) { \
		const int rl(r-(rd)), cl(c-(cd)), rr(r+(rd)), cr(c+(cd)); \
		const Type vl(mask.isInside(ImageRef(cl,rl)) ? mask(rl,cl) : uint8_t(empty)); \
		const Type vr(mask.isInside(ImageRef(cr,rr)) ? mask(rr,cr) : uint8_t(empty)); \
		if ((vl == border && vr == empty) || (vr == border && vl == empty)) { \
			v = empty; \
			continue; \
		} \
	}
	#if DILATE_EXTRA
	for (int i=0; i<2; ++i)
	#endif
	for (int r=0; r<mask.rows; ++r) {
		for (int c=0; c<mask.cols; ++c) {
			Type& v = mask(r,c);
			if (v != interior)
				continue;
			ERODEDIR(0, 1);
			ERODEDIR(1, 0);
			ERODEDIR(1, 1);
			ERODEDIR(-1, 1);
		}
	}
	#undef ERODEDIR

	// mark all interior pixels with empty neighbors as border
	// 标记所有内点中邻域是空的像素为边界
	for (int r=0; r<mask.rows; ++r) {
		for (int c=0; c<mask.cols; ++c) {
			Type& v = mask(r,c);
			if (v != interior)
				continue;
			if (mask(r-1,c) == empty ||
				mask(r,c-1) == empty ||
				mask(r+1,c) == empty ||
				mask(r,c+1) == empty)
				v = border;
		}
	}

	#if 0
	// mark all interior pixels with border neighbors on two sides as border
	{
	Image8U orgMask;
	mask.copyTo(orgMask);
	for (int r=0; r<mask.rows; ++r) {
		for (int c=0; c<mask.cols; ++c) {
			Type& v = mask(r,c);
			if (v != interior)
				continue;
			if ((orgMask(r+1,c+0) == border && orgMask(r+0,c+1) == border) ||
				(orgMask(r+1,c+0) == border && orgMask(r-0,c-1) == border) ||
				(orgMask(r-1,c-0) == border && orgMask(r+0,c+1) == border) ||
				(orgMask(r-1,c-0) == border && orgMask(r-0,c-1) == border))
				v = border;
		}
	}
	}
	#endif

	// compute the set of valid pixels at the border of the texture patch
	// 计算有效像素集合
	#define ISEMPTY(mask, x,y) (mask(y,x) == empty)
	const int width(mask.width()), height(mask.height());
	typedef std::unordered_set<ImageRef> PixelSet;
	PixelSet borderPixels;
	for (int y=0; y<height; ++y) {
		for (int x=0; x<width; ++x) {
			if (ISEMPTY(mask, x,y))
				continue;
			// valid border pixels need no invalid neighbors
			// 有效的边界像素不需要无效的邻域
			if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
				borderPixels.insert(ImageRef(x,y));
				continue;
			}
			// check the direct neighborhood of all invalid pixels
			// 检查所有无效像素的直接邻域
			for (int j=-1; j<=1; ++j) {
				for (int i=-1; i<=1; ++i) {
					// if the valid pixel has an invalid neighbor...
					const int xn(x+i), yn(y+j);
					if (ISINSIDE(xn, 0, width) &&
						ISINSIDE(yn, 0, height) &&
						ISEMPTY(mask, xn,yn)) {
						// add the pixel to the set of valid border pixels
						// 将该像素添加到有效的边界像素集
						borderPixels.insert(ImageRef(x,y));
						goto CONTINUELOOP;
					}
				}
			}
			CONTINUELOOP:;
		}
	}

	// iteratively erode all border pixels
	// 迭代腐蚀所有边界像素
	{
	Image8U orgMask;
	mask.copyTo(orgMask);
	typedef std::vector<ImageRef> PixelVector;
	for (int s=0; s<stripWidth; ++s) {
		PixelVector emptyPixels(borderPixels.begin(), borderPixels.end());
		borderPixels.clear();
		// mark the new empty pixels as empty in the mask
		// 在mask中将新的空像素标记为空
		for (PixelVector::const_iterator it=emptyPixels.cbegin(); it!=emptyPixels.cend(); ++it)
			orgMask(*it) = empty;
		// find the set of valid pixels at the border of the valid area
		// 在有效区域的边界上找到有效像素集
		for (PixelVector::const_iterator it=emptyPixels.cbegin(); it!=emptyPixels.cend(); ++it) {
			for (int j=-1; j<=1; j++) {
				for (int i=-1; i<=1; i++) {
					const int xn(it->x+i), yn(it->y+j);
					if (ISINSIDE(xn, 0, width) &&
						ISINSIDE(yn, 0, height) &&
						!ISEMPTY(orgMask, xn, yn))
						borderPixels.insert(ImageRef(xn,yn));
				}
			}
		}
	}
	#undef ISEMPTY

	// mark all remaining pixels empty in the mask
	// 将mask中剩余的像素标记为空
	for (int y=0; y<height; ++y) {
		for (int x=0; x<width; ++x) {
			if (orgMask(y,x) != empty)
				mask(y,x) = empty;
		}
	}
	}

	// mark all border pixels 标记所有的边界像素
	for (PixelSet::const_iterator it=borderPixels.cbegin(); it!=borderPixels.cend(); ++it)
		mask(*it) = border;

	#if 0
	// dilate border 
	// 边界膨胀
	{
	Image8U orgMask;
	mask.copyTo(orgMask);
	for (int r=HalfSize; r<RowsEnd; ++r) {
		for (int c=HalfSize; c<ColsEnd; ++c) {
			const Type v(orgMask(r, c));
			if (v != border)
				continue;
			for (int i=-HalfSize; i<=HalfSize; ++i) {
				const int rw(r+i);
				for (int j=-HalfSize; j<=HalfSize; ++j) {
					const int cw(c+j);
					Type& vi = mask(rw, cw);
					if (vi == empty)
						vi = border;
				}
			}
		}
	}
	}
	#endif
}

inline MeshTexture::Color ColorLaplacian(const Image32F3& img, int i) {
	const int width(img.width());
	return img(i-width) + img(i-1) + img(i+1) + img(i+width) - img(i)*4.f;
}

// 泊松融合参考博客介绍https://blog.csdn.net/hjimce/article/details/45716603  参考论文Poisson Image Editing
/**
 * @brief 泊松融合用来校正纹理接缝处点的颜色差异，泊松融合方程见课件推导
 * 泊松方程：Ax=b  A 是稀疏矩阵（对应边界的是1 对应内部是Laplace算子） x是mask对应的纹理校正后的颜色值 b是边界border颜色和中间点inter的散度值
 * @param[in] src 纹理块对应的图像
 * @param[in] dst 纹理块对应的图像边界颜色值已经处理过使用的接缝处边界是两边patch颜色均值，顶点是相关所有patch颜色均值
 * @param[in] mask 标记用来融合的纹理条包括内外边界border和内部inter。见课件图示
 * @param[in] bias 权重用来计算像素的Laplace时，控制src和dst占比，ColorLaplacian(src,i)*bias + ColorLaplacian(dst,i)*(1.f-bias))
 */
void MeshTexture::PoissonBlending(const Image32F3& src, Image32F3& dst, const Image8U& mask, float bias)
{
	ASSERT(src.width() == mask.width() && src.width() == dst.width());
	ASSERT(src.height() == mask.height() && src.height() == dst.height());
	ASSERT(src.channels() == 3 && dst.channels() == 3 && mask.channels() == 1);
	ASSERT(src.type() == CV_32FC3 && dst.type() == CV_32FC3 && mask.type() == CV_8U);

	#ifndef _RELEASE
	// check the mask border has no pixels marked as interior
	// 确认mask边界是否有像素被标记为内部
	for (int x=0; x<mask.cols; ++x)
		ASSERT(mask(0,x) != interior && mask(mask.rows-1,x) != interior);
	for (int y=0; y<mask.rows; ++y)
		ASSERT(mask(y,0) != interior && mask(y,mask.cols-1) != interior);
	#endif

	const int n(dst.area());
	const int width(dst.width());

	TImage<MatIdx> indices(dst.size());
	indices.memset(0xff);
    // 泊松方程 Ax=b 
	MatIdx nnz(0);
	for (int i = 0; i < n; ++i)
		if (mask(i) != empty)
			indices(i) = nnz++;
   
	Colors coeffB(nnz); //b
	CLISTDEF0(MatEntry) coeffA(0, nnz); //A
	for (int i = 0; i < n; ++i) {
		switch (mask(i)) {
		case border: {
			const MatIdx idx(indices(i));
			ASSERT(idx != -1);
			coeffA.AddConstruct(idx, idx, 1.f);
			// 边界处颜色不变直接使用之前计算的均值即dst对应的颜色
			coeffB[idx] = (const Color&)dst(i);
		} break;
		case interior: {
			const MatIdx idxUp(indices(i - width));
			const MatIdx idxLeft(indices(i - 1));
			const MatIdx idxCenter(indices(i));
			const MatIdx idxRight(indices(i + 1));
			const MatIdx idxDown(indices(i + width));
			// all indices should be either border conditions or part of the optimization
			// 所有索引应该是在边界或者内部待优化
			ASSERT(idxUp != -1 && idxLeft != -1 && idxCenter != -1 && idxRight != -1 && idxDown != -1);
			// [1,1,-4,1,1]Laplace算子
			coeffA.AddConstruct(idxCenter, idxUp, 1.f);
			coeffA.AddConstruct(idxCenter, idxLeft, 1.f);
			coeffA.AddConstruct(idxCenter, idxCenter,-4.f);
			coeffA.AddConstruct(idxCenter, idxRight, 1.f);
			coeffA.AddConstruct(idxCenter, idxDown, 1.f);
			// set target coefficient 
			// 内部点inter的散度计算，即Laplace算子使用,如果bias*原图+(1-bias)*边界处理过的图dst
			// div(5)=[V(2)+V(4)+V(6)+V(8)]-4*V(5)
			coeffB[idxCenter] = (bias == 1.f ?
								 ColorLaplacian(src,i) :
								 ColorLaplacian(src,i)*bias + ColorLaplacian(dst,i)*(1.f-bias));
		} break;
		}
	}
    // 构建稀疏矩阵A
	SparseMat A(nnz, nnz);
	A.setFromTriplets(coeffA.Begin(), coeffA.End());
	coeffA.Release();
	// eigen稀疏矩阵求解
	#ifdef TEXOPT_SOLVER_SPARSELU
	// use SparseLU factorization
	// (faster, but not working if EIGEN_DEFAULT_TO_ROW_MAJOR is defined, bug inside Eigen)
	const Eigen::SparseLU< SparseMat, Eigen::COLAMDOrdering<MatIdx> > solver(A);
	#else
	// use BiCGSTAB solver
	const Eigen::BiCGSTAB< SparseMat, Eigen::IncompleteLUT<float> > solver(A);
	#endif
	ASSERT(solver.info() == Eigen::Success);
	for (int channel=0; channel<3; ++channel) {
		const Eigen::Map< Eigen::VectorXf, Eigen::Unaligned, Eigen::Stride<0,3> > b(coeffB.Begin()->ptr()+channel, nnz);
		const Eigen::VectorXf x(solver.solve(b));
		ASSERT(solver.info() == Eigen::Success);
		for (int i = 0; i < n; ++i) {
			const MatIdx index(indices(i));
			if (index != -1)
				dst(i)[channel] = x[index];
		}
	}
}
/**
 * @brief 局部颜色校正：泊松融合  在纹理块边界处进行局部颜色校正，使得纹理块间过渡比较平滑。泊松纹理融合具体见课件
 * 
 */
void MeshTexture::LocalSeamLeveling()
{
	ASSERT(!seamVertices.IsEmpty());
	const unsigned numPatches(texturePatches.GetSize()-1);
	// adjust texture patches locally, so that the border continues smoothly inside the patch
	// 局部调整纹理patch，使得边界平滑过渡到patch内部
	#ifdef TEXOPT_USE_OPENMP
	#pragma omp parallel for schedule(dynamic)
	for (int i=0; i<(int)numPatches; ++i) {
	#else
	for (unsigned i=0; i<numPatches; ++i) {
	#endif
		const uint32_t idxPatch((uint32_t)i);
		TexturePatch& texturePatch = texturePatches[idxPatch];
		// extract image
		// 取patch对应的image
		const Image8U3& image0(images[texturePatch.label].image);
		Image32F3 image, imageOrg;
		image0(texturePatch.rect).convertTo(image, CV_32FC3, 1.0/255.0);
		image.copyTo(imageOrg);
		// render patch coverage
		// 渲染patch对应的mask
		Image8U mask(texturePatch.rect.size());
		{
			mask.memset(0);
			RasterPatchCoverageData data(mask);
			FOREACHPTR(pIdxFace, texturePatch.faces) {
				const FIndex idxFace(*pIdxFace);
				data.tri = faceTexcoords.Begin()+idxFace*3;
				ColorMap::RasterizeTriangle(data.tri[0], data.tri[1], data.tri[2], data);
			}
		}
		// render the patch border meeting neighbor patches
		// 渲染与邻域patch的边界
		const TexCoord offset(texturePatch.rect.tl()); // patch相对在整个纹理图中的偏移量
		FOREACHPTR(pSeamVertex, seamVertices) {
			const SeamVertex& seamVertex0 = *pSeamVertex;
			if (seamVertex0.patches.GetSize() < 2)
				continue;
			const uint32_t idxVertPatch0(seamVertex0.patches.Find(idxPatch));
			if (idxVertPatch0 == SeamVertex::Patches::NO_INDEX)
				continue;
			const SeamVertex::Patch& patch0 = seamVertex0.patches[idxVertPatch0];
			const TexCoord p0(patch0.proj-offset);
			// for each edge of this vertex belonging to this patch...
			// 处理属于patch0的所有包含该顶点的edge
			FOREACHPTR(pEdge0, patch0.edges) {
				// select the same edge leaving from the adjacent vertex
				// 选择远离相邻顶点的同一条边
				const SeamVertex& seamVertex1 = seamVertices[pEdge0->idxSeamVertex];
				const uint32_t idxVertPatch0Adj(seamVertex1.patches.Find(idxPatch));
				ASSERT(idxVertPatch0Adj != SeamVertex::Patches::NO_INDEX);
				const SeamVertex::Patch& patch0Adj = seamVertex1.patches[idxVertPatch0Adj];
				const TexCoord p0Adj(patch0Adj.proj-offset);
				// find the other patch sharing the same edge (edge with same adjacent vertex)
				// 找到共享同一个edge的其它patch
				FOREACH(idxVertPatch1, seamVertex0.patches) {
					if (idxVertPatch1 == idxVertPatch0)
						continue;
					const SeamVertex::Patch& patch1 = seamVertex0.patches[idxVertPatch1];
					const uint32_t idxEdge1(patch1.edges.Find(pEdge0->idxSeamVertex));
					if (idxEdge1 == SeamVertex::Patch::Edges::NO_INDEX)
						continue;
					//pi不用减去对应patch的偏移量原因是p1对应的就是view原图image1  而p0的imageOrg取得原图中的对应纹理块，所以坐标要减去纹理块起始坐标
					const TexCoord& p1(patch1.proj);
					// select the same edge belonging to the second patch leaving from the adjacent vertex
					const uint32_t idxVertPatch1Adj(seamVertex1.patches.Find(patch1.idxPatch));
					ASSERT(idxVertPatch1Adj != SeamVertex::Patches::NO_INDEX);
					const SeamVertex::Patch& patch1Adj = seamVertex1.patches[idxVertPatch1Adj];
					const TexCoord& p1Adj(patch1Adj.proj);
					// this is an edge separating two (valid) patches;
					// draw it on this patch as the mean color of the two patches
					// edge 分开了两个patch，计算edge在两个patch的平均值（边界条件）
					const Image8U3& image1(images[texturePatches[patch1.idxPatch].label].image);
					RasterPatchMeanEdgeData data(image, mask, imageOrg, image1, p0, p0Adj, p1, p1Adj);
					Image32F3::DrawLine(p0, p0Adj, data);
					// skip remaining patches,
					// as a manifold edge is shared by maximum two face (one in each patch), which we found already
					break;
				}
			}
		}
		// render the vertices at the patch border meeting neighbor patches
		// 渲染patch边界的顶点，计算所有包含该顶点patch，计算颜色均值
		const Sampler sampler;
		FOREACHPTR(pSeamVertex, seamVertices) {
			const SeamVertex& seamVertex = *pSeamVertex;
			if (seamVertex.patches.GetSize() < 2)
				continue;
			const uint32_t idxVertPatch(seamVertex.patches.Find(idxPatch));
			if (idxVertPatch == SeamVertex::Patches::NO_INDEX)
				continue;
			AccumColor accumColor;
			// for each patch...
			FOREACHPTR(pPatch, seamVertex.patches) {
				const SeamVertex::Patch& patch = *pPatch;
				// add its view to the vertex mean color
				// 将邻接patch view颜色值加入均值计算中
				const Image8U3& img(images[texturePatches[patch.idxPatch].label].image);
				accumColor.Add(img.sample<Sampler,Color>(sampler, patch.proj)/255.f, 1.f);
			}
			const SeamVertex::Patch& thisPatch = seamVertex.patches[idxVertPatch];
			const ImageRef pt(ROUND2INT(thisPatch.proj-offset));
			image(pt) = accumColor.Normalized();
			mask(pt) = border;
		}
		// make sure the border is continuous and
		// keep only the exterior tripe of the given size
		// 确保边界是连续的，因为patch全局已经调整过了，所以局部只调整patch边界向里20个像素构成的边界带见论文Let There Be Color!中fig.5
		ProcessMask(mask, 20);
		// compute texture patch blending
		// 泊松融合
		PoissonBlending(imageOrg, image, mask);
		// apply color correction to the patch image
		// 应用校正的颜色到patch中
		cv::Mat imagePatch(images[texturePatch.label].image(texturePatch.rect));
		for (int r=0; r<image.rows; ++r) {
			for (int c=0; c<image.cols; ++c) {
				if (mask(r,c) == empty)
					continue;
				const Color& a = image(r,c);
				Pixel8U& v = imagePatch.at<Pixel8U>(r,c);
				for (int p=0; p<3; ++p)
					v[p] = (uint8_t)CLAMP(ROUND2INT(a[p]*255.f), 0, 255);
			}
		}
	}
}
/**
 * @brief 生成纹理图
 * 
 * @param[in] bGlobalSeamLeveling  控制全局颜色校正的开关
 * @param[in] bLocalSeamLeveling   控制局部颜色校正的开关
 * @param[in] nTextureSizeMultiple 
 * @param[in] nRectPackingHeuristic 
 * @param[in] colEmpty             rgb颜色值用来填充纹理图上空缺部分的颜色
 */
void MeshTexture::GenerateTexture(bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty)
{
	// project patches in the corresponding view and compute texture-coordinates and bounding-box
	// Step 1 投影patch到对应view上计算纹理坐标和包围盒
	const int border(2); // 纹理图的预留边界
	faceTexcoords.Resize(faces.GetSize()*3); // 纹理坐标size是faces的3倍，因为记录的是face的三个顶点的纹理坐标
	#ifdef TEXOPT_USE_OPENMP
	const unsigned numPatches(texturePatches.GetSize()-1); // 减一是去掉最后一个没有label的patch
	#pragma omp parallel for schedule(dynamic)
	for (int_t idx=0; idx<(int_t)numPatches; ++idx) {
		TexturePatch& texturePatch = texturePatches[(uint32_t)idx];
	#else
	for (TexturePatch *pTexturePatch=texturePatches.Begin(), *pTexturePatchEnd=texturePatches.End()-1; pTexturePatch<pTexturePatchEnd; ++pTexturePatch) {
		TexturePatch& texturePatch = *pTexturePatch;
	#endif
		const Image& imageData = images[texturePatch.label];
		// project vertices and compute bounding-box
		// 投影顶点，计算在图像上的所有投影点的包围盒
		AABB2f aabb(true); // 用来计算纹理坐标包围盒
		FOREACHPTR(pIdxFace, texturePatch.faces) {
			const FIndex idxFace(*pIdxFace);
			const Face& face = faces[idxFace];
			// 指针操作访问每个face的纹理坐标地址
			TexCoord* texcoords = faceTexcoords.Begin()+idxFace*3; 
			for (int i=0; i<3; ++i) {
				texcoords[i] = imageData.camera.ProjectPointP(vertices[face[i]]);
				ASSERT(imageData.image.isInsideWithBorder(texcoords[i], border));
				aabb.InsertFull(texcoords[i]);
			}
		}
		// compute relative texture coordinates
		// 计算相对纹理坐标：上面patch投影得到一个纹理块aabb取其坐标最大最小，将其放在最终的纹理图上，会相对在原图位置有一个偏移量offset
		ASSERT(imageData.image.isInside(Point2f(aabb.ptMin)));
		ASSERT(imageData.image.isInside(Point2f(aabb.ptMax)));
		// 计算纹理块最大包围矩形 rect.xy是纹理块在纹理图上的起始坐标
		texturePatch.rect.x = FLOOR2INT(aabb.ptMin[0])-border;
		texturePatch.rect.y = FLOOR2INT(aabb.ptMin[1])-border;
		// 计算长和宽，加了两倍的border宽度，即取纹理块时多取一2个像素的宽度。原因是如果不在原图上多取一点，三维模型显示时纹理会有
		// 黑色接缝（实际是没有的只是渲染的时候如果刚好在边界会出现这种现象，所以一般会多取一点），大家可以自己设置为0看下实验效果
		texturePatch.rect.width = CEIL2INT(aabb.ptMax[0]-aabb.ptMin[0])+border*2;
		texturePatch.rect.height = CEIL2INT(aabb.ptMax[1]-aabb.ptMin[1])+border*2;
		ASSERT(imageData.image.isInside(texturePatch.rect.tl()));
		ASSERT(imageData.image.isInside(texturePatch.rect.br()));
		// rect.tl(top/left即左上角)指的是纹理块在原始图像中的起始坐标也就是相对原点的偏移量
		const TexCoord offset(texturePatch.rect.tl());
		FOREACHPTR(pIdxFace, texturePatch.faces) {
			const FIndex idxFace(*pIdxFace);
			TexCoord* texcoords = faceTexcoords.Begin()+idxFace*3;
			// 因为刚计算的纹理坐标还是相对于原图的，所以需要减去起始点，让纹理块坐标从（0，0）点开始
			for (int v=0; v<3; ++v)
				texcoords[v] -= offset;
		}
	}
	{
		// init last patch to point to a small uniform color patch
		// 初始化最后一个无label的patch将其指向一个相同颜色
		TexturePatch& texturePatch = texturePatches.Last();
		const int sizePatch(border*2+1);
		texturePatch.rect = cv::Rect(0,0, sizePatch,sizePatch);
		FOREACHPTR(pIdxFace, texturePatch.faces) {
			const FIndex idxFace(*pIdxFace);
			TexCoord* texcoords = faceTexcoords.Begin()+idxFace*3;
			for (int i=0; i<3; ++i)
				texcoords[i] = TexCoord(0.5f, 0.5f);
		}
	}

	// perform seam leveling
	// Step 2 处理纹理接缝，主要是首先进行全局颜色校正，然后在纹理块交界处进行泊松融合消除纹理块交界处的颜色差异
	if (texturePatches.GetSize() > 2 && (bGlobalSeamLeveling || bLocalSeamLeveling)) {
		// create seam vertices and edges
		// 创建不同纹理块间连接处的顶点和边
		CreateSeamVertices();

		// perform global seam leveling
		// 全局颜色校正
		if (bGlobalSeamLeveling) {
			TD_TIMER_STARTD();
			GlobalSeamLeveling();
			DEBUG_ULTIMATE("\tglobal seam leveling completed (%s)", TD_TIMER_GET_FMT().c_str());
		}

		// perform local seam leveling
		// 局部颜色校正
		if (bLocalSeamLeveling) {
			TD_TIMER_STARTD();
			LocalSeamLeveling();
			DEBUG_ULTIMATE("\tlocal seam leveling completed (%s)", TD_TIMER_GET_FMT().c_str());
		}
	}

	// merge texture patches with overlapping rectangles 
	// Step 3 合并纹理块：如果两个纹理块label相同，且小的包含在大的里面则合并。
	for (unsigned i=0; i<texturePatches.GetSize()-2; ++i) {
		TexturePatch& texturePatchBig = texturePatches[i];
		for (unsigned j=1; j<texturePatches.GetSize()-1; ++j) {
			if (i == j)
				continue;
			TexturePatch& texturePatchSmall = texturePatches[j];
			// 如果label不同，不能合并
			if (texturePatchBig.label != texturePatchSmall.label)
				continue;
			// 如果小patch不被包含在大patch里也不能合并
			if (!RectsBinPack::IsContainedIn(texturePatchSmall.rect, texturePatchBig.rect))
				continue;
			// translate texture coordinates
			// 变换合并后小patch的纹理坐标。
			const TexCoord offset(texturePatchSmall.rect.tl()-texturePatchBig.rect.tl());// 计算两个patch在原始图像中起始点相对偏移量
			FOREACHPTR(pIdxFace, texturePatchSmall.faces) {
				const FIndex idxFace(*pIdxFace);
				TexCoord* texcoords = faceTexcoords.Begin()+idxFace*3;
				// 将小patch的纹理坐标变换到大path的纹理坐标系中
				for (int v=0; v<3; ++v)
					texcoords[v] += offset;
			}
			// join faces lists
			// 将小patch的faces合并到大的里面
			texturePatchBig.faces.JoinRemove(texturePatchSmall.faces);
			// remove the small patch
			// 在texturePatches移除小path
			texturePatches.RemoveAtMove(j--);
		}
	}

	// Step 4 create texture 创建纹理图,将每个纹理块集中到同一张纹理图中，更新新的纹理坐标。纹理坐标和顶点面信息会保存在obj中
	// 纹理图会保存在jpg/png等。mtl存储了一些贴图的材质信息具体见课件介绍 obj,mtl,jpg
	// 值得注意的是：这里我们会发现每个纹理块大小是不一样的所以最终看到的纹理图会有很多大小不同的纹理块
	{
		// arrange texture patches to fit the smallest possible texture image
		// 排列纹理块以组合最小尺寸纹理图像
		RectsBinPack::RectArr rects(texturePatches.GetSize());
		FOREACH(i, texturePatches)
			rects[i] = texturePatches[i].rect;
		int textureSize(RectsBinPack::ComputeTextureSize(rects, nTextureSizeMultiple));
		// increase texture size till all patches fit
		// 增加纹理图size,直到所有纹理块都包含进去
		while (true) {
			TD_TIMER_STARTD();
			bool bPacked(false);
			const unsigned typeRectsBinPack(nRectPackingHeuristic/100);
			const unsigned typeSplit((nRectPackingHeuristic-typeRectsBinPack*100)/10);
			const unsigned typeHeuristic(nRectPackingHeuristic%10);
			switch (typeRectsBinPack) {
			case 0: {
				MaxRectsBinPack pack(textureSize, textureSize);
				bPacked = pack.Insert(rects, (MaxRectsBinPack::FreeRectChoiceHeuristic)typeHeuristic);
				break; }
			case 1: {
				SkylineBinPack pack(textureSize, textureSize, typeSplit!=0);
				bPacked = pack.Insert(rects, (SkylineBinPack::LevelChoiceHeuristic)typeHeuristic);
				break; }
			case 2: {
				GuillotineBinPack pack(textureSize, textureSize);
				bPacked = pack.Insert(rects, false, (GuillotineBinPack::FreeRectChoiceHeuristic)typeHeuristic, (GuillotineBinPack::GuillotineSplitHeuristic)typeSplit);
				break; }
			default:
				ABORT("error: unknown RectsBinPack type");
			}
			DEBUG_ULTIMATE("\tpacking texture completed: %u patches, %u texture-size (%s)", rects.GetSize(), textureSize, TD_TIMER_GET_FMT().c_str());
			if (bPacked)
				break;
			textureSize *= 2;
		}

		// create texture image创建纹理图
		const float invNorm(1.f/(float)(textureSize-1));
		textureDiffuse.create(textureSize, textureSize);
		textureDiffuse.setTo(cv::Scalar(colEmpty.b, colEmpty.g, colEmpty.r));
		#ifdef TEXOPT_USE_OPENMP
		#pragma omp parallel for schedule(dynamic)
		for (int_t i=0; i<(int_t)texturePatches.GetSize(); ++i) {
		#else
		FOREACH(i, texturePatches) {
		#endif
			const uint32_t idxPatch((uint32_t)i);
			const TexturePatch& texturePatch = texturePatches[idxPatch];
			const RectsBinPack::Rect& rect = rects[idxPatch];
			// copy patch image赋值patch图像
			ASSERT((rect.width == texturePatch.rect.width && rect.height == texturePatch.rect.height) ||
				   (rect.height == texturePatch.rect.width && rect.width == texturePatch.rect.height));
			int x(0), y(1);
			if (texturePatch.label != NO_ID) {
				const Image& imageData = images[texturePatch.label];
				cv::Mat patch(imageData.image(texturePatch.rect));
				if (rect.width != texturePatch.rect.width) {
					// flip patch and texture-coordinates
					// 翻转patch和对应纹理坐标
					patch = patch.t();
					x = 1; y = 0;
				}
				patch.copyTo(textureDiffuse(rect));
			}
			// compute final texture coordinates
			// 计算最终的纹理坐标
			const TexCoord offset(rect.tl());
			FOREACHPTR(pIdxFace, texturePatch.faces) {
				const FIndex idxFace(*pIdxFace);
				TexCoord* texcoords = faceTexcoords.Begin()+idxFace*3;
				for (int v=0; v<3; ++v) {
					TexCoord& texcoord = texcoords[v];
					// translate, normalize and flip Y axis
					// 纹理坐标变换，归一化，翻转Y轴
					texcoord = TexCoord(
						(texcoord[x]+offset.x)*invNorm,
						1.f-(texcoord[y]+offset.y)*invNorm
					);
				}
			}
		}
	}
}

// texture mesh 
/**
 * @brief 纹理贴图，首先给每个face（三角网格）选择一个视图（图像），然后生成纹理块（texture patch）。由于不同纹理块来自不同的
 *        图像故光照角度不同，所以不同patch间会有颜色差异，因此需要进行颜色校正：先全局校正整体颜色差异（globel）再局部调整接缝处的颜色差
 *        异(local seam leveling)
 * @param[in] nResolutionLevel     scale用于计算纹理贴图的图像分辨率 =image_size/2^nResolutionLevel
 * @param[in] nMinResolution       贴图最小分辨率阈值，与上述分辨率相比取最大值
 * @param[in] fOutlierThreshold    颜色差异阈值，用于face选择投影的view时，剔除与大多view不同的外点view
 * @param[in] fRatioDataSmoothness 平滑系数
 * @param[in] bGlobalSeamLeveling  控制是否全局纹理融合，bool型
 * @param[in] bLocalSeamLeveling   控制是否局部纹理融合，bool型
 * @param[in] nTextureSizeMultiple 
 * @param[in] nRectPackingHeuristic 
 * @param[in] colEmpty             rgb颜色值用来填充纹理图上空缺部分的颜色
 * @return true 
 * @return false 
 * 参考论文：Let There Be Color! Large-Scale Texturing of 3D Reconstructions
 */
bool Scene::TextureMesh(unsigned nResolutionLevel, unsigned nMinResolution, float fOutlierThreshold, float fRatioDataSmoothness, bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty)
{
	MeshTexture texture(*this, nResolutionLevel, nMinResolution);

	// assign the best view to each face 
	// Step 1 给每个face（三角网格）分配最佳视图view 
	{
		TD_TIMER_STARTD();
		if (!texture.FaceViewSelection(fOutlierThreshold, fRatioDataSmoothness))
			return false;
		DEBUG_EXTRA("Assigning the best view to each face completed: %u faces (%s)", mesh.faces.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

	// generate the texture image and atlas 
	// Step 2 生成纹理图像，并进行纹理颜色校正与融合
	{
		TD_TIMER_STARTD();
		texture.GenerateTexture(bGlobalSeamLeveling, bLocalSeamLeveling, nTextureSizeMultiple, nRectPackingHeuristic, colEmpty);
		DEBUG_EXTRA("Generating texture atlas and image completed: %u patches, %u image size (%s)", texture.texturePatches.GetSize(), mesh.textureDiffuse.width(), TD_TIMER_GET_FMT().c_str());
	}

	return true;
} // TextureMesh
/*----------------------------------------------------------------*/
