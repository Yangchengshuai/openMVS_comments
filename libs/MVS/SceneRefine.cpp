/*
* SceneRefine.cpp
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

using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////

// uncomment to ensure edge size and improve vertex valence
// (should enable more stable flow)
#define MESHOPT_ENSUREEDGESIZE 1 // 0 - at all resolution

// uncomment to use constant z-buffer bias
// (should be enough, as the numerical error does not depend on the depth)
#define MESHOPT_DEPTHCONSTBIAS 0.05f

// uncomment to enable memory pool
// (should reduce the allocation times for frequent used images)
#define MESHOPT_TYPEPOOL

// uncomment to enable CERES optimization module
// (similar performance with the custom minimizer)
#ifdef _USE_CERES
#define MESHOPT_CERES
#endif

#ifdef MESHOPT_TYPEPOOL
#define DEC_BitMatrix(var)		BitMatrix& var = *BitMatrixPool()
#define DEC_Image(type, var)	TImage<type>& var = *ImagePool<type>()
#define DST_BitMatrix(var)		BitMatrixPool(&(var))
#define DST_Image(var)			ImagePool(&(var))
#else
#define DEC_BitMatrix(var)		BitMatrix var;
#define DEC_Image(type, var)	TImage<type> var;
#define DST_BitMatrix(var)
#define DST_Image(var)
#endif


// S T R U C T S ///////////////////////////////////////////////////

typedef float Real;
typedef Mesh::Vertex Vertex;
typedef Mesh::VIndex VIndex;
typedef Mesh::Face Face;
typedef Mesh::FIndex FIndex;

class MeshRefine {
public:
	typedef TPoint3<Real> Grad; // 梯度
	typedef CLISTDEF0IDX(Grad,VIndex) GradArr; // 存放每个点梯度的list

	typedef std::unordered_set<FIndex> CameraFaces; // 无序集合：存放相机能看到的所有face的id
	typedef SEACAVE::cList<CameraFaces,const CameraFaces&,2> CameraFacesArr; //上述的list

	typedef TImage<cuint32_t> FaceMap; // 存放投影到图像上的face的id（每个像素值是face id）
	typedef TImage<Point3f> BaryMap;  // 图像上每个像素对应的空间位置在face中的重心坐标

	// store necessary data about a view
	// view 存储每帧图像相关信息
	struct View {
		typedef TPoint2<float> Grad;
		typedef TImage<Grad> ImageGrad;
		Image32F image; // image pixels 灰度图
		ImageGrad imageGrad; // image pixel gradients 梯度图
		TImage<Real> imageMean; // image pixels mean 平均值
		TImage<Real> imageVar; // image pixels variance 方差
		FaceMap faceMap; // remember for each pixel what face projects there 记录每个像素上是哪个face投影的
		DepthMap depthMap; // depth-map 深度图
		BaryMap baryMap; // barycentric coordinates 重心坐标
	};
	typedef SEACAVE::cList<View,const View&,2> ViewsArr;

	// used to render a mesh for optimization
	// 用于渲染mesh,计算facemap barymap
	struct RasterMesh : TRasterMesh<RasterMesh> {
		typedef TRasterMesh<RasterMesh> Base;
		FaceMap& faceMap;
		BaryMap& baryMap;
		FIndex idxFace;
		RasterMesh(const Mesh::VertexArr& _vertices, const Camera& _camera, DepthMap& _depthMap, FaceMap& _faceMap, BaryMap& _baryMap)
			: Base(_vertices, _camera, _depthMap), faceMap(_faceMap), baryMap(_baryMap) {}
		void Clear() {
			Base::Clear();
			faceMap.memset((uint8_t)NO_ID);
			baryMap.memset(0);
		}
		// 计算像素点对应depth，faceID，重心
		void Raster(const ImageRef& pt) {
			if (!depthMap.isInsideWithBorder<int,4>(pt))
				return;
			const float z((float)INVERT(normalPlane.dot(camera.TransformPointI2C(Point2(pt)))));
			ASSERT(z > 0);
			Depth& depth = depthMap(pt);
			if (depth == 0 || depth > z) {
				depth = z;
				faceMap(pt) = idxFace;
				baryMap(pt) = CorrectBarycentricCoordinates(BarycentricCoordinatesUV(pti[0], pti[1], pti[2], Point2f(pt)));
			}
		}
	};


public:
	MeshRefine(Scene& _scene, unsigned _nReduceMemory, unsigned _nAlternatePair=true, Real _weightRegularity=1.5f, Real _ratioRigidityElasticity=0.8f, unsigned _nResolutionLevel=0, unsigned _nMinResolution=640, unsigned nMaxViews=8, unsigned nMaxThreads=1);
	~MeshRefine();

	bool IsValid() const { return !pairs.IsEmpty(); }

	bool InitImages(Real scale, Real sigma=0);

	void ListVertexFacesPre();
	void ListVertexFacesPost();
	void ListCameraFaces();

	void ListFaceAreas(Mesh::AreaArr& maxAreas);
	void SubdivideMesh(uint32_t maxArea, float fDecimate=1.f, unsigned nCloseHoles=15, unsigned nEnsureEdgeSize=1);

	double ScoreMesh(double* gradients);

	// given a vertex position and a projection camera, compute the projected position and its derivative
	// 投影顶点到图像上
	template <typename TP, typename TX, typename T, typename TJ>
	static T ProjectVertex(const TP* P, const TX* X, T* x, TJ* jacobian=NULL);

	static bool IsDepthSimilar(const DepthMap& depthMap, const Point2f& pt, Depth z);
	static void ProjectMesh(
		const Mesh::VertexArr& vertices, const Mesh::FaceArr& faces, const CameraFaces& cameraFaces,
		const Camera& camera, const Image8U::Size& size,
		DepthMap& depthMap, FaceMap& faceMap, BaryMap& baryMap);
	static void ImageMeshWarp(
		const DepthMap& depthMapA, const Camera& cameraA,
		const DepthMap& depthMapB, const Camera& cameraB,
		const Image32F& imageB, Image32F& imageA, BitMatrix& mask);
	static void ComputeLocalVariance(
		const Image32F& image, const BitMatrix& mask,
		TImage<Real>& imageMean, TImage<Real>& imageVar);
	static float ComputeLocalZNCC(
		const Image32F& imageA, const TImage<Real>& imageMeanA, const TImage<Real>& imageVarA,
		const Image32F& imageB, const TImage<Real>& imageMeanB, const TImage<Real>& imageVarB,
		const BitMatrix& mask, TImage<Real>& imageZNCC, TImage<Real>& imageDZNCC);
	static void ComputePhotometricGradient(
		const Mesh::FaceArr& faces, const Mesh::NormalArr& normals,
		const DepthMap& depthMapA, const FaceMap& faceMapA, const BaryMap& baryMapA, const Camera& cameraA,
		const Camera& cameraB, const View& viewB,
		const TImage<Real>& imageDZNCC, const BitMatrix& mask, GradArr& photoGrad, UnsignedArr& photoGradNorm, Real RegularizationScale);
	static float ComputeSmoothnessGradient1(
		const Mesh::VertexArr& vertices, const Mesh::VertexVerticesArr& vertexVertices, const BoolArr& vertexBoundary,
		GradArr& smoothGrad1, VIndex idxStart, VIndex idxEnd);
	static void ComputeSmoothnessGradient2(
		const GradArr& smoothGrad1, const Mesh::VertexVerticesArr& vertexVertices, const BoolArr& vertexBoundary,
		GradArr& smoothGrad2, VIndex idxStart, VIndex idxEnd);
	template<typename TYPE>
	static TYPE* TypePool(TYPE* = NULL);
	template<typename TYPE>
	static inline TImage<TYPE>* ImagePool(TImage<TYPE>* pImage = NULL) { return TypePool< TImage<TYPE> >(pImage); }
	static inline BitMatrix* BitMatrixPool(BitMatrix* pMask = NULL) { return TypePool<BitMatrix>(pMask); }

	static void* ThreadWorkerTmp(void*);
	void ThreadWorker();
	void WaitThreadWorkers(size_t nJobs);
	void ThSelectNeighbors(uint32_t idxImage, std::unordered_set<uint64_t>& mapPairs, unsigned nMaxViews);
	void ThInitImage(uint32_t idxImage, Real scale, Real sigma);
	void ThProjectMesh(uint32_t idxImage, const CameraFaces& cameraFaces);
	void ThProcessPair(uint32_t idxImageA, uint32_t idxImageB);
	void ThSmoothVertices1(VIndex idxStart, VIndex idxEnd);
	void ThSmoothVertices2(VIndex idxStart, VIndex idxEnd);

public:
	const Real weightRegularity; // 光度一致性和正则项平衡系数a scalar regularity weight to balance between photo-consistency and regularization terms
	Real ratioRigidityElasticity; // 正则项系数a scalar ratio used to compute the regularity gradient as a combination of rigidity and elasticity
	const unsigned nResolutionLevel; // 控制用来优化mesh的图像分辨率how many times to scale down the images before mesh optimization
	const unsigned nMinResolution; // 图像最小分辨率how many times to scale down the images before mesh optimization
	const unsigned nReduceMemory; // 是否减少内存占用recompute image mean and variance in order to reduce memory requirements
	unsigned nAlternatePair; // 控制如何使用图像对里面图像的作为参考图像using an image pair alternatively as reference image (0 - both, 1 - alternate, 2 - only left, 3 - only right)
	unsigned iteration; // 记录第几次迭代 current refinement iteration

	Scene& scene; // 存放mesh的顶点和facethe mesh vertices and faces

	// gradient related
	// 梯度相关的变量
	float scorePhoto;
	float scoreSmooth;
	GradArr photoGrad;
	FloatArr photoGradNorm;
	FloatArr vertexDepth;
	GradArr smoothGrad1;
	GradArr smoothGrad2;

	// valid after ListCameraFaces()
	Mesh::NormalArr& faceNormals; //每个face的法线 normals corresponding to each face

	// valid the entire time, but changes
	// 整个期间都有效
	Mesh::VertexArr& vertices; // 顶点
	Mesh::FaceArr& faces; // face
	Mesh::VertexVerticesArr& vertexVertices; // 每个顶点相邻的顶点（one-ring 顶点） for each vertex, the list of adjacent vertices
	Mesh::VertexFacesArr& vertexFaces; // 每个顶点包含的facesfor each vertex, the list of faces containing it
	BoolArr& vertexBoundary; // 存储每个顶点是否在边界for each vertex, stores if it is at the boundary or not

	// constant the entire time
	// 整个优化期间固定不变的量
	ImageArr& images; // 用来优化的图像
	ViewsArr views; // 图像信息views' data
	PairIdxArr pairs; // 用来优化mesh的图像对image pairs used to refine the mesh

	// multi-threading
	static SEACAVE::EventQueue events; // internal events queue (processed by the working threads)
	static SEACAVE::cList<SEACAVE::Thread> threads; // worker threads
	static CriticalSection cs; // mutex
	static Semaphore sem; // signal job end

	enum { HalfSize = 3 }; // half window size used to compute ZNCC
};

// call with empty parameter to get an unused image;
// call with an image pointer retrieved earlier to signal that is not needed anymore
template<typename TYPE>
TYPE* MeshRefine::TypePool(TYPE* pObj)
{
	typedef CAutoPtr<TYPE> TypePtr;
	static CriticalSection cs;
	static cList<TypePtr,TYPE*> objects;
	static cList<TYPE*,TYPE*,0> unused;
	Lock l(cs);
	if (pObj == NULL) {
		if (unused.IsEmpty())
			return objects.AddConstruct(new TYPE);
		return unused.RemoveTail();
	} else {
		ASSERT(objects.Find(pObj) != NO_IDX);
		ASSERT(unused.Find(pObj) == NO_IDX);
		unused.Insert(pObj);
		return NULL;
	}
}


enum EVENT_TYPE {
	EVT_JOB = 0,
	EVT_CLOSE,
};

class EVTClose : public Event
{
public:
	EVTClose() : Event(EVT_CLOSE) {}
};
class EVTSelectNeighbors : public Event
{
public:
	uint32_t idxImage;
	std::unordered_set<uint64_t>& mapPairs;
	unsigned nMaxViews;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThSelectNeighbors(idxImage, mapPairs, nMaxViews);
		return true;
	}
	EVTSelectNeighbors(uint32_t _idxImage, std::unordered_set<uint64_t>& _mapPairs, unsigned _nMaxViews) : Event(EVT_JOB), idxImage(_idxImage), mapPairs(_mapPairs), nMaxViews(_nMaxViews) {}
};
class EVTInitImage : public Event
{
public:
	uint32_t idxImage;
	Real scale, sigma;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThInitImage(idxImage, scale, sigma);
		return true;
	}
	EVTInitImage(uint32_t _idxImage, Real _scale, Real _sigma) : Event(EVT_JOB), idxImage(_idxImage), scale(_scale), sigma(_sigma) {}
};
class EVTProjectMesh : public Event
{
public:
	uint32_t idxImage;
	const MeshRefine::CameraFaces& cameraFaces;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThProjectMesh(idxImage, cameraFaces);
		return true;
	}
	EVTProjectMesh(uint32_t _idxImage, const MeshRefine::CameraFaces& _cameraFaces) : Event(EVT_JOB), idxImage(_idxImage), cameraFaces(_cameraFaces) {}
};
class EVTProcessPair : public Event
{
public:
	uint32_t idxImageA, idxImageB;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThProcessPair(idxImageA, idxImageB);
		return true;
	}
	EVTProcessPair(uint32_t _idxImageA, uint32_t _idxImageB) : Event(EVT_JOB), idxImageA(_idxImageA), idxImageB(_idxImageB) {}
};
class EVTSmoothVertices1 : public Event
{
public:
	VIndex idxStart, idxEnd;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThSmoothVertices1(idxStart, idxEnd);
		return true;
	}
	EVTSmoothVertices1(VIndex _idxStart, VIndex _idxEnd) : Event(EVT_JOB), idxStart(_idxStart), idxEnd(_idxEnd) {}
};
class EVTSmoothVertices2 : public Event
{
public:
	VIndex idxStart, idxEnd;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThSmoothVertices2(idxStart, idxEnd);
		return true;
	}
	EVTSmoothVertices2(VIndex _idxStart, VIndex _idxEnd) : Event(EVT_JOB), idxStart(_idxStart), idxEnd(_idxEnd) {}
};

SEACAVE::EventQueue MeshRefine::events;
SEACAVE::cList<SEACAVE::Thread> MeshRefine::threads;
CriticalSection MeshRefine::cs;
Semaphore MeshRefine::sem;
// 初始化
MeshRefine::MeshRefine(Scene& _scene, unsigned _nReduceMemory, unsigned _nAlternatePair, Real _weightRegularity, Real _ratioRigidityElasticity, unsigned _nResolutionLevel, unsigned _nMinResolution, unsigned nMaxViews, unsigned nMaxThreads)
	:
	weightRegularity(_weightRegularity),
	ratioRigidityElasticity(_ratioRigidityElasticity),
	nResolutionLevel(_nResolutionLevel),
	nMinResolution(_nMinResolution), 
	nReduceMemory(_nReduceMemory),
	nAlternatePair(_nAlternatePair),
	scene(_scene),
	faceNormals(_scene.mesh.faceNormals),
	vertices(_scene.mesh.vertices),
	faces(_scene.mesh.faces),
	vertexVertices(_scene.mesh.vertexVertices),
	vertexFaces(_scene.mesh.vertexFaces),
	vertexBoundary(_scene.mesh.vertexBoundary),
	images(_scene.images)
{
	// start worker threads
	// 启动线程
	ASSERT(nMaxThreads > 0);
	ASSERT(threads.IsEmpty());
	threads.Resize(nMaxThreads);
	FOREACHPTR(pThread, threads)
		pThread->start(ThreadWorkerTmp, this);
	// keep only best neighbor views for each image
	// 给每个图像只保留一个邻域帧
	std::unordered_set<uint64_t> mapPairs;
	mapPairs.reserve(images.GetSize()*nMaxViews);
	ASSERT(events.IsEmpty());
	FOREACH(idxImage, images)
		events.AddEvent(new EVTSelectNeighbors(idxImage, mapPairs, nMaxViews));
	WaitThreadWorkers(images.GetSize());
	pairs.Reserve(mapPairs.size());
	for (uint64_t pair: mapPairs)
		pairs.AddConstruct(pair);
}
MeshRefine::~MeshRefine()
{
	// wait for the working threads to close
	// 等待线程关闭
	FOREACH(i, threads)
		events.AddEvent(new EVTClose());
	FOREACHPTR(pThread, threads)
		pThread->join();
	scene.mesh.ReleaseExtra();
}

// load and initialize all images at the given scale
// and compute the gradient for each input image
// optional: blur them using the given sigma
// resize图像，计算梯度，滤波（可选）
bool MeshRefine::InitImages(Real scale, Real sigma)
{
	views.Resize(images.GetSize());
	ASSERT(events.IsEmpty());
	FOREACH(idxImage, images)
		events.AddEvent(new EVTInitImage(idxImage, scale, sigma));
	WaitThreadWorkers(images.GetSize());
	iteration = 0;
	return true;
}

// extract array of triangles incident to each vertex
// and check each vertex if it is at the boundary or not
// 提取每个顶点的相关faces即三角face的三个顶点中包含该顶点的所有faces
void MeshRefine::ListVertexFacesPre()
{
	scene.mesh.EmptyExtra();
	scene.mesh.ListIncidenteFaces();
}
void MeshRefine::ListVertexFacesPost()
{
	scene.mesh.ListIncidenteVertices();
	scene.mesh.ListBoundaryVertices();
}

// extract array of faces viewed by each image
// 提取被每个图像看到的所有面
void MeshRefine::ListCameraFaces()
{
	// extract array of faces viewed by each camera
	// 提取被每个相机看到的所有面
	CameraFacesArr arrCameraFaces(images.GetSize());
	{
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
	typedef TOctree<Mesh::VertexArr,float,3> Octree;
	const Octree octree(vertices);
	#if 0 && !defined(_RELEASE)
	Octree::DEBUGINFO_TYPE info;
	octree.GetDebugInfo(&info);
	Octree::LogDebugInfo(info);
	#endif
	FOREACH(ID, images) {
		const Image& imageData = images[ID];
		if (!imageData.IsValid())
			continue;
		typedef TFrustum<float,5> Frustum;
		FacesInserter inserter(vertexFaces, arrCameraFaces[ID]);
		const Frustum frustum(Frustum::MATRIX3x4(((PMatrix::CEMatMap)imageData.camera.P).cast<float>()), (float)imageData.width, (float)imageData.height);
		octree.Traverse(frustum, inserter);
	}
	}

	// project mesh to each camera plane
	// 投影mesh到每个相机平面
	ASSERT(events.IsEmpty());
	FOREACH(idxImage, images)
		events.AddEvent(new EVTProjectMesh(idxImage, arrCameraFaces[idxImage]));
	WaitThreadWorkers(images.GetSize());
}

// compute for each face the projection area as the maximum area in both images of a pair
// 计算face的最大投影面积，调用前要保证ListCameraFaces先调用
// (make sure ListCameraFaces() was called before)
void MeshRefine::ListFaceAreas(Mesh::AreaArr& maxAreas)
{
	ASSERT(maxAreas.IsEmpty());
	// for each image, compute the projection area of visible faces
	// 对每帧图像，计算其能看到的所有faces在该图像上的投影面积
	typedef cList<Mesh::AreaArr> ImageAreaArr;
	ImageAreaArr viewAreas(images.GetSize()); // 存储每帧的能看到的faces的投影面积
	FOREACH(idxImage, images) {
		const Image& imageData = images[idxImage];
		if (!imageData.IsValid())
			continue;
		Mesh::AreaArr& areas = viewAreas[idxImage];
		areas.Resize(faces.GetSize());
		areas.Memset(0);
		const FaceMap& faceMap = views[idxImage].faceMap; // faces投影在image上map，即与图像对应的map的每个像素记录了是哪个face投影的
		// compute area covered by all vertices (incident faces) viewed by this image
		// 计算每个face覆盖的面积
		for (int j=0; j<faceMap.rows; ++j) {
			for (int i=0; i<faceMap.cols; ++i) {
				const FIndex idxFace(faceMap(j,i));
				ASSERT((idxFace == NO_ID && views[idxImage].depthMap(j,i) == 0) || (idxFace != NO_ID && views[idxImage].depthMap(j,i) > 0));
				if (idxFace == NO_ID)
					continue;
				++areas[idxFace];
			}
		}
	}
	// for each pair, mark the faces that have big projection areas in both images
	// 对每个depth计算的图像对，记录face投影面积最大的值
	maxAreas.Resize(faces.GetSize());
	maxAreas.Memset(0);
	FOREACHPTR(pPair, pairs) {
		const Mesh::AreaArr& areasA = viewAreas[pPair->i];
		const Mesh::AreaArr& areasB = viewAreas[pPair->j];
		ASSERT(areasA.GetSize() == areasB.GetSize());
		FOREACH(f, areasA) {
			//? 为啥不是取pair里面的最大值
			const uint16_t minArea(MINF(areasA[f], areasB[f]));
			uint16_t& maxArea = maxAreas[f];
			if (maxArea < minArea)
				maxArea = minArea;
		}
	}
}

// decimate or subdivide mesh such that for each face there is no image pair in which
// its projection area is bigger than the given number of pixels in both images
/**
 * @brief mesh细分或下采样
 * 
 * @param[in] maxArea     每个三角网格最大面积
 * @param[in] fDecimate   下采样率 
 * @param[in] nCloseHoles  补洞最小尺寸
 * @param[in] nEnsureEdgeSize 是否要保证边界尺寸不会太大或太小 默认1
 */
void MeshRefine::SubdivideMesh(uint32_t maxArea, float fDecimate, unsigned nCloseHoles, unsigned nEnsureEdgeSize)
{
	Mesh::AreaArr maxAreas;

	// first decimate if necessary
	const bool bNoDecimation(fDecimate >= 1.f); // 采样率大于等于1则无需采样
	const bool bNoSimplification(maxArea == 0);
	if (!bNoDecimation) {
		if (fDecimate > 0.f) {
			// decimate to the desired resolution
			// 下采样
			scene.mesh.Clean(fDecimate, 0.f, false, nCloseHoles, 0, false);
			scene.mesh.Clean(1.f, 0.f, false, nCloseHoles, 0, true);

			#ifdef MESHOPT_ENSUREEDGESIZE
			// make sure there are no edges too small or too long
			// 保证边界不会过大过短
			if (nEnsureEdgeSize > 0 && bNoSimplification) {
				scene.mesh.EnsureEdgeSize();
				scene.mesh.Clean(1.f, 0.f, false, nCloseHoles, 0, true);
			}
			#endif

			// re-map vertex and camera faces
			// 前面采样后，顶点和face会发生变化，需重新计算
			ListVertexFacesPre();
		} else {
			// extract array of faces viewed by each camera
			// 提取被每个图像看到的所有面
			ListCameraFaces();

			// estimate the faces' area that have big projection areas in both images of a pair
			// 计算最大投影面积
			ListFaceAreas(maxAreas);
			ASSERT(!maxAreas.IsEmpty());

			const float fMaxArea((float)(maxArea > 0 ? maxArea : 64));
			const float fMedianArea(6.f*(float)Mesh::AreaArr(maxAreas).GetMedian()); //取中值
			// 如果投影面积均值比要求的最大值小，说明网格过于稠密需要下采样
			if (fMedianArea < fMaxArea) { 
				maxAreas.Empty();

				// decimate to the auto detected resolution
				// 下采样
				scene.mesh.Clean(MAXF(0.1f, fMedianArea/fMaxArea), 0.f, false, nCloseHoles, 0, false);
				scene.mesh.Clean(1.f, 0.f, false, nCloseHoles, 0, true);

				#ifdef MESHOPT_ENSUREEDGESIZE
				// make sure there are no edges too small or too long
				// 保证边界不会过大过短
				if (nEnsureEdgeSize > 0 && bNoSimplification) {
					scene.mesh.EnsureEdgeSize();
					scene.mesh.Clean(1.f, 0.f, false, nCloseHoles, 0, true);
				}
				#endif

				// re-map vertex and camera faces
				ListVertexFacesPre();
			}
		}
	}
	if (bNoSimplification)
		return;
	// 前面采样后，顶点和face会发生变化，需重新计算
	if (maxAreas.IsEmpty()) {
		// extract array of faces viewed by each camera
		ListCameraFaces();

		// estimate the faces' area that have big projection areas in both images of a pair
		ListFaceAreas(maxAreas);
	}

	// subdivide mesh faces if its projection area is bigger than the given number of pixels
	// 如果投影面积比给的阈值大，说明网格比较稀疏，需进行网格细分提高分辨率
	const size_t numVertsOld(vertices.GetSize());
	const size_t numFacesOld(faces.GetSize());
	scene.mesh.Subdivide(maxAreas, maxArea);

	#ifdef MESHOPT_ENSUREEDGESIZE
	// make sure there are no edges too small or too long
	#if MESHOPT_ENSUREEDGESIZE==1
	if ((nEnsureEdgeSize == 1 && !bNoDecimation) || nEnsureEdgeSize > 1)
	#endif
	{
		scene.mesh.EnsureEdgeSize();
		scene.mesh.Clean(1.f, 0.f, false, nCloseHoles, 0, true);
	}
	#endif

	// re-map vertex and camera faces
	// 重新计算提取每个顶点的相关faces即三角face的三个顶点中包含该顶点的所有faces
	ListVertexFacesPre();

	DEBUG_EXTRA("Mesh subdivided: %u/%u -> %u/%u vertices/faces", numVertsOld, numFacesOld, vertices.GetSize(), faces.GetSize());

	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (VERBOSITY_LEVEL > 3)
		scene.mesh.Save(MAKE_PATH("MeshSubdivided.ply"));
	#endif
}


// score mesh using photo-consistency
// and compute vertices gradient using analytical method
// 计算顶点调整量使得loss更小mesh更准确
/**
 * @brief 利用光度一致性（ZNCC）和正则项构建优化目标函数，计算每个顶点的梯度。公式推导见课件
 * 
 * @param[out] gradients 每个顶点的梯度
 * @return double 分数
 */
double MeshRefine::ScoreMesh(double* gradients)
{
	// extract array of faces viewed by each camera
	// 提取每个相机看到的faces
	ListCameraFaces();

	// compute face normals
	// 计算faces 的法线
	scene.mesh.ComputeNormalFaces();

	// for each pair of images, compute a photo-consistency score
	// between the reference image and the pixels of the second image
	// projected in the reference image through the mesh surface
	// Step 1 通过mesh投影到reference图像上的faces,投影到ref和src像素上，对每一对图像（reference和source图像）计算一个光度一致性分数 
	scorePhoto = 0;
	photoGrad.Resize(vertices.GetSize());
	photoGrad.Memset(0);
	photoGradNorm.Resize(vertices.GetSize());
	photoGradNorm.Memset(0);
	if (!vertexDepth.IsEmpty()) {
		ASSERT(vertexDepth.GetSize() == vertices.GetSize());
		vertexDepth.MemsetValue(FLT_MAX);
	}
	ASSERT(events.IsEmpty());
	FOREACHPTR(pPair, pairs) {
		ASSERT(pPair->i < pPair->j);
		switch (nAlternatePair) {
		case 1:
			events.AddEvent(iteration%2 ? new EVTProcessPair(pPair->j,pPair->i) : new EVTProcessPair(pPair->i,pPair->j));
			break;
		case 2:
			events.AddEvent(new EVTProcessPair(pPair->i, pPair->j));
			break;
		case 3:
			events.AddEvent(new EVTProcessPair(pPair->j, pPair->i));
			break;
		default:
			for (int ip=0; ip<2; ++ip)
				events.AddEvent(ip ? new EVTProcessPair(pPair->j,pPair->i) : new EVTProcessPair(pPair->i,pPair->j));
		}
	}
	WaitThreadWorkers(nAlternatePair ? pairs.GetSize() : pairs.GetSize()*2);

	// loop through all vertices and compute the smoothing score
	// Step 2 循环mesh上所有的顶点，计算平滑分数主要是计算顶点Laplace: grad1=1/NΣvi-v,(vi 是v的one-ring的所有点)
	scoreSmooth = 0;
	const VIndex idxStep((vertices.GetSize()+(VIndex)threads.GetSize()-1)/(VIndex)threads.GetSize());
	smoothGrad1.Resize(vertices.GetSize());
	{
	ASSERT(events.IsEmpty());
	VIndex idx(0);
	while (idx<vertices.GetSize()) {
		const VIndex idxNext(MINF(idx+idxStep, vertices.GetSize()));
		events.AddEvent(new EVTSmoothVertices1(idx, idxNext));
		idx = idxNext;
	}
	WaitThreadWorkers(threads.GetSize());
	}
	// loop through all vertices and compute the smoothing gradient
	// Step 3 循环所有顶点计算平滑梯度,计算的是图像梯度的梯度*权重 grad2=nrm*(1/N*Σgradvi-gradv)
	// nrm=1/(1+1/Nisum(1/Nj))
	smoothGrad2.Resize(vertices.GetSize());
	{
	ASSERT(events.IsEmpty());
	VIndex idx(0);
	while (idx<vertices.GetSize()) {
		const VIndex idxNext(MINF(idx+idxStep, vertices.GetSize()));
		events.AddEvent(new EVTSmoothVertices2(idx, idxNext));
		idx = idxNext;
	}
	WaitThreadWorkers(threads.GetSize());
	}

	// set the final gradient as the combination of photometric and smoothness gradients
	// Step 4 结合光度一致性和平滑梯度计算最终的梯度
	if (ratioRigidityElasticity >= 1.f) {
		FOREACH(v, vertices)
			((Point3d*)gradients)[v] = photoGradNorm[v] > 0 ?
				Cast<double>(photoGrad[v]/photoGradNorm[v] + smoothGrad2[v]*weightRegularity) :
				Cast<double>(smoothGrad2[v]*weightRegularity);
	} else {
		// compute smoothing gradient as a combination of level 1 and 2 of the Laplacian operator;
		// (see page 105 of "Stereo and Silhouette Fusion for 3D Object Modeling from Uncalibrated Images Under Circular Motion" C. Hernandez, 2004)
		// 平滑梯度采用Laplace1和Laplace2结合
		const Real rigidity((Real(1)-ratioRigidityElasticity)*weightRegularity);
		const Real elasticity(ratioRigidityElasticity*weightRegularity);
		FOREACH(v, vertices)
			((Point3d*)gradients)[v] = photoGradNorm[v] > 0 ?
				Cast<double>(photoGrad[v]/photoGradNorm[v] + smoothGrad2[v]*elasticity - smoothGrad1[v]*rigidity) :
				Cast<double>(smoothGrad2[v]*elasticity - smoothGrad1[v]*rigidity);
	}
	return (nAlternatePair ? 0.2f : 0.1f)*scorePhoto + 0.01f*scoreSmooth;
}


// given a vertex position and a projection camera, compute the projected position and its derivative
// returns the depth
// 投影顶点到图像
template <typename TP, typename TX, typename T, typename TJ>
T MeshRefine::ProjectVertex(const TP* P, const TX* X, T* x, TJ* jacobian)
{
	const TX&  x1(X[0]);
	const TX&  x2(X[1]);
	const TX&  x3(X[2]);

	const TP& p1_1(P[ 0]);
	const TP& p1_2(P[ 1]);
	const TP& p1_3(P[ 2]);
	const TP& p1_4(P[ 3]);
	const TP& p2_1(P[ 4]);
	const TP& p2_2(P[ 5]);
	const TP& p2_3(P[ 6]);
	const TP& p2_4(P[ 7]);
	const TP& p3_1(P[ 8]);
	const TP& p3_2(P[ 9]);
	const TP& p3_3(P[10]);
	const TP& p3_4(P[11]);

	const TP t5(p3_4+p3_1*x1+p3_2*x2+p3_3*x3);
	const TP t6(1.0/t5);
	const TP t10(p1_4+p1_1*x1+p1_2*x2+p1_3*x3);
	const TP t11(t10*t6);
	const TP t15(p2_4+p2_1*x1+p2_2*x2+p2_3*x3);
	const TP t16(t15*t6);
	x[0] = T(t11);
	x[1] = T(t16);
	if (jacobian) {
		jacobian[0] = TJ((p1_1-p3_1*t11)*t6);
		jacobian[1] = TJ((p1_2-p3_2*t11)*t6);
		jacobian[2] = TJ((p1_3-p3_3*t11)*t6);
		jacobian[3] = TJ((p2_1-p3_1*t16)*t6);
		jacobian[4] = TJ((p2_2-p3_2*t16)*t6);
		jacobian[5] = TJ((p2_3-p3_3*t16)*t6);
	}
	return T(t5);
}


// check if any of the depths surrounding the given coordinate is similar to the given value
bool MeshRefine::IsDepthSimilar(const DepthMap& depthMap, const Point2f& pt, Depth z)
{
	const ImageRef tl(FLOOR2INT(pt));
	for (int x=0; x<2; ++x) {
		for (int y=0; y<2; ++y) {
			const ImageRef ir(tl.x+x, tl.y+y);
			if (!depthMap.isInsideWithBorder<int,3>(ir))
				continue;
			const Depth& depth = depthMap(ir);
			#ifndef MESHOPT_DEPTHCONSTBIAS
			if (depth <= 0 || ABS(depth-z) > z*0.01f /*!IsDepthSimilar(depth, z, 0.01f)*/)
			#else
			if (depth <= 0 || depth+MESHOPT_DEPTHCONSTBIAS < z)
			#endif
				continue;
			return true;
		}
	}
	return false;
}

// project mesh to the given camera plane
// 将mesh投影到给定的相机平面上
void MeshRefine::ProjectMesh(
	const Mesh::VertexArr& vertices, const Mesh::FaceArr& faces, const CameraFaces& cameraFaces,
	const Camera& camera, const Image8U::Size& size,
	DepthMap& depthMap, FaceMap& faceMap, BaryMap& baryMap)
{
	// init view data
	// 初始化
	depthMap.create(size);
	faceMap.create(size);
	baryMap.create(size);
	// project all triangles on this image and keep the closest ones
	// 投影mesh到图像上得到每个像素的faceid 深度
	RasterMesh rasterer(vertices, camera, depthMap, faceMap, baryMap);
	rasterer.Clear();
	for (auto idxFace : cameraFaces) {
		const Face& facet = faces[idxFace];
		rasterer.idxFace = idxFace;
		rasterer.Project(facet);
	}
}

// project image from view B to view A through the mesh;
// the projected image is stored in imageA
// (imageAB is assumed to be initialize to the right size)
// 通过mesh(depthA)将B 图像投影到A上，存储在imageA
void MeshRefine::ImageMeshWarp(
	const DepthMap& depthMapA, const Camera& cameraA,
	const DepthMap& depthMapB, const Camera& cameraB,
	const Image32F& imageB, Image32F& imageA, BitMatrix& mask)
{
	ASSERT(!imageA.empty());
	typedef Sampler::Linear<float> Sampler; // 用来像素线性插值
	const Sampler sampler;
	mask.create(imageA.size());
	mask.memset(0);
	for (int j=0; j<depthMapA.rows; ++j) {
		for (int i=0; i<depthMapA.cols; ++i) {
			const Depth& depthA = depthMapA(j,i);
			if (depthA <= 0)
				continue;
			// A的depth计算世界坐标系下坐标X
			const Point3 X(cameraA.TransformPointI2W(Point3(i,j,depthA)));
			const Point3f ptC(cameraB.TransformPointW2C(X)); // 投影到B上得到相机坐标系下的3D坐标
			const Point2f pt(cameraB.TransformPointC2I(ptC)); // 像素坐标
			if (!IsDepthSimilar(depthMapB, pt, ptC.z)) // 如果投影到B的深度与当前深度不相似有可能是遮挡区域或计算错误
				continue;
			imageA(j,i) = imageB.sample<Sampler,Sampler::Type>(sampler, pt);
			mask.set(j,i); // 记录成功warp的像素坐标
		}
	}
}

// compute local variance for each image pixel
// 计算每个像素的局部方差
void MeshRefine::ComputeLocalVariance(const Image32F& image, const BitMatrix& mask, TImage<Real>& imageMean, TImage<Real>& imageVar)
{
	ASSERT(image.size() == mask.size());
	imageMean.create(image.size());
	imageVar.create(image.size());
	imageMean.memset(0);
	imageVar.memset(0);
	const int RowsEnd(image.rows-HalfSize); 
	const int ColsEnd(image.cols-HalfSize);
	const int n(SQUARE(HalfSize*2+1));
	DEC_Image(double, imageSum);
	DEC_Image(double, imageSumSq); // 像素值平方积分图
	// 计算积分图，像素值平方的积分图
	#if CV_MAJOR_VERSION > 2
	cv::integral(image, imageSum, imageSumSq, CV_64F, CV_64F);
	#else
	cv::integral(image, imageSum, imageSumSq, CV_64F);
	#endif
	for (int r=HalfSize; r<RowsEnd; ++r) {
		for (int c=HalfSize; c<ColsEnd; ++c) {
			if (!mask(r,c))
				continue;
			// 计算均值
			imageMean(r,c) = (Real)((
				imageSum(r+HalfSize+1, c+HalfSize+1) -
				imageSum(r+HalfSize+1, c-HalfSize  ) -
				imageSum(r-HalfSize,   c+HalfSize+1) +
				imageSum(r-HalfSize,   c-HalfSize  ) ) * (1.0/(double)n));
		}
	}
	DST_Image(imageSum);
	for (int r=HalfSize; r<RowsEnd; ++r) {
		for (int c=HalfSize; c<ColsEnd; ++c) {
			if (!mask(r,c))
				continue;
			const Real sumSq = (Real)((
				imageSumSq(r+HalfSize+1, c+HalfSize+1) -
				imageSumSq(r+HalfSize+1, c-HalfSize  ) -
				imageSumSq(r-HalfSize,   c+HalfSize+1) +
				imageSumSq(r-HalfSize,   c-HalfSize  ) ) * (1.0/(double)n));
			// 计算方差 var=1/n*ΣI^2 - mean(I)*mean(I) 简化版
			imageVar(r,c) = MAXF(sumSq-SQUARE(imageMean(r,c)), Real(0.0001));
		}
	}
	DST_Image(imageSumSq);
}

// compute local ZNCC and its gradient for each image pixel
// zncc=((ΣIa*Ib)-mean(Ia*Ib))/sqrt((ΣIa^2-mean(Ia)*mean(Ia))*(ΣIb^2-mean(Ib)*mean(Ib)))
// 计算NCC和梯度
float MeshRefine::ComputeLocalZNCC(
	const Image32F& imageA, const TImage<Real>& imageMeanA, const TImage<Real>& imageVarA,
	const Image32F& imageB, const TImage<Real>& imageMeanB, const TImage<Real>& imageVarB,
	const BitMatrix& mask, TImage<Real>& imageZNCC, TImage<Real>& imageDZNCC)
{
	ASSERT(imageA.size() == mask.size() && imageB.size() == mask.size() && !mask.empty());
	float score(0);
	imageZNCC.create(mask.size());
	imageDZNCC.create(mask.size());
	const int RowsEnd(mask.rows-HalfSize);
	const int ColsEnd(mask.cols-HalfSize);
	const int n(SQUARE(HalfSize*2+1));
	imageZNCC.memset(0);
	DEC_Image(double, imageABSum);
	{
		DEC_Image(float, imageAB);
		// Ia*Ib
		cv::multiply(imageA, imageB, imageAB);
		cv::integral(imageAB, imageABSum, CV_64F);
		DST_Image(imageAB);
	}
	DEC_Image(Real, imageInvSqrtVAVB);
	imageInvSqrtVAVB.create(mask.size());
	for (int r=HalfSize; r<RowsEnd; ++r) {
		for (int c=HalfSize; c<ColsEnd; ++c) {
			if (!mask(r, c))
				continue;
			const Real cv = (Real)((
				imageABSum(r+HalfSize+1, c+HalfSize+1) -
				imageABSum(r+HalfSize+1, c-HalfSize  ) -
				imageABSum(r-HalfSize,   c+HalfSize+1) +
				imageABSum(r-HalfSize,   c-HalfSize  ) ) * (1.0/(double)n));
			const Real invSqrtVAVB(Real(1)/SQRT(imageVarA(r,c)*imageVarB(r,c)));
			// zncc=((ΣIa*Ib)-mean(Ia*Ib))/sqrt((ΣIa^2-mean(Ia)*mean(Ia))*(ΣIb^2-mean(Ib)*mean(Ib)))
			imageZNCC(r,c) = (cv - imageMeanA(r,c)*imageMeanB(r,c)) * invSqrtVAVB;
			imageInvSqrtVAVB(r,c) = invSqrtVAVB;
		}
	}
	// 计算ZNCC梯度dZNCC=(Ia-mean(Ia))/sqrt((ΣIa^2-mean(Ia)*mean(Ia))*(ΣIb^2-mean(Ib)*mean(Ib))) - (Ib-mean(Ib))*zncc/ΣIb^2-mean(Ib)*mean(Ib)
	DST_Image(imageABSum);
	imageDZNCC.memset(0);
	for (int r=HalfSize; r<RowsEnd; ++r) {
		for (int c=HalfSize; c<ColsEnd; ++c) {
			if (!mask(r,c))
				continue;
			#if 1
			const Real ZNCC(imageZNCC(r,c));
			const Real invSqrtVAVB(imageInvSqrtVAVB(r,c));
			const Real ZNCCinvVB(ZNCC/imageVarB(r,c));
			const Real dZNCC((Real)imageA(r,c)*invSqrtVAVB - (Real)imageB(r,c)*ZNCCinvVB + imageMeanB(r,c)*ZNCCinvVB - imageMeanA(r,c)*invSqrtVAVB);
			#else
			Real sumA(0), sumB(0), sumC(0);
			int n(0);
			for (int i=-HalfSize; i<=HalfSize; ++i) {
				const int rw(r+i);
				for (int j=-HalfSize; j<=HalfSize; ++j) {
					const int cw(c+j);
					if (!mask(rw,cw))
						continue;
					const Real invSqrtVAVB(imageInvSqrtVAVB(rw,cw));
					const Real ZNCCinvVB(imageZNCC(rw,cw)/imageVarB(rw,cw));
					sumA += invSqrtVAVB;
					sumB -= ZNCCinvVB;
					sumC += imageMeanB(rw,cw)*ZNCCinvVB - imageMeanA(rw,cw)*invSqrtVAVB;
					++n;
				}
			}
			if (n == 0)
				continue;
			const Real dZNCC((sumA*imageA(r,c) + sumB*imageB(r,c) + sumC)/(Real)n);
			const Real ZNCC(imageZNCC(r,c));
			#endif
			const Real minVAVB(MINF(imageVarA(r,c),imageVarB(r,c)));
			const Real ReliabilityFactor(minVAVB/(minVAVB+Real(0.0015)));
			imageDZNCC(r,c) = -ReliabilityFactor*dZNCC;
			score += (float)(ReliabilityFactor*(Real(1)-ZNCC));
		}
	}
	DST_Image(imageInvSqrtVAVB);
	return score;
}

// compute the photometric gradient for all vertices seen by an image pair
// !!! v=v+n*d 计算光度梯度,推导最小二乘优化求解过程，链式法则 参考slam第二版VO里面优化T的过程
void MeshRefine::ComputePhotometricGradient(
	const Mesh::FaceArr& faces, const Mesh::NormalArr& normals,
	const DepthMap& depthMapA, const FaceMap& faceMapA, const BaryMap& baryMapA, const Camera& cameraA,
	const Camera& cameraB, const View& viewB,
	const TImage<Real>& imageDZNCC, const BitMatrix& mask, GradArr& photoGrad, UnsignedArr& photoGradNorm, Real RegularizationScale)
{
	ASSERT(faces.GetSize() == normals.GetSize() && !faces.IsEmpty());
	ASSERT(depthMapA.size() == mask.size() && faceMapA.size() == mask.size() && baryMapA.size() == mask.size() && imageDZNCC.size() == mask.size() && !mask.empty());
	ASSERT(viewB.image.size() == viewB.imageGrad.size() && !viewB.image.empty());
	const int RowsEnd(mask.rows-HalfSize);
	const int ColsEnd(mask.cols-HalfSize);
	typedef Sampler::Linear<View::Grad::Type> Sampler;
	const Sampler sampler;
	TMatrix<Real,2,3> xJac;
	Point2f xB;
	photoGrad.Memset(0);
	photoGradNorm.Memset(0);
	// 逐个像素计算
	for (int r=HalfSize; r<RowsEnd; ++r) {
		for (int c=HalfSize; c<ColsEnd; ++c) {
			if (!mask(r,c))
				continue;
			const FIndex idxFace(faceMapA(r,c));
			ASSERT(idxFace != NO_ID);
			const Grad N(normals[idxFace]);
			const Point3 rayA(cameraA.RayPoint(Point2(c,r))); // view ray
			const Grad dA(normalized(rayA));
			const Real Nd(N.dot(dA));
			#if 1
			if (Nd > -0.1)
				continue;
			#endif
			const Depth depthA(depthMapA(r,c));
			ASSERT(depthA > 0);
			const Point3 X(rayA*REAL(depthA)+cameraA.C);
			// project point in second image and
			// projection Jacobian matrix in the second image of the 3D point on the surface
			// !!! 将点投影到第二个图像上 ，雅可比矩阵计算 todo 
			const float depthB(ProjectVertex(cameraB.P.val, X.ptr(), xB.ptr(), xJac.val));
			ASSERT(depthB > 0);
			// compute gradient in image B
			// 计算B图像的梯度
			const TMatrix<Real,1,2> gB(viewB.imageGrad.sample<Sampler,View::Grad>(sampler, xB));
			// compute gradient scale
			// 计算梯度的尺度，沿法线方向
			const Real dZNCC(imageDZNCC(r,c));
			const Real sg((gB*(xJac*(const TMatrix<Real,3,1>&)dA))(0)*dZNCC*RegularizationScale/Nd);
			// add gradient to the three vertices
			// 将梯度应用到三个顶点上
			const Face& face(faces[idxFace]);
			const Point3f& b(baryMapA(r,c));
			for (int v=0; v<3; ++v) {
				const Grad g(N*(sg*(Real)b[v]));// 法线方向
				const VIndex idxVert(face[v]);
				photoGrad[idxVert] += g;
				++photoGradNorm[idxVert];
			}
		}
	}
}

// computes the discrete analog of the Laplacian using
// the umbrella-operator on the first triangle ring at each point
// 在顶点 one-ring上计算Laplace
float MeshRefine::ComputeSmoothnessGradient1(
	const Mesh::VertexArr& vertices, const Mesh::VertexVerticesArr& vertexVertices, const BoolArr& vertexBoundary,
	GradArr& smoothGrad1, VIndex idxStart, VIndex idxEnd)
{
	ASSERT(!vertices.IsEmpty() && vertices.GetSize() == vertexVertices.GetSize() && vertices.GetSize() == smoothGrad1.GetSize());
	float score(0);
	for (VIndex idxV=idxStart; idxV<idxEnd; ++idxV) {
		Grad& grad = smoothGrad1[idxV];
		grad = Grad::ZERO;
		#if 1
		if (vertexBoundary[idxV])
			continue;
		#endif
		const Mesh::VertexIdxArr& verts = vertexVertices[idxV]; //当前顶点周围相邻的one-ring点（包含该顶点的所有faces上的点）
		if (verts.IsEmpty())
			continue;
		// 邻域点求和
		FOREACH(v, verts)
			grad += Cast<Real>(vertices[verts[v]]);
		// grad1=1/NΣvi-v,(vi 是v的one-ring的所有点)
		grad = grad/(Real)verts.GetSize() - Cast<Real>(vertices[idxV]);
		const float regularityScore((float)norm(grad));
		ASSERT(ISFINITE(regularityScore));
		score += regularityScore;
	}
	return score;
}
// same as above, but used to compute level 2;
// normalized as in "Stereo and Silhouette Fusion for 3D Object Modeling from Uncalibrated Images Under Circular Motion" C. Hernandez, 2004
// 同上，计算的是图像梯度的梯度*权重 grad2=nrm*(1/N*Σgradvi-gradv)
void MeshRefine::ComputeSmoothnessGradient2(
	const GradArr& smoothGrad1, const Mesh::VertexVerticesArr& vertexVertices, const BoolArr& vertexBoundary,
	GradArr& smoothGrad2, VIndex idxStart, VIndex idxEnd)
{
	ASSERT(!smoothGrad1.IsEmpty() && smoothGrad1.GetSize() == vertexVertices.GetSize() && smoothGrad1.GetSize() == smoothGrad2.GetSize());
	for (VIndex idxV=idxStart; idxV<idxEnd; ++idxV) {
		Grad& grad = smoothGrad2[idxV];
		grad = Grad::ZERO;
		#if 1
		if (vertexBoundary[idxV])
			continue;
		#endif
		const Mesh::VertexIdxArr& verts = vertexVertices[idxV];
		if (verts.IsEmpty())
			continue;
		Real w(0);
		// 计算邻域点（one-ring）的梯度和，记录邻域点的邻域(one-ring)个数的逆之和
		FOREACH(v, verts) {
			const VIndex idxVert(verts[v]);
			grad += smoothGrad1[idxVert];
			const VIndex numVert(vertexVertices[idxVert].GetSize());
			if (numVert > 0)
				w += Real(1)/(Real)numVert;
		}
		const Real numVert((Real)verts.GetSize());
		const Real nrm(Real(1)/(Real(1)+w/numVert));
		grad = grad*(nrm/numVert) - smoothGrad1[idxV]*nrm;
	}
}


void* MeshRefine::ThreadWorkerTmp(void* arg) {
	MeshRefine& refine = *((MeshRefine*)arg);
	refine.ThreadWorker();
	return NULL;
}
void MeshRefine::ThreadWorker()
{
	while (true) {
		CAutoPtr<Event> evt(events.GetEvent());
		switch (evt->GetID()) {
		case EVT_JOB:
			evt->Run(this);
			break;
		case EVT_CLOSE:
			return;
		default:
			ASSERT("Should not happen!" == NULL);
		}
		sem.Signal();
	}
}
void MeshRefine::WaitThreadWorkers(size_t nJobs)
{
	while (nJobs-- > 0)
		sem.Wait();
	ASSERT(events.IsEmpty());
}
void MeshRefine::ThSelectNeighbors(uint32_t idxImage, std::unordered_set<uint64_t>& mapPairs, unsigned nMaxViews)
{
	// keep only best neighbor views
	// 仅选择一个最好的邻域
	const float fMinArea(0.1f);
	const float fMinScale(0.2f), fMaxScale(3.2f);
	const float fMinAngle(FD2R(2.5f)), fMaxAngle(FD2R(45.f));
	const Image& imageData = images[idxImage];
	if (!imageData.IsValid())
		return;
	ViewScoreArr neighbors(imageData.neighbors);
	Scene::FilterNeighborViews(neighbors, fMinArea, fMinScale, fMaxScale, fMinAngle, fMaxAngle, nMaxViews);
	Lock l(cs);
	FOREACHPTR(pNeighbor, neighbors) {
		ASSERT(images[pNeighbor->idx.ID].IsValid());
		mapPairs.insert(MakePairIdx((uint32_t)idxImage, pNeighbor->idx.ID));
	}
}
// resize图像，计算梯度，滤波（可选）
void MeshRefine::ThInitImage(uint32_t idxImage, Real scale, Real sigma)
{
	Image& imageData = images[idxImage];
	if (!imageData.IsValid())
		return;
	// load and init image
	// resize图像
	unsigned level(nResolutionLevel);
	const unsigned imageSize(imageData.RecomputeMaxResolution(level, nMinResolution));
	if ((imageData.image.empty() || MAXF(imageData.width,imageData.height) != imageSize) && !imageData.ReloadImage(imageSize))
		ABORT("can not load image");
	View& view = views[idxImage];
	Image32F& img = view.image;
	imageData.image.toGray(img, cv::COLOR_BGR2GRAY, true);
	imageData.image.release();
	if (sigma > 0)
		cv::GaussianBlur(img, img, cv::Size(), sigma);
	if (scale < 1.0) {
		cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_AREA);
		imageData.width = img.width(); imageData.height = img.height();
	}
	// 更新相机内参
	imageData.UpdateCamera(scene.platforms);
	if (!nReduceMemory) {
		// compute image mean and variance
		// 如果不需要减少内存消耗可以先计算图像在局部窗口内的均值和方差
		ComputeLocalVariance(img, BitMatrix(img.size(), 0xFF), view.imageMean, view.imageVar);
	}
	// compute image gradient
	// 计算图像梯度存储到imageGrad
	typedef View::Grad::Type GradType;
	TImage<GradType> grad[2];
	#if 0
	cv::Sobel(img, grad[0], cv::DataType<GradType>::type, 1, 0, 3, 1.0/8.0);
	cv::Sobel(img, grad[1], cv::DataType<GradType>::type, 0, 1, 3, 1.0/8.0);
	#elif 1
	// 分别计算x,y方向的梯度
	const TMatrix<GradType,3,5> kernel(CreateDerivativeKernel3x5());
	cv::filter2D(img, grad[0], cv::DataType<GradType>::type, kernel);
	cv::filter2D(img, grad[1], cv::DataType<GradType>::type, kernel.t());
	#else
	const TMatrix<GradType,5,7> kernel(CreateDerivativeKernel5x7());
	cv::filter2D(img, grad[0], cv::DataType<GradType>::type, kernel);
	cv::filter2D(img, grad[1], cv::DataType<GradType>::type, kernel.t());
	#endif
	// 将grad的两个单通道融合为一个2通道的矩阵imageGrad
	cv::merge(grad, 2, view.imageGrad);
}
void MeshRefine::ThProjectMesh(uint32_t idxImage, const CameraFaces& cameraFaces)
{
	const Image& imageData = images[idxImage];
	if (!imageData.IsValid())
		return;
	// project mesh to the given camera plane
	View& view = views[idxImage];
	ProjectMesh(vertices, faces, cameraFaces, imageData.camera, view.image.size(),
				view.depthMap, view.faceMap, view.baryMap);
}
// 
/**
 * @brief 计算左右图像的光度一致性用来计算下降梯度 计算的公式推导见课件
 * 
 * @param[in] idxImageA 左图像
 * @param[in] idxImageB 右图像
 */
void MeshRefine::ThProcessPair(uint32_t idxImageA, uint32_t idxImageB)
{
	// fetch view A data
	// 取viewA 相关数据
	const Image& imageDataA = images[idxImageA];
	ASSERT(imageDataA.IsValid());
	const View& viewA = views[idxImageA]; 
	const BaryMap& baryMapA = viewA.baryMap;   // 重心
	const FaceMap& faceMapA = viewA.faceMap;   // pixel对应的faceid
	const DepthMap& depthMapA = viewA.depthMap;// 深度图
	const Image32F& imageA = viewA.image;      // 图像
	const Camera& cameraA = imageDataA.camera; // 相机参数
	// fetch view B data
	// 取viewB相关数据
	const Image& imageDataB = images[idxImageB];
	ASSERT(imageDataB.IsValid());
	const View& viewB = views[idxImageB];
	const DepthMap& depthMapB = viewB.depthMap;
	const Image32F& imageB = viewB.image;
	const Camera& cameraB = imageDataB.camera;
	// warp imageB to imageA using the mesh
	// 用mesh warp B 到A图像
	DEC_BitMatrix(mask);
	DEC_Image(float, imageAB);
	imageA.copyTo(imageAB);
	// warp b->a
	ImageMeshWarp(depthMapA, cameraA, depthMapB, cameraB, imageB, imageAB, mask);
	// compute ZNCC and its gradient
	// 计算 ZNCC和其梯度
	const TImage<Real> *imageMeanA, *imageVarA;
	if (nReduceMemory) {
		DEC_Image(Real, _imageMeanA);
		DEC_Image(Real, _imageVarA);
		// 计算局部均值和方差
		ComputeLocalVariance(viewA.image, mask, _imageMeanA, _imageVarA);
		imageMeanA = &_imageMeanA;
		imageVarA = &_imageVarA;
	} else {
		imageMeanA = &viewA.imageMean;
		imageVarA = &viewA.imageVar;
	}
	DEC_Image(Real, imageMeanAB);
	DEC_Image(Real, imageVarAB);
	// 计算均值和方差
	ComputeLocalVariance(imageAB, mask, imageMeanAB, imageVarAB);
	DEC_Image(Real, imageZNCC);
	DEC_Image(Real, imageDZNCC);
	// 计算NCC
	const float score(ComputeLocalZNCC(imageA, *imageMeanA, *imageVarA, imageAB, imageMeanAB, imageVarAB, mask, imageZNCC, imageDZNCC));
	#ifdef MESHOPT_TYPEPOOL
	DST_Image(imageZNCC);
	DST_Image(imageVarAB);
	DST_Image(imageMeanAB);
	if (nReduceMemory) {
		DST_Image(*((TImage<Real>*)imageMeanA));
		DST_Image(*((TImage<Real>*)imageVarA));
	}
	DST_Image(imageAB);
	#endif
	// compute field gradient
	GradArr _photoGrad(photoGrad.GetSize());
	UnsignedArr _photoGradNorm(photoGrad.GetSize());
	const Real RegularizationScale((Real)((REAL)(imageDataA.avgDepth*imageDataB.avgDepth)/(cameraA.GetFocalLength()*cameraB.GetFocalLength())));
	// 计算光度梯度
	ComputePhotometricGradient(faces, faceNormals, depthMapA, faceMapA, baryMapA, cameraA, cameraB, viewB, imageDZNCC, mask, _photoGrad, _photoGradNorm, RegularizationScale);
	DST_Image(imageDZNCC);
	DST_BitMatrix(mask);
	Lock l(cs);
	// 赋值
	if (vertexDepth.IsEmpty()) {
		FOREACH(i, photoGrad) {
			if (_photoGradNorm[i] > 0) {
				photoGrad[i] += _photoGrad[i];
				photoGradNorm[i] += 1.f;
			}
		}
	} else {
		const float depth(MINF(imageDataA.avgDepth, imageDataB.avgDepth));
		FOREACH(i, photoGrad) {
			if (_photoGradNorm[i] > 0) {
				photoGrad[i] += _photoGrad[i];
				photoGradNorm[i] += 1.f;
				if (vertexDepth[i] > depth)
					vertexDepth[i] = depth;
			}
		}
	}
	scorePhoto += (float)RegularizationScale*score;
}
void MeshRefine::ThSmoothVertices1(VIndex idxStart, VIndex idxEnd)
{
	// 计算顶点梯度
	const float score(ComputeSmoothnessGradient1(vertices, vertexVertices, vertexBoundary, smoothGrad1, idxStart, idxEnd));
	Lock l(cs);
	scoreSmooth += score;
}
void MeshRefine::ThSmoothVertices2(VIndex idxStart, VIndex idxEnd)
{
	// 计算顶点梯度的梯度
	ComputeSmoothnessGradient2(smoothGrad1, vertexVertices, vertexBoundary, smoothGrad2, idxStart, idxEnd);
}
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

#ifdef MESHOPT_CERES

#pragma push_macro("LOG")
#undef LOG
#pragma push_macro("CHECK")
#undef CHECK
#pragma push_macro("ERROR")
#undef ERROR
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#pragma pop_macro("ERROR")
#pragma pop_macro("CHECK")
#pragma pop_macro("LOG")

namespace ceres {
// 构建最小二乘问题
class MeshProblem : public FirstOrderFunction, public IterationCallback
{
public:
	MeshProblem(MeshRefine& _refine) : refine(_refine), params(refine.vertices.GetSize()*3) {
		// init params
		FOREACH(i, refine.vertices)
			*((Point3d*)params.Begin()+i) = refine.vertices[i];
	}
	virtual ~MeshProblem() {}

	void ApplyParams() const {
		FOREACH(i, refine.vertices)
			refine.vertices[i] = *((Point3d*)params.Begin()+i);
	}
	void ApplyParams(const double* parameters) const {
		memcpy(params.Begin(), parameters, sizeof(double)*params.GetSize());
		ApplyParams();
	}

	bool Evaluate(const double* const parameters, double* cost, double* gradient) const {
		// update surface parameters
		// 更新参数
		ApplyParams(parameters);
		// evaluate residuals and gradients
		// 计算梯度
		Point3dArr gradients;
		if (!gradient) {
			gradients.Resize(refine.vertices.GetSize());
			gradient = (double*)gradients.Begin();
		}
		// 代价函数计算模型
		*cost = refine.ScoreMesh(gradient);
		return true;
	}

	CallbackReturnType operator()(const IterationSummary& summary) {
		refine.iteration = summary.iteration;
		return ceres::SOLVER_CONTINUE;
	}

	int NumParameters() const { return (int)params.GetSize(); }
	const double* GetParameters() const { return params.Begin(); }
	double* GetParameters() { return params.Begin(); }

protected:
	MeshRefine& refine;
	DoubleArr params;
};
} // namespace ceres

#endif // MESHOPT_CERES


// optimize mesh using photo-consistency  photometric-consistency
// fThPlanarVertex - threshold used to remove vertices on planar patches (percentage of the minimum depth, 0 - disable)
/**
 * @brief mesh优化，主要是采用光度一致性。代码用两种方式：一种调用Ceres库一种直接实现最小化优化。两者效果和性能均没有明显区别
 * 主要参考文献：High Accuracy and Visibility-Consistent Dense Multiview Stereo
 * 
 * @param[in] nResolutionLevel  用来refine的图片分辨率
 * @param[in] nMinResolution    用来refine的图片最小分辨率
 * @param[in] nMaxViews         最大view个数
 * @param[in] fDecimateMesh     下采样率
 * @param[in] nCloseHoles       用来补洞的阈值，洞小于该值才会补
 * @param[in] nEnsureEdgeSize   是否保持边界
 * @param[in] nMaxFaceArea      refine过程中网格三角面的面积最大值（单位是像素）如果想让最终网格稠密可以将此值设小但是会比较耗性能
 * @param[in] nScales           控制用几个图像尺度去优化mesh 默认3
 * @param[in] fScaleStep        尺度步长 默认0.5
 * @param[in] nReduceMemory     是否减少内存占用 默认1
 * @param[in] nAlternatePair    控制如何使用图像对里面图像的作为参考图像using an image pair alternatively as reference image (0 - both, 1 - alternate, 2 - only left, 3 - only right)
 * @param[in] fRegularityWeight 权重系数
 * @param[in] fRatioRigidityElasticity 
 * @param[in] fThPlanarVertex   平面顶点阈值
 * @param[in] fGradientStep     梯度下降的步长
 * @return true 
 * @return false 
 */
bool Scene::RefineMesh(unsigned nResolutionLevel, unsigned nMinResolution, unsigned nMaxViews,
					   float fDecimateMesh, unsigned nCloseHoles, unsigned nEnsureEdgeSize, unsigned nMaxFaceArea,
					   unsigned nScales, float fScaleStep,
					   unsigned nReduceMemory, unsigned nAlternatePair, float fRegularityWeight, float fRatioRigidityElasticity, float fThPlanarVertex, float fGradientStep)
{
	MeshRefine refine(*this, nReduceMemory, nAlternatePair, fRegularityWeight, fRatioRigidityElasticity, nResolutionLevel, nMinResolution, nMaxViews, nMaxThreads);
	if (!refine.IsValid())
		return false;

	// run the mesh optimization on multiple scales (coarse to fine)
	// 多尺度去优化mesh(从低分辨率到高分辨率)
	for (unsigned nScale=0; nScale<nScales; ++nScale) {
		// init images
		// Step 1 image初始化主要是resize和滤波，假设输入的图像分辨率1280*960，nResolutionLevel为1(2^1)，则用来refine的图像分辨率是640*480 
		// 多尺度refine对应的scale 0.25 0.5 1 图像分辨率 160*120 320*240 640*480
		const Real scale(POWI(fScaleStep, (int)(nScales-nScale-1)));
		const Real step(POWI(2.f, (int)(nScales-nScale))); // 用来计算图像高斯滤波sigma参数
		DEBUG_ULTIMATE("Refine mesh at: %.2f image scale", scale);
		if (!refine.InitImages(scale, Real(0.12)*step+Real(0.2)))
			return false;

		// extract array of triangles incident to each vertex
		// Step 2提取每个顶点的相关faces即三角face的三个顶点中包含该顶点的所有faces
		refine.ListVertexFacesPre();

		// automatic mesh subdivision
		// Step 3网格采样，补洞
		refine.SubdivideMesh(nMaxFaceArea, nScale == 0 ? fDecimateMesh : 1.f, nCloseHoles, nEnsureEdgeSize);

		// extract array of triangle normals
		// Step 4提取每个三角面的信息
		refine.ListVertexFacesPost();

		#if TD_VERBOSE != TD_VERBOSE_OFF
		if (VERBOSITY_LEVEL > 2)
			mesh.Save(MAKE_PATH(String::FormatString("MeshRefine%u.ply", nScales-nScale-1)));
		#endif

		// Step 5 计算最小化目标函数：两种方法：一种直接使用ceres库进行优化，一种直接实现迭代优化。
		#ifdef MESHOPT_CERES
		// Ceres求解优化问题见课件介绍 
		if (fGradientStep == 0) {
			// DefineProblem
			// 定义problem
			refine.ratioRigidityElasticity = 1.f;
			ceres::MeshProblem* problemData(new ceres::MeshProblem(refine));
			ceres::GradientProblem problem(problemData);
			// SetMinimizerOptions
			// 设置最小化优化参数
			ceres::GradientProblemSolver::Options options;
			if (VERBOSITY_LEVEL > 1) {
				options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
				//输出到cout
				options.minimizer_progress_to_stdout = true;
			} else {
				options.logging_type = ceres::LoggingType::SILENT;
				options.minimizer_progress_to_stdout = false;
			}
			// 设置优化中止条件
			options.function_tolerance = 1e-3;
			options.gradient_tolerance = 1e-7;
			options.max_num_line_search_step_size_iterations = 10;
			options.callbacks.push_back(problemData);
			ceres::GradientProblemSolver::Summary summary; // 优化信息
			// SolveProblem
			// 开始求解
			ceres::Solve(options, problem, problemData->GetParameters(), &summary);
			DEBUG_ULTIMATE(summary.FullReport().c_str());
			switch (summary.termination_type) {
			case ceres::TerminationType::NO_CONVERGENCE:
				DEBUG_EXTRA("CERES: maximum number of iterations reached!");
			case ceres::TerminationType::CONVERGENCE:
			case ceres::TerminationType::USER_SUCCESS:
				break;
			default:
				VERBOSE("CERES surface refine error: %s!", summary.message.c_str());
				return false;
			}
			ASSERT(summary.IsSolutionUsable());
			// 求解的参数应用到mesh上
			problemData->ApplyParams();
		} else
		#endif // MESHOPT_CERES
		{
			// 主要参考文献：High Accuracy and Visibility-Consistent Dense Multiview Stereo
			// loop a constant number of iterations and apply the gradient
			// 利用光度一致性优化mesh的顶点坐标，迭代指定次数
			int iters(75); // 迭代次数
			double gstep(0.4); // 优化时梯度下降的步长 
			if (fGradientStep > 1) {
				iters = FLOOR2INT(fGradientStep);
				gstep = (fGradientStep-(float)iters)*10;
			}
			// 计算迭代次数，图像分辨率越高次数越少
			iters = MAXF(iters/(int)(nScale+1),8); 
			const int iterStop(iters*7/10);
			const int iterStart(fThPlanarVertex > 0 ? iters*4/10 : INT_MAX);
			Eigen::Matrix<double,Eigen::Dynamic,3,Eigen::RowMajor> gradients(refine.vertices.GetSize(),3);
			Util::Progress progress(_T("Processed iterations"), iters);
			GET_LOGCONSOLE().Pause();
			for (int iter=0; iter<iters; ++iter) {
				refine.iteration = (unsigned)iter;
				refine.nAlternatePair = (iter+1 < iters ? nAlternatePair : 0);
				refine.ratioRigidityElasticity = (iter <= iterStop ? fRatioRigidityElasticity : 1.f);
				const bool bAdaptMesh(iter >= iterStart && (iter-iterStart)%3 == 0 && iters-iter > 5);
				// evaluate residuals and gradients
				if (bAdaptMesh)
					refine.vertexDepth.Resize(refine.vertices.GetSize());
			    // !!! 重点,实现优化计算了每一次迭代顶点的调整量
				const double cost = refine.ScoreMesh(gradients.data());
				double gv(0);
				VIndex numVertsRemoved(0);
				if (bAdaptMesh) {
					// apply gradients and
					// remove planar vertices (small gradient and almost on the center of their surrounding patch)
					// 应用梯度，移除平面上的顶点
					ASSERT(refine.vertexDepth.GetSize() == refine.vertices.GetSize());
					Mesh::VertexIdxArr vertexRemove;
					FOREACH(v, refine.vertices) {
						Vertex& vert = refine.vertices[v];
						const Point3d grad(gradients.row(v));
						vert -= Cast<Vertex::Type>(grad*gstep);
						const double gn(norm(grad)); // 梯度的幅值
						gv += gn;
						const float depth(refine.vertexDepth[v]);
						if (depth < FLT_MAX) {
							const float th(depth*fThPlanarVertex);
							// 剔除平面上的点
							if (!refine.vertexBoundary[v] && (float)gn < th && norm(refine.smoothGrad1[v]) < th)
								vertexRemove.Insert(v);
						}
					}
					if (!vertexRemove.IsEmpty()) {
						// 移除要剔除的顶点
						numVertsRemoved = vertexRemove.GetSize();
						mesh.Decimate(vertexRemove);
						refine.ListVertexFacesPost();
					}
					refine.vertexDepth.Empty();
				} else {
					// apply gradients
					// 应用梯度
					FOREACH(v, refine.vertices) {
						Vertex& vert = refine.vertices[v];
						const Point3d grad(gradients.row(v));
						// 梯度下降，调整量=梯度*乘个步长
						vert -= Cast<Vertex::Type>(grad*gstep);
						// 梯度的幅值
						gv += norm(grad);
					}
				}
				DEBUG_EXTRA("\t%2d. f: %.5f (%.4e)\tg: %.5f (%.4e - %.4e)\ts: %.3f\tv: %5u", iter+1, cost, cost/refine.vertices.GetSize(), gradients.norm(), gradients.norm()/refine.vertices.GetSize(), gv/refine.vertices.GetSize(), gstep, numVertsRemoved);
				// 调整步长 减小步长
				gstep *= 0.98;
				progress.display(iter);
			}
			GET_LOGCONSOLE().Play();
			progress.close();
		}

		#if TD_VERBOSE != TD_VERBOSE_OFF
		if (VERBOSITY_LEVEL > 2)
			mesh.Save(MAKE_PATH(String::FormatString("MeshRefined%u.ply", nScales-nScale-1)));
		#endif
	}

	return true;
} // RefineMesh
/*----------------------------------------------------------------*/
