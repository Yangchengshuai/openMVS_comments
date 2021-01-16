/*
* Scene.h
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

#ifndef _MVS_SCENE_H_
#define _MVS_SCENE_H_


// I N C L U D E S /////////////////////////////////////////////////

#include "SceneDensify.h"
#include "Mesh.h"


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace MVS {

// Forward declarations
struct MVS_API DenseDepthMapData;
/**
 * @brief scene类包含了mvs从depth计算到纹理贴图整个功能的所有函数。成员变量：相机参数，pose，图像，及重建的中间结果pointcloud mesh。
 * 
 */
class MVS_API Scene
{
public:
	PlatformArr platforms; // camera platforms, each containing the mounted cameras and all known poses
	ImageArr images; // images, each referencing a platform's camera pose
	PointCloud pointcloud; // point-cloud (sparse or dense), each containing the point position and the views seeing it
	Mesh mesh; // mesh, represented as vertices and triangles, constructed from the input point-cloud

	unsigned nCalibratedImages; // number of valid images

	unsigned nMaxThreads; // maximum number of threads used to distribute the work load

public:
	/**
	 * @brief Construct a new Scene object 构造函数
	 * 
	 * @param[in] _nMaxThreads             线程数（可根据实际情况进行设置，默认是0）
	 */
	inline Scene(unsigned _nMaxThreads=0) : nMaxThreads(Thread::getMaxThreads(_nMaxThreads)) {}

	/**
	 * @brief 内存释放，释放成员变量：platforms, images, pointcloud, mesh
	 * 
	 */
	void Release();

	/**
	 * @brief 判断成员变量mesh和pointcloud是否为空
	 * 
	 * @return true   mesh和pointcloud均为空
	 * @return false  mesh和pointcloud均不为空
	 */
	bool IsEmpty() const;
	bool LoadInterface(const String& fileName);
	bool SaveInterface(const String& fileName, int version=-1) const;

	bool LoadDMAP(const String& fileName);
	bool Import(const String& fileName);

	bool Load(const String& fileName, bool bImport=false);
	bool Save(const String& fileName, ARCHIVE_TYPE type=ARCHIVE_BINARY_ZIP) const;

	/**
	 * @brief 选择用来计算reference image的深度图的邻域帧neighborViews。主要是依据共视的3d特征点个数和特征点与两个图像的相机中心
	 * 连线的夹角来衡量计算一个score排序取分数较高的前几张图来作为source image。
	 * 
	 * @param[in] ID 				reference image的id
	 * @param[in] points            3D特征点
	 * @param[in] nMinViews         最小邻域值
	 * @param[in] nMinPointViews 	3d特征点共视的最小view数，低于这个认为不准确点不参与选择邻域的计算
	 * @param[in] fOptimAngle       角度阈值
	 * @return true 				最后选到的邻域个数不小于nMinViews
	 * @return false                最后选到的邻域个数小于nMinViews
	 */
	bool SelectNeighborViews(uint32_t ID, IndexArr& points, unsigned nMinViews=3, unsigned nMinPointViews=2, float fOptimAngle=FD2R(10));

	//!!! 这个滤波条件对于随意拍摄的序列图像很难有足够满足的邻域，一般可根据实际情况进行调整
	static bool FilterNeighborViews(ViewScoreArr& neighbors, float fMinArea=0.1f, float fMinScale=0.2f, float fMaxScale=2.4f, float fMinAngle=FD2R(3), float fMaxAngle=FD2R(45), unsigned nMaxViews=12);

	bool ExportCamerasMLP(const String& fileName, const String& fileNameScene) const;

	// Dense reconstruction
	/**
	 * @brief depth计算+点云计算的主函数：主要包含depth计算（ComputeDepthMaps，DenseReconstructionEstimate，DenseReconstructionFilter）
	 * 和点云融合
	 *
	 * @param[in] nFusionMode 融合模式：默认0计算depth并融合成点云，1是不融合只做depth计算。
	 * @return true           稠密重建成功
	 * @return false 		  重建失败
	 */
	bool DenseReconstruction(int nFusionMode=0);
	bool ComputeDepthMaps(DenseDepthMapData& data);
	void DenseReconstructionEstimate(void*);
	void DenseReconstructionFilter(void*);

	/**
	 * @brief 点云滤波，由DenseReconstruction得到的点云除了包含xyz坐标外还有view信息（能看到该点的所有帧ID）代表了当前点的可见性，
	 *实际使用时，可根据需要对点云进行滤波，根据传入的可见性阈值thRemove来控制滤波强度. 
	 *
	 * @param[in] thRemove 可见性阈值thRemove，低于该值的点被滤掉。
	 */
	void PointCloudFilter(int thRemove=-1);

	// Mesh reconstruction
	bool ReconstructMesh(float distInsert=2, bool bUseFreeSpaceSupport=true, unsigned nItersFixNonManifold=4,
						 float kSigma=2.f, float kQual=1.f, float kb=4.f,
						 float kf=3.f, float kRel=0.1f/*max 0.3*/, float kAbs=1000.f/*min 500*/, float kOutl=400.f/*max 700.f*/,
						 float kInf=(float)(INT_MAX/8));

	// Mesh refinement
	bool RefineMesh(unsigned nResolutionLevel, unsigned nMinResolution, unsigned nMaxViews, float fDecimateMesh, unsigned nCloseHoles, unsigned nEnsureEdgeSize, unsigned nMaxFaceArea, unsigned nScales, float fScaleStep, unsigned nReduceMemory, unsigned nAlternatePair, float fRegularityWeight, float fRatioRigidityElasticity, float fThPlanarVertex, float fGradientStep);
	#ifdef _USE_CUDA
	bool RefineMeshCUDA(unsigned nResolutionLevel, unsigned nMinResolution, unsigned nMaxViews, float fDecimateMesh, unsigned nCloseHoles, unsigned nEnsureEdgeSize, unsigned nMaxFaceArea, unsigned nScales, float fScaleStep, unsigned nAlternatePair, float fRegularityWeight, float fRatioRigidityElasticity, float fGradientStep);
	#endif

	// Mesh texturing
	bool TextureMesh(unsigned nResolutionLevel, unsigned nMinResolution, float fOutlierThreshold=0.f, float fRatioDataSmoothness=0.3f, bool bGlobalSeamLeveling=true, bool bLocalSeamLeveling=true, unsigned nTextureSizeMultiple=0, unsigned nRectPackingHeuristic=3, Pixel8U colEmpty=Pixel8U(255,127,39));

	#ifdef _USE_BOOST
	// implement BOOST serialization
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*version*/) {
		ar & platforms;
		ar & images;
		ar & pointcloud;
		ar & mesh;
	}
	#endif
};
/*----------------------------------------------------------------*/

} // namespace MVS

#endif // _MVS_SCENE_H_
