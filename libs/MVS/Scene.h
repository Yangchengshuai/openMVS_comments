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

// Forward declarations 提前声明depth计算的结构体
struct MVS_API DenseDepthMapData;

/**
 * @brief scene类包含了mvs从depth计算到纹理贴图整个功能的所有函数。成员变量：相机参数，pose，图像，及重建的中间结果pointcloud mesh。
 * Step 1 Dense reconstruction 深度重建
 * Step 2 Mesh reconstruction  三维曲面重建
 * Step 3 Mesh refinement      mesh 优化
 * Step 4 Mesh texturing       纹理贴图
 */
class MVS_API Scene
{
public:
	PlatformArr platforms; // 相机内参和位姿。 camera platforms, each containing the mounted cameras and all known poses
	ImageArr images;       // 图像 每帧对应的位姿在platforms中。 images, each referencing a platform's camera pose
	PointCloud pointcloud; // 点云（开始存储的是读入的稀疏点云，后续存储的是深度图计算融合后稠密点云。point-cloud (sparse or dense), each containing the point position and the views seeing it
	Mesh mesh;             // 网格模型（顶点和faces），由pointcloud计算得到的。mesh, represented as vertices and triangles, constructed from the input point-cloud

	unsigned nCalibratedImages; // 有效图像个数。 number of valid images

	unsigned nMaxThreads;       // 最大线程。  maximum number of threads used to distribute the work load

public:
	// 构造函数，创建scene对象
	inline Scene(unsigned _nMaxThreads=0) : nMaxThreads(Thread::getMaxThreads(_nMaxThreads)) {}

	// 内存释放，释放成员变量：platforms, images, pointcloud, mesh
	void Release();

	/**
	 * @brief 判断成员变量mesh和pointcloud是否为空
	 * 
	 * @return true   mesh和pointcloud均为空
	 * @return false  mesh和pointcloud均不为空
	 */
	bool IsEmpty() const;

	// 以下六个函数是输入输出参数
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
	 * @return false                最后选到的邻域个数小于nMinViews 选邻域失败
	 */
	bool SelectNeighborViews(uint32_t ID, IndexArr& points, unsigned nMinViews=3, unsigned nMinPointViews=2, float fOptimAngle=FD2R(10));

	//!!! 这个滤波条件对于随意拍摄的序列图像很难有足够满足的邻域，一般可根据实际情况进行参数调整
	static bool FilterNeighborViews(ViewScoreArr& neighbors, float fMinArea=0.1f, float fMinScale=0.2f, float fMaxScale=2.4f, float fMinAngle=FD2R(3), float fMaxAngle=FD2R(45), unsigned nMaxViews=12);

	bool ExportCamerasMLP(const String& fileName, const String& fileNameScene) const;

	// Step 1 Dense reconstruction 深度重建
	/**
	 * @brief depth计算+点云计算的主函数：主要包含depth计算（ComputeDepthMaps，DenseReconstructionEstimate，DenseReconstructionFilter）
	 * 和点云融合
 	 * @param[in] nFusionMode   控制参数：-1采用SGM/tSGM计算，只输出视差图 -2 计算加融合视差图 0是深度图计算和融合 1采用patchMatch方式 只输出深度图
     * <0是用SGM计算视差的方式计算深度，>=0用PatchMatch方式计算深度
     * @return true  计算成功
     * @return false 深度图计算失败
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

	// Step 2 Mesh reconstruction 三维曲面重建
	/**
	 * @brief mesh重建：首先用之前depth计算得到的点云，逐点插入（插入过程中如果待插入点与上一个已插入的点在view中投影足够远）构建四面体。
 	 *                  计算score四面体构建的每条边，参考论文（Multi-View Reconstruction PreservingWeakly-Supported Surfaces），最后用graphcut提取重建的曲面。
 	 * @param[in] distInsert            插入点的最近距离，待插入点与已插入点距离要不小于这个值，小于则不插入。控制插入点的密度，太密集也会影响后续计算效率。
     * @param[in] bUseFreeSpaceSupport  是否使用 free space support
     * @param[in] nItersFixNonManifold  修复mesh的非流形结构的迭代次数
     * @param[in] kSigma                表示最小可重构对象
     * @param[in] kQual     
     * @param[in] kb 
     * @param[in] kf 
     * @param[in] kRel 
     * @param[in] kAbs 
     * @param[in] kOutl 
     * @param[in] kInf                  graphcut s-t中s权重为常数默认kInf=(float)(INT_MAX/8)
     * @return true 
     * @return false 
     */
	bool ReconstructMesh(float distInsert=2, bool bUseFreeSpaceSupport=true, unsigned nItersFixNonManifold=4,
						 float kSigma=2.f, float kQual=1.f, float kb=4.f,
						 float kf=3.f, float kRel=0.1f/*max 0.3*/, float kAbs=1000.f/*min 500*/, float kOutl=400.f/*max 700.f*/,
						 float kInf=(float)(INT_MAX/8));

	// Step 3 Mesh refinement mesh优化
	/**
 	 * @brief mesh优化，主要是采用光度一致性。代码用两种方式：一种调用Ceres库一种直接实现最小化优化。两者效果和性能均没有明显区别
	 * 主要参考文献：High Accuracy and Visibility-Consistent Dense Multiview Stereo
 	 * 
 	 * @param[in] nResolutionLevel  用来refine的图片分辨率
 	 * @param[in] nMinResolution    用来refine的图片最小分辨率
 	 * @param[in] nMaxViews         最大view个数
 	 * @param[in] fDecimateMesh     下采样率
 	 * @param[in] nCloseHoles       用来补洞的阈值，洞大于该值才会补
 	 * @param[in] nEnsureEdgeSize   是否保持边界
 	 * @param[in] nMaxFaceArea      refine过程中网格三角面的面积最大值（单位是像素）如果想让最终网格稠密可以将此值设小但是会比较耗性能
 	 * @param[in] nScales           控制用几个图像尺度去优化mesh 默认3
 	 * @param[in] fScaleStep        尺度步长 默认0.5
 	 * @param[in] nReduceMemory     是否减少内存占用 默认1
 	 * @param[in] nAlternatePair    可选择的图像对
 	 * @param[in] fRegularityWeight 权重系数
 	 * @param[in] fRatioRigidityElasticity 
 	 * @param[in] fThPlanarVertex   平面顶点阈值
 	 * @param[in] fGradientStep     梯度下降的步长
 	 * @return true 
 	 * @return false 
 	 */
	bool RefineMesh(unsigned nResolutionLevel, unsigned nMinResolution, unsigned nMaxViews, float fDecimateMesh, unsigned nCloseHoles, unsigned nEnsureEdgeSize, unsigned nMaxFaceArea, unsigned nScales, float fScaleStep, unsigned nReduceMemory, unsigned nAlternatePair, float fRegularityWeight, float fRatioRigidityElasticity, float fThPlanarVertex, float fGradientStep);
	#ifdef _USE_CUDA
	bool RefineMeshCUDA(unsigned nResolutionLevel, unsigned nMinResolution, unsigned nMaxViews, float fDecimateMesh, unsigned nCloseHoles, unsigned nEnsureEdgeSize, unsigned nMaxFaceArea, unsigned nScales, float fScaleStep, unsigned nAlternatePair, float fRegularityWeight, float fRatioRigidityElasticity, float fGradientStep);
	#endif

	// Step 4 Mesh texturing 纹理贴图
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
 	 * @param[in] colEmpty             rgb颜色值，用来填充纹理图上空缺部分的颜色
 	 * @return true 
 	 * @return false 
 	 * 参考论文：Let There Be Color! Large-Scale Texturing of 3D Reconstructions
 	 */
	bool TextureMesh(unsigned nResolutionLevel, unsigned nMinResolution, float fOutlierThreshold=0.f, float fRatioDataSmoothness=0.3f, bool bGlobalSeamLeveling=true, bool bLocalSeamLeveling=true, unsigned nTextureSizeMultiple=0, unsigned nRectPackingHeuristic=3, Pixel8U colEmpty=Pixel8U(255,127,39));

	#ifdef _USE_BOOST
	// implement BOOST serialization
	// boost 序列化存储成员变量
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
