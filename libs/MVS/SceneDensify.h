/*
* SceneDensify.h
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

#ifndef _MVS_SCENEDENSIFY_H_
#define _MVS_SCENEDENSIFY_H_


// I N C L U D E S /////////////////////////////////////////////////

#include "SemiGlobalMatcher.h"


// S T R U C T S ///////////////////////////////////////////////////

namespace MVS {
	
// Forward declarations
class MVS_API Scene;
	
// structure used to compute all depth-maps
// 用来计算所有深度的类
class MVS_API DepthMapsData
{
public:
	DepthMapsData(Scene& _scene);
	~DepthMapsData();
	/**
	 * @brief 给每帧图像全局选一个最优的target image用来深度计算。
	 * 
	 * @param[in] images 		记录用来计算depth的有效帧id
	 * @param[in] imagesMap 	记录计算depth的帧在全局中id与depth计算数据结构中新id对应关系
	 * @param[in] neighborsMap 	记录每帧对应的参考帧
	 * @return true 
	 * @return false 
	 */
	bool SelectViews(IIndexArr& images, IIndexArr& imagesMap, IIndexArr& neighborsMap);
	/**
	 * @brief 给每帧选择邻域views，
	 * 
	 * @param[in] depthData 单帧深度计算相关数据
	 * @return true 
	 * @return false 
	 */
	bool SelectViews(DepthData& depthData);
	/**
	 * @brief 初始化计算深度图的图像
	 * 
	 * @param[in] depthData    深度图数据
	 * @param[in] idxNeighbor  邻域ID
	 * @param[in] numNeighbors 邻域个数
	 * @return true 
	 * @return false 
	 */
	bool InitViews(DepthData& depthData, IIndex idxNeighbor, IIndex numNeighbors);
	/**
	 * @brief 深度图初始化，主要是利用特征点进行初始化
	 * 
	 * @param[in] depthData 单帧深度计算相关数据
	 * @return true 
	 * @return false 
	 */
	bool InitDepthMap(DepthData& depthData);
	/**
	 * @brief 深度图计算
	 * 
	 * @param[in] idxImage  图像id
	 * @return true 
	 * @return false 
	 */
	bool EstimateDepthMap(IIndex idxImage);

	// 滤波
	bool RemoveSmallSegments(DepthData& depthData);
	bool GapInterpolation(DepthData& depthData);
	/**
	 * @brief 深度图滤波
	 * 
	 * @param[in] depthData 
	 * @param[in] idxNeighbors 邻域id
	 * @param[in] bAdjust true是对当前depth根据邻域投影到当前帧深度进行调整
	 * @return true 
	 * @return false 
	 */
	bool FilterDepthMap(DepthData& depthData, const IIndexArr& idxNeighbors, bool bAdjust=true);

	/**
	 * @brief depth融合，融合为一个点云
	 * 
	 * @param[in] pointcloud 融合的点云
	 * @param[in] bEstimateColor 是否重建点云的颜色信息
	 * @param[in] bEstimateNormal 是否重建点云的法线信息
	 */
	void FuseDepthMaps(PointCloud& pointcloud, bool bEstimateColor, bool bEstimateNormal);

protected:
	static void* STCALL ScoreDepthMapTmp(void*);
	static void* STCALL EstimateDepthMapTmp(void*);
	static void* STCALL EndDepthMapTmp(void*);

public:
	Scene& scene;

	DepthDataArr arrDepthData;// 存放的是每帧depth（见数据结构DepthData)和对应的Id。

	// used internally to estimate the depth-maps
	Image8U::Size prevDepthMapSize;      // 记录上一个重建的depth的大小，remember the size of the last estimated depth-map
	Image8U::Size prevDepthMapSizeTrg;   // 记录上一个重建depth的target图像大小... same for target image
	DepthEstimator::MapRefArr coords;    // 之字形搜索与图像坐标的映射，refer图像，map pixel index to zigzag matrix coordinates
	DepthEstimator::MapRefArr coordsTrg; // 之字形搜索与图像坐标的映射，target image
};
/*----------------------------------------------------------------*/

struct MVS_API DenseDepthMapData {
	Scene& scene;			
	IIndexArr images;                 // 图像信息
	IIndexArr neighborsMap;           // 记录每帧对应的参考帧
	DepthMapsData depthMaps;
	volatile Thread::safe_t idxImage; // 当前计算的帧id
	SEACAVE::EventQueue events;       // 内部深度计算事件队列 internal events queue (processed by the working threads)
	Semaphore sem;
	CAutoPtr<Util::Progress> progress;
	int nFusionMode;                  // 深度图计算方式控制<0 用SGM/tSGM >=0 用patchMatch
	STEREO::SemiGlobalMatcher sgm;    // sgm算法
	// 初始化函数
	DenseDepthMapData(Scene& _scene, int _nFusionMode=0);
	// 析构函数
	~DenseDepthMapData();

	void SignalCompleteDepthmapFilter();
};
/*----------------------------------------------------------------*/

} // namespace MVS

#endif
