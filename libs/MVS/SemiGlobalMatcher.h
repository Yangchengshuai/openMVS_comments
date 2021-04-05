/*
* SemiGlobalMatcher.h
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

#ifndef _MVS_SEMIGLOBALMATCHER_H_
#define _MVS_SEMIGLOBALMATCHER_H_


// I N C L U D E S /////////////////////////////////////////////////

#include "DepthMap.h"


// D E F I N E S ///////////////////////////////////////////////////

// similarity method
// 匹配代价计算两种方式：基于census，基于带权重ncc，由于带权重ncc在patchMatch中已经讲过，
// 所以sgm 主要介绍基于census的代价匹配
#define SGM_SIMILARITY_WZNCC 1
#define SGM_SIMILARITY_CENSUS 2 
#define SGM_SIMILARITY SGM_SIMILARITY_CENSUS


// S T R U C T S ///////////////////////////////////////////////////

namespace MVS {

class Scene;

namespace STEREO {

// An implementation of the popular Semi-Global Matching (SGM) algorithm.
//参考文献：SGM：Stereo Processing by Semiglobal Matching and Mutual Information
//         tSGM:SURE: Photogrammetric surface reconstruction from imagery
// SGM 算法主要实现两种经典SGM和tSGM,主要区别是代价聚合的视差搜索范围不同，故聚合代价有区别。经典SGM使用的全局视差范围，每个
// 像素的视差范围都是相同。tSGM 视差搜素范围每个像素的范围都不一样。这样可以大大减少代价聚合的计算量。
class MVS_API SemiGlobalMatcher
{
public:
	typedef uint8_t  Mask;      // used to mark the valid region to be estimated 记录视差计算的有效区域
	typedef int16_t  Disparity; // contains the allowable disparity range 视差图包括每个像素的视差range
	typedef uint8_t  Cost;      // used to describe a single disparity cost 视差代价
	typedef uint16_t AccumCost; // used to accumulate CostType values 视差聚合代价
	typedef uint64_t Index; // index pointing to pixel costs/accumulated-costs 记录每个指向代价体的索引

	enum : Mask      { INVALID = 0, VALID = DECLARE_NO_INDEX(Mask) }; // invalid/valid mask value
	enum : Disparity { NO_DISP = DECLARE_NO_INDEX(Disparity) }; // invalid disparity value
	enum : AccumCost { NO_ACCUMCOST = DECLARE_NO_INDEX(AccumCost) }; // invalid accumulated cost value
	enum : Index     { NO_INDEX = DECLARE_NO_INDEX(Index) }; // invalid index value

	struct Range {
		Disparity minDisp; //最小视差
		Disparity maxDisp; //最大视差
		Disparity avgDisp() const { return (minDisp+maxDisp)>>1; }
		Disparity numDisp() const { return maxDisp-minDisp; }
		Disparity isValid() const { return minDisp<maxDisp; }
	};
	struct PixelData {
		Index idx; //   在代价里面的所有index to pixel costs/accumulated-costs data
		Range range; // 视差范围 first and last disparity evaluated for this pixel 记录视差范围
	};

	typedef TImage<Mask> MaskMap; //   累计视差代价的图像对应maskimage of accumulated disparity cost
	typedef TImage<Disparity> DisparityMap; // 视差图disparity image type
	typedef TImage<AccumCost> AccumCostMap; // 累计视差代价的图像image of accumulated disparity cost
	typedef CLISTDEF0IDX(PixelData,Index) PixelMap; // 像素map map of pixel data
	typedef CLISTDEF0IDX(Cost,Index) CostsMap; // 预先计算的视差map map of pre-computed disparity costs
	typedef CLISTDEF0IDX(AccumCost,Index) AccumCostsMap; // 累加视差代价map map of accumulated disparity costs

	enum : int { numDirs = 4 }; // 代价聚合路径数2 or 4 directions accumulated per pixel pass
	#if SGM_SIMILARITY == SGM_SIMILARITY_CENSUS
	enum : int { halfWindowSizeX = 3, halfWindowSizeY = 4 }; // patch kernel size
	#else
	enum : int { halfWindowSizeX = 3, halfWindowSizeY = 3 }; // patch kernel size
	#endif
	enum : int { windowSizeX = halfWindowSizeX*2+1, windowSizeY = halfWindowSizeY*2+1, numTexels = windowSizeX*windowSizeY }; // patch kernel info

	#if SGM_SIMILARITY == SGM_SIMILARITY_CENSUS
	typedef uint64_t Census; // census 值used to store Census transform
	typedef TImage<Census> CensusMap; // census图image of Census transforms
	typedef Image8U ImageGray; // 灰度图 used to store intensities image
	STATIC_ASSERT(sizeof(Census)*8 >= numTexels);
	#else
	typedef WeightedPatchFix<numTexels> WeightedPatch; // pre-computed patch weights
	typedef Image32F ImageGray; // used to store normalized float intensities image
	#endif


	// Available sub-pixel options
	enum SgmSubpixelMode {
		SUBPIXEL_NA = 0, // no sub-pixel processing
		SUBPIXEL_LINEAR,
		SUBPIXEL_POLY4,
		SUBPIXEL_PARABOLA,
		SUBPIXEL_SINE,
		SUBPIXEL_COSINE,
		SUBPIXEL_LC_BLEND // probably the best option
	};

	struct ViewData {
		Image8U3 imageColor; // 彩色图color image
		ImageGray imageGray; // 灰度图intensity image
		#if SGM_SIMILARITY == SGM_SIMILARITY_CENSUS
		CensusMap imageCensus; // census图像image of Census transforms (valid per scale)
		#endif
		ViewData GetImage(REAL scale) const {
			if (ISEQUAL(scale, REAL(1))) {
				#if SGM_SIMILARITY == SGM_SIMILARITY_CENSUS
				// census计算
				CensusTransform(imageGray, const_cast<CensusMap&>(imageCensus));
				#endif
				return *this;
			}
			ASSERT(scale < 1);
			ViewData scaledView;
			cv::resize(imageColor, scaledView.imageColor, cv::Size(), scale, scale, cv::INTER_AREA);
			cv::resize(imageGray, scaledView.imageGray, cv::Size(), scale, scale, cv::INTER_AREA);
			#if SGM_SIMILARITY == SGM_SIMILARITY_CENSUS
			CensusTransform(scaledView.imageGray, scaledView.imageCensus);
			#endif
			return scaledView;
		}
	};

	typedef MVS_API Point2f DepthRange;
	typedef MVS_API TImage<DepthRange> DepthRangeMap;

public:
	SemiGlobalMatcher(SgmSubpixelMode subpixelMode=SUBPIXEL_LC_BLEND, Disparity subpixelSteps=4, AccumCost P1=3, AccumCost P2=4, float P2alpha=14, float P2beta=38);
	~SemiGlobalMatcher();

	void Match(const Scene& scene, IIndex idxImage, IIndex numNeighbors, unsigned minResolution=320);
	void Fuse(const Scene& scene, IIndex idxImage, IIndex numNeighbors, unsigned minViews, DepthMap& depthMap, ConfidenceMap& confMap);

	static void CreateThreads(unsigned nMaxThreads=1);
	static void DestroyThreads();
	static void* ThreadWorker(void*);
	static void WaitThreadWorkers(unsigned nJobs);

	static bool ExportDisparityDataRaw(const String& fileName, const DisparityMap&, const AccumCostMap&, const cv::Size& imageSize, const Matrix3x3& H, const Matrix4x4& Q, Disparity subpixelSteps);
	static bool ExportDisparityDataRawFull(const String& fileName, const DisparityMap&, const AccumCostMap&, const cv::Size& imageSize, const Matrix3x3& H, const Matrix4x4& Q, Disparity subpixelSteps);
	static bool ImportDisparityDataRaw(const String& fileName, DisparityMap&, AccumCostMap&, cv::Size& imageSize, Matrix3x3& H, Matrix4x4& Q, Disparity& subpixelSteps);
	static bool ImportDisparityDataRawFull(const String& fileName, DisparityMap&, AccumCostMap&, cv::Size& imageSize, Matrix3x3& H, Matrix4x4& Q, Disparity& subpixelSteps);
	static Image8U3 DisparityMap2Image(const DisparityMap&, Disparity minDisparity=NO_DISP, Disparity maxDisparity=NO_DISP);
	static bool ExportDisparityMap(const String& fileName, const DisparityMap&, Disparity minDisparity=NO_DISP, Disparity maxDisparity=NO_DISP);
	static bool ExportPointCloud(const String& fileName, const Image&, const DisparityMap&, const Matrix4x4& Q, Disparity subpixelSteps);
	static bool ImportPointCloud(const String& fileName, const ImageArr& images, PointCloud&);

protected:
	void Match(const ViewData& leftImage, const ViewData& rightImage, DisparityMap& disparityMap, AccumCostMap& costMap);
	Index Disparity2RangeMap(const DisparityMap& disparityMap, const MaskMap& maskMap, Disparity minNumDisp=3, Disparity minNumDispInvalid=16);
	#if SGM_SIMILARITY == SGM_SIMILARITY_CENSUS
	static void CensusTransform(const Image8U& imageGray, CensusMap& imageCensus);
	#endif
	static void ConsistencyCrossCheck(DisparityMap& l2r, const DisparityMap& r2l, Disparity thCross=1);
	static void FilterByCost(DisparityMap&, const AccumCostMap&, AccumCost th);
	static void ExtractMask(const DisparityMap&, MaskMap&, int thValid=3);
	static void FlipDirection(const DisparityMap& l2r, DisparityMap& r2l);
	static void UpscaleMask(MaskMap& maskMap, const cv::Size& size2x);
	void RefineDisparityMap(DisparityMap& disparityMap) const;
	void DisplayState(const cv::Size& size) const;

	static CLISTDEF0IDX(AccumCost,int) GenerateP2s(AccumCost P2, float P2alpha, float P2beta);
	static void Depth2DisparityMap(const DepthMap&, const Matrix3x3& invH, const Matrix4x4& invQ, Disparity subpixelSteps, DisparityMap&);
	static void Disparity2DepthMap(const DisparityMap&, const AccumCostMap&, const Matrix3x3& H, const Matrix4x4& Q, Disparity subpixelSteps, DepthMap&, ConfidenceMap&);
	static bool ProjectDisparity2DepthMap(const DisparityMap&, const AccumCostMap&, const Matrix4x4& Q, Disparity subpixelSteps, DepthMap&, DepthRangeMap&, ConfidenceMap&);

protected:
	// parameters
	SgmSubpixelMode subpixelMode;
	Disparity subpixelSteps;
	AccumCost P1;
	CLISTDEF0IDX(AccumCost,int) P2s;

	// main memory buffers that must be allocated
	PixelMap imagePixels;
	CostsMap imageCosts;
	AccumCostsMap imageAccumCosts;
	Disparity maxNumDisp; // maximum number of disparities per-pixel

	// multi-threading
	static EventThreadPool threads; // worker threads
	static Semaphore sem; // signal job end
};
/*----------------------------------------------------------------*/


MVS_API bool ExportCamerasEngin(const Scene&, const String& fileName);
/*----------------------------------------------------------------*/

} // namespace STEREO

} // namespace MVS

#endif // _MVS_SEMIGLOBALMATCHER_H_
