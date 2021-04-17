/*
* SemiGlobalMatcher.cpp
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
#include "SemiGlobalMatcher.h"
#include "Scene.h"

using namespace MVS;

using namespace STEREO;


// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable OpenCV filter demo
//#define _USE_FILTER_DEMO


// S T R U C T S ///////////////////////////////////////////////////

#ifdef _USE_FILTER_DEMO
#include "opencv2/ximgproc/disparity_filter.hpp"

const cv::String keys =
"{help h usage ? |        | print this message                                                }"
"{GT             |None    | optional ground-truth disparity (MPI-Sintel or Middlebury format) }"
"{dst_path       |None    | optional path to save the resulting filtered disparity map        }"
"{dst_raw_path   |None    | optional path to save raw disparity map before filtering          }"
"{algorithm      |bm      | stereo matching method (bm or sgbm)                               }"
"{filter         |wls_conf| used post-filtering (wls_conf or wls_no_conf)                     }"
"{no-display     |        | don't display results                                             }"
"{no-downscale   |        | force stereo matching on full-sized views to improve quality      }"
"{dst_conf_path  |None    | optional path to save the confidence map used in filtering        }"
"{vis_mult       |1.0     | coefficient used to scale disparity map visualizations            }"
"{max_disparity  |160     | parameter of stereo matching                                      }"
"{window_size    |-1      | parameter of stereo matching                                      }"
"{wls_lambda     |8000.0  | parameter of post-filtering                                       }"
"{wls_sigma      |1.5     | parameter of post-filtering                                       }"
;

cv::Rect computeROI(cv::Size2i src_sz, cv::Ptr<cv::StereoMatcher> matcher_instance)
{
	int min_disparity = matcher_instance->getMinDisparity();
	int num_disparities = matcher_instance->getNumDisparities();
	int block_size = matcher_instance->getBlockSize();

	int bs2 = block_size/2;
	int minD = min_disparity, maxD = min_disparity + num_disparities - 1;

	int xmin = maxD + bs2;
	int xmax = src_sz.width + minD - bs2;
	int ymin = bs2;
	int ymax = src_sz.height - bs2;

	return cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
}

int disparityFiltering(cv::Mat left, cv::Mat right, int argc, const LPCSTR* argv)
{
	cv::CommandLineParser parser(argc,argv,keys);
	parser.about("Disparity Filtering Demo");
	if(parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	cv::String GT_path = parser.get<cv::String>("GT");
	cv::String dst_path = parser.get<cv::String>("dst_path");
	cv::String dst_raw_path = parser.get<cv::String>("dst_raw_path");
	cv::String dst_conf_path = parser.get<cv::String>("dst_conf_path");
	cv::String algo = parser.get<cv::String>("algorithm");
	cv::String filter = parser.get<cv::String>("filter");
	bool no_display = parser.has("no-display");
	bool no_downscale = parser.has("no-downscale");
	int max_disp = parser.get<int>("max_disparity");
	double lambda = parser.get<double>("wls_lambda");
	double sigma  = parser.get<double>("wls_sigma");
	double vis_mult = parser.get<double>("vis_mult");

	int wsize;
	if(parser.get<int>("window_size")>=0) //user provided window_size value
		wsize = parser.get<int>("window_size");
	else
	{
		if(algo=="sgbm")
			wsize = 3; //default window size for SGBM
		else if(!no_downscale && algo=="bm" && filter=="wls_conf")
			wsize = 7; //default window size for BM on downscaled views (downscaling is performed only for wls_conf)
		else
			wsize = 15; //default window size for BM on full-sized views
	}
	if(!parser.check())
	{
		parser.printErrors();
		return -1;
	}

	cv::Mat GT_disp;
	bool noGT(cv::ximgproc::readGT(GT_path,GT_disp)!=0);

	cv::Mat left_for_matcher, right_for_matcher;
	cv::Mat left_disp,right_disp;
	cv::Mat filtered_disp;
	cv::Mat conf_map(left.rows,left.cols,CV_8U);
	conf_map = cv::Scalar(255);
	cv::Rect ROI;
	cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
	double matching_time, filtering_time;
	if(max_disp<=0 || max_disp%16!=0)
	{
		std::cout<<"Incorrect max_disparity value: it should be positive and divisible by 16";
		return -1;
	}
	if(wsize<=0 || wsize%2!=1)
	{
		std::cout<<"Incorrect window_size value: it should be positive and odd";
		return -1;
	}
	if(filter=="wls_conf") // filtering with confidence (significantly better quality than wls_no_conf)
	{
		if(!no_downscale)
		{
			// downscale the views to speed-up the matching stage, as we will need to compute both left
			// and right disparity maps for confidence map computation
			//! [downscale]
			max_disp/=2;
			if(max_disp%16!=0)
				max_disp += 16-(max_disp%16);
			resize(left ,left_for_matcher ,cv::Size(),0.5,0.5, cv::INTER_LINEAR_EXACT);
			resize(right,right_for_matcher,cv::Size(),0.5,0.5, cv::INTER_LINEAR_EXACT);
			//! [downscale]
		}
		else
		{
			left_for_matcher  = left.clone();
			right_for_matcher = right.clone();
		}

		if(algo=="bm")
		{
			//! [matching]
			cv::Ptr<cv::StereoBM> left_matcher = cv::StereoBM::create(max_disp,wsize);
			wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
			cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

			cvtColor(left_for_matcher,  left_for_matcher,  cv::COLOR_BGR2GRAY);
			cvtColor(right_for_matcher, right_for_matcher, cv::COLOR_BGR2GRAY);

			matching_time = (double)cv::getTickCount();
			left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
			right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
			matching_time = ((double)cv::getTickCount() - matching_time)/cv::getTickFrequency();
			//! [matching]
		}
		else if(algo=="sgbm")
		{
			cv::Ptr<cv::StereoSGBM> left_matcher  = cv::StereoSGBM::create(-max_disp/2+1,max_disp,wsize);
			left_matcher->setP1(24*wsize*wsize);
			left_matcher->setP2(96*wsize*wsize);
			left_matcher->setPreFilterCap(63);
			left_matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
			wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
			cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

			matching_time = (double)cv::getTickCount();
			left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
			right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
			matching_time = ((double)cv::getTickCount() - matching_time)/cv::getTickFrequency();
		}
		else
		{
			std::cout<<"Unsupported algorithm";
			return -1;
		}

		//! [filtering]
		wls_filter->setLambda(lambda);
		wls_filter->setSigmaColor(sigma);
		filtering_time = (double)cv::getTickCount();
		wls_filter->filter(left_disp,left,filtered_disp,right_disp);
		filtering_time = ((double)cv::getTickCount() - filtering_time)/cv::getTickFrequency();
		//! [filtering]
		conf_map = wls_filter->getConfidenceMap();

		// Get the ROI that was used in the last filter call:
		ROI = wls_filter->getROI();
		if(!no_downscale)
		{
			// upscale raw disparity and ROI back for a proper comparison:
			resize(left_disp,left_disp,cv::Size(),2.0,2.0,cv::INTER_LINEAR_EXACT);
			left_disp = left_disp*2.0;
			ROI = cv::Rect(ROI.x*2,ROI.y*2,ROI.width*2,ROI.height*2);
		}
	}
	else if(filter=="wls_no_conf")
	{
		/* There is no convenience function for the case of filtering with no confidence, so we
		will need to set the ROI and matcher parameters manually */

		left_for_matcher  = left.clone();
		right_for_matcher = right.clone();

		if(algo=="bm")
		{
			cv::Ptr<cv::StereoBM> matcher  = cv::StereoBM::create(max_disp,wsize);
			matcher->setTextureThreshold(0);
			matcher->setUniquenessRatio(0);
			cvtColor(left_for_matcher,  left_for_matcher, cv::COLOR_BGR2GRAY);
			cvtColor(right_for_matcher, right_for_matcher, cv::COLOR_BGR2GRAY);
			ROI = computeROI(left_for_matcher.size(),matcher);
			wls_filter = cv::ximgproc::createDisparityWLSFilterGeneric(false);
			wls_filter->setDepthDiscontinuityRadius((int)ceil(0.33*wsize));

			matching_time = (double)cv::getTickCount();
			matcher->compute(left_for_matcher,right_for_matcher,left_disp);
			matching_time = ((double)cv::getTickCount() - matching_time)/cv::getTickFrequency();
		}
		else if(algo=="sgbm")
		{
			cv::Ptr<cv::StereoSGBM> matcher  = cv::StereoSGBM::create(0,max_disp,wsize);
			matcher->setUniquenessRatio(0);
			matcher->setDisp12MaxDiff(1000000);
			matcher->setSpeckleWindowSize(0);
			matcher->setP1(24*wsize*wsize);
			matcher->setP2(96*wsize*wsize);
			matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
			ROI = computeROI(left_for_matcher.size(),matcher);
			wls_filter = cv::ximgproc::createDisparityWLSFilterGeneric(false);
			wls_filter->setDepthDiscontinuityRadius((int)ceil(0.5*wsize));

			matching_time = (double)cv::getTickCount();
			matcher->compute(left_for_matcher,right_for_matcher,left_disp);
			matching_time = ((double)cv::getTickCount() - matching_time)/cv::getTickFrequency();
		}
		else
		{
			std::cout<<"Unsupported algorithm";
			return -1;
		}

		wls_filter->setLambda(lambda);
		wls_filter->setSigmaColor(sigma);
		filtering_time = (double)cv::getTickCount();
		wls_filter->filter(left_disp,left,filtered_disp,cv::Mat(),ROI);
		filtering_time = ((double)cv::getTickCount() - filtering_time)/cv::getTickFrequency();
	}
	else
	{
		std::cout<<"Unsupported filter";
		return -1;
	}

	//collect and print all the stats:
	std::cout.precision(2);
	std::cout<<"Matching time:  "<<matching_time<<"s"<<std::endl;
	std::cout<<"Filtering time: "<<filtering_time<<"s"<<std::endl;
	std::cout<<std::endl;

	double MSE_before,percent_bad_before,MSE_after,percent_bad_after;
	if(!noGT)
	{
		MSE_before = cv::ximgproc::computeMSE(GT_disp,left_disp,ROI);
		percent_bad_before = cv::ximgproc::computeBadPixelPercent(GT_disp,left_disp,ROI);
		MSE_after = cv::ximgproc::computeMSE(GT_disp,filtered_disp,ROI);
		percent_bad_after = cv::ximgproc::computeBadPixelPercent(GT_disp,filtered_disp,ROI);

		std::cout.precision(5);
		std::cout<<"MSE before filtering: "<<MSE_before<<std::endl;
		std::cout<<"MSE after filtering:  "<<MSE_after<<std::endl;
		std::cout<<std::endl;
		std::cout.precision(3);
		std::cout<<"Percent of bad pixels before filtering: "<<percent_bad_before<<std::endl;
		std::cout<<"Percent of bad pixels after filtering:  "<<percent_bad_after<<std::endl;
	}

	if(dst_path!="None")
	{
		cv::Mat filtered_disp_vis;
		cv::ximgproc::getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);
		cv::imwrite(dst_path,filtered_disp_vis);
	}
	if(dst_raw_path!="None")
	{
		cv::Mat raw_disp_vis;
		cv::ximgproc::getDisparityVis(left_disp,raw_disp_vis,vis_mult);
		cv::imwrite(dst_raw_path,raw_disp_vis);
	}
	if(dst_conf_path!="None")
	{
		cv::imwrite(dst_conf_path,conf_map);
	}

	if(!no_display)
	{
		cv::namedWindow("left", cv::WINDOW_AUTOSIZE);
		cv::imshow("left", left);
		cv::namedWindow("right", cv::WINDOW_AUTOSIZE);
		cv::imshow("right", right);

		if(!noGT)
		{
			cv::Mat GT_disp_vis;
			cv::ximgproc::getDisparityVis(GT_disp,GT_disp_vis,vis_mult);
			cv::namedWindow("ground-truth disparity", cv::WINDOW_AUTOSIZE);
			cv::imshow("ground-truth disparity", GT_disp_vis);
		}

		//! [visualization]
		cv::Mat raw_disp_vis;
		cv::ximgproc::getDisparityVis(left_disp,raw_disp_vis,vis_mult);
		cv::namedWindow("raw disparity", cv::WINDOW_AUTOSIZE);
		cv::imshow("raw disparity", raw_disp_vis);
		cv::Mat filtered_disp_vis;
		cv::ximgproc::getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);
		cv::namedWindow("filtered disparity", cv::WINDOW_AUTOSIZE);
		cv::imshow("filtered disparity", filtered_disp_vis);
		cv::waitKey();
		//! [visualization]
	}

	return 0;
}
#endif
/*----------------------------------------------------------------*/


// apply a gradient like filter
void PrefilterXSobel(const cv::Mat& src, cv::Mat& dst, int ftzero=31)
{
	const int OFS = 256 * 4, TABSZ = OFS * 2 + 256;
	uint8_t tab[TABSZ];
	for (int x = 0; x < TABSZ; x++)
		tab[x] = (uint8_t)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero * 2 : x - OFS + ftzero);
	uint8_t val0 = tab[0 + OFS];

	#ifdef _USE_SSE
	volatile bool useSIMD = cv::checkHardwareSupport(CV_CPU_SSE2);
	#endif

	ASSERT(src.type() == CV_8U);
	const cv::Size size(src.size());
	dst.create(size, src.type());
	int y;
	for (y = 0; y < size.height - 1; y += 2) {
		const uint8_t* srow1 = src.ptr<uint8_t>(y);
		const uint8_t* srow0 = y > 0 ? srow1 - src.step : size.height > 1 ? srow1 + src.step : srow1;
		const uint8_t* srow2 = y < size.height - 1 ? srow1 + src.step : size.height > 1 ? srow1 - src.step : srow1;
		const uint8_t* srow3 = y < size.height - 2 ? srow1 + src.step * 2 : srow1;
		uint8_t* dptr0 = dst.ptr<uint8_t>(y);
		uint8_t* dptr1 = dptr0 + dst.step;

		dptr0[0] = dptr0[size.width - 1] = dptr1[0] = dptr1[size.width - 1] = val0;
		int x = 1;

		#ifdef _USE_SSE
		if (useSIMD) {
			__m128i z = _mm_setzero_si128(), ftz = _mm_set1_epi16((short)ftzero);
			__m128i ftz2 = _mm_set1_epi8(cv::saturate_cast<uint8_t>(ftzero * 2));
			for (; x <= size.width - 9; x += 8) {
				__m128i c0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow0 + x - 1)), z);
				__m128i c1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow1 + x - 1)), z);
				__m128i d0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow0 + x + 1)), z);
				__m128i d1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow1 + x + 1)), z);

				d0 = _mm_sub_epi16(d0, c0);
				d1 = _mm_sub_epi16(d1, c1);

				__m128i c2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow2 + x - 1)), z);
				__m128i c3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow3 + x - 1)), z);
				__m128i d2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow2 + x + 1)), z);
				__m128i d3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow3 + x + 1)), z);

				d2 = _mm_sub_epi16(d2, c2);
				d3 = _mm_sub_epi16(d3, c3);

				__m128i v0 = _mm_add_epi16(d0, _mm_add_epi16(d2, _mm_add_epi16(d1, d1)));
				__m128i v1 = _mm_add_epi16(d1, _mm_add_epi16(d3, _mm_add_epi16(d2, d2)));
				v0 = _mm_packus_epi16(_mm_add_epi16(v0, ftz), _mm_add_epi16(v1, ftz));
				v0 = _mm_min_epu8(v0, ftz2);

				_mm_storel_epi64((__m128i*)(dptr0 + x), v0);
				_mm_storel_epi64((__m128i*)(dptr1 + x), _mm_unpackhi_epi64(v0, v0));
			}
		}
		#endif

		for (; x < size.width - 1; x++) {
			int d0 = srow0[x + 1] - srow0[x - 1], d1 = srow1[x + 1] - srow1[x - 1],
				d2 = srow2[x + 1] - srow2[x - 1], d3 = srow3[x + 1] - srow3[x - 1];
			int v0 = tab[d0 + d1 * 2 + d2 + OFS];
			int v1 = tab[d1 + d2 * 2 + d3 + OFS];
			dptr0[x] = (uint8_t)v0;
			dptr1[x] = (uint8_t)v1;
		}
	}

	for (; y < size.height; y++) {
		uint8_t* dptr = dst.ptr<uint8_t>(y);
		for (int x = 0; x < size.width; x++)
			dptr[x] = val0;
	}
}
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

enum EVENT_TYPE {
	EVT_JOB = 0,
	EVT_CLOSE,
};

class EVTClose : public Event
{
public:
	EVTClose() : Event(EVT_CLOSE) {}
};
class EVTPixelProcess : public Event
{
public:
	typedef std::function<void (int,int,int)> FncPixel;
	const cv::Size size;
	volatile Thread::safe_t& idxPixel;
	const FncPixel fncPixel;
	bool Run(void*) override {
		const int numPixels(size.area());
		int idx;
		while ((idx=(int)Thread::safeInc(idxPixel)) < numPixels)
			fncPixel(idx, idx/size.width, idx%size.width);
		return true;
	}
	EVTPixelProcess(cv::Size s, volatile Thread::safe_t& idx, FncPixel f) : Event(EVT_JOB), size(s), idxPixel(idx), fncPixel(f) {}
};
class EVTPixelAccumInc : public Event
{
public:
	typedef std::function<void (int)> FncPixel;
	const int numPixels;
	volatile Thread::safe_t& idxPixel;
	const FncPixel fncPixel;
	bool Run(void*) override {
		int idx;
		while ((idx=(int)Thread::safeInc(idxPixel)) < numPixels)
			fncPixel(idx);
		return true;
	}
	EVTPixelAccumInc(int s, volatile Thread::safe_t& idx, FncPixel f) : Event(EVT_JOB), numPixels(s), idxPixel(idx), fncPixel(f) {}
};
class EVTPixelAccumDec : public Event
{
public:
	typedef std::function<void (int)> FncPixel;
	volatile Thread::safe_t& idxPixel;
	const FncPixel fncPixel;
	bool Run(void*) override {
		int idx;
		while ((idx=(int)Thread::safeDec(idxPixel)) >= 0)
			fncPixel(idx);
		return true;
	}
	EVTPixelAccumDec(volatile Thread::safe_t& idx, FncPixel f) : Event(EVT_JOB), idxPixel(idx), fncPixel(f) {}
};
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

// - P1 and P2s are algorithm constants very similar to those from the original SGM algorithm;
//   they are set to defaults according to the patch size
// - alpha & beta form the final P2 as P2*(1+alpha*e^(-DI^2/(2*beta^2)))
//   where DI is the difference in image intensity I(x)-I(x_prev) in [0,255] range
// - subpixelSteps represents how much sub-pixel accuracy is searched/stored;
//   if 1 no sub-pixel precision, if for example 4 a 0.25 sub-pixel accuracy is stored;
//   the stored value is quantized and represented as integer: val=(float)valStored/subpixelSteps
/**
 * @brief SGM 具体实现的结构体,代价聚合公式参考课件讲解内容
 * 
 * @param[in] _subpixelMode  是否用亚像素
 * @param[in] _subpixelSteps 亚像素的精度（亚像素精度，如果是4则对于精度是0.25，主要是视差值存储的都是整型。所以得到的视差会乘这个值转成整型。具体转回float真值时会除掉这个数）
 * @param[in] _P1 代价聚合时，如果与邻域像素的视差相差一个像素，给的一个小的惩罚值
 * @param[in] P2  代价聚合时，如果相差大于一个像素的给的一个大的惩罚值
 * @param[in] P2alpha  对大的惩罚值p2进行调整时的参数，_P2=（P2*(1+alpha*e^(-DI^2/(2*beta^2)))）时，与图像灰度差相关的参数
 * @param[in] P2beta   同上
 * _P2>_P1(较小的惩罚项可以让算法能够适应视差变化小的情形，如倾斜的平面或者连续的曲面，较大的惩罚项可以让算法正确处理视差非连续情况)
 */
SemiGlobalMatcher::SemiGlobalMatcher(SgmSubpixelMode _subpixelMode, Disparity _subpixelSteps, AccumCost _P1, AccumCost P2, float P2alpha, float P2beta)
	:
	subpixelMode(_subpixelMode),
	subpixelSteps(_subpixelSteps),
	P1(_P1), P2s(GenerateP2s(P2, P2alpha, P2beta))
{
}

SemiGlobalMatcher::~SemiGlobalMatcher()
{
}

CLISTDEF0IDX(SemiGlobalMatcher::AccumCost,int) SemiGlobalMatcher::GenerateP2s(AccumCost P2, float P2alpha, float P2beta)
{
	// p2'=p2*(1+alpha*e^(-i1-i2))^2/2*beta^2))
	CLISTDEF0IDX(AccumCost,int) P2s(256);
	FOREACH(i, P2s)
		P2s[i] = (AccumCost)ROUND2INT(P2*(1.f+P2alpha*EXP(-SQUARE((float)i)/(2.f*SQUARE(P2beta)))));
	return P2s;
}


// Compute SGM stereo for this image and each of the neighbor views:
//  - minResolution is the resolution of the top of the pyramid for tSGM;
//    can be 0 to force the standard SGM algorithm
// SGM/tSGM算法计算视差图
//
// An implementation of the popular Semi-Global Matching (SGM) algorithm.
//参考文献：SGM：Stereo Processing by Semiglobal Matching and Mutual Information
//         tSGM:SURE: Photogrammetric surface reconstruction from imagery
// SGM 算法主要实现两种经典SGM和tSGM,主要区别是代价聚合的视差搜索范围不同，故聚合代价有区别。经典SGM使用的全局视差范围，每个
// 像素的视差范围都是相同。tSGM 视差搜素范围每个像素的范围都不一样。这样可以大大减少代价聚合的计算量。
// Step 3_2_2 SGM/tSGM:算法
void SemiGlobalMatcher::Match(const Scene& scene, IIndex idxImage, IIndex numNeighbors, unsigned minResolution)
{
	const Image& leftImage = scene.images[idxImage];
	const float fMinScore(MAXF(leftImage.neighbors.front().score*(OPTDENSE::fViewMinScoreRatio*0.1f), OPTDENSE::fViewMinScore));
	FOREACH(idxNeighbor, leftImage.neighbors) {
		const ViewScore& neighbor = leftImage.neighbors[idxNeighbor];
		// exclude neighbors that over the limit or too small score
		// 分数较小的邻域帧不参与计算
		ASSERT(scene.images[neighbor.idx.ID].IsValid());
		if ((numNeighbors && idxNeighbor >= numNeighbors) ||
			(neighbor.score < fMinScore))
			break;
		// check if the disparity-map was already estimated for the same image pairs
		// 判断是否已经进行视差图计算，如果已经计算则不再计算
		const Image& rightImage = scene.images[neighbor.idx.ID];
		const String pairName(MAKE_PATH(String::FormatString("%04u_%04u", leftImage.ID, rightImage.ID)));
		if (File::isPresent((pairName+".dimap").c_str()) || File::isPresent(MAKE_PATH(String::FormatString("%04u_%04u.dimap", rightImage.ID, leftImage.ID))))
			continue;
		TD_TIMER_STARTD();
		IndexArr points;
		Matrix3x3 H; Matrix4x4 Q;
		ViewData leftData, rightData;
		MaskMap leftMaskMap, rightMaskMap; {
		// fetch pairs of corresponding image points
		//TODO: use precomputed points from SelectViews()
		//!!! 查找左右图能看到的3D点，改进点是可以提前计算好避免重复计算浪费时间
		Point3fArr leftPoints, rightPoints;
		FOREACH(idxPoint, scene.pointcloud.points) {
			const PointCloud::ViewArr& views = scene.pointcloud.pointViews[idxPoint];
			if (views.FindFirst(idxImage) != PointCloud::ViewArr::NO_INDEX) {
				points.push_back((uint32_t)idxPoint);
				if (views.FindFirst(neighbor.idx.ID) != PointCloud::ViewArr::NO_INDEX) {
					const Point3 X(scene.pointcloud.points[idxPoint]);
					leftPoints.emplace_back(leftImage.camera.TransformPointW2I3(X));
					rightPoints.emplace_back(rightImage.camera.TransformPointW2I3(X));
				}
			}
		}
		// stereo-rectify image pair
		// Step 3_2_2_1 极线校正,H 像素坐标从原图转换到校正后的图上，Q 把校正图像上的[x' y' disparity 1] 转换到原图坐标系下 [x*z y*z z 1]*w 
		if (!Image::StereoRectifyImages(leftImage, rightImage, leftPoints, rightPoints, leftData.imageColor, rightData.imageColor, leftMaskMap, rightMaskMap, H, Q))
			continue;
		ASSERT(leftData.imageColor.size() == rightData.imageColor.size());
		}
		#ifdef _USE_FILTER_DEMO
		// run openCV implementation
		const LPCSTR argv[] = {"disparityFiltering", "-algorithm=sgbm", "-max_disparity=160", "-no-downscale"};
		disparityFiltering(leftData.imageColor, rightData.imageColor, (int)SizeOfArray(argv), argv);
		#endif
		// color to gray conversion
		// 彩色图转灰度图
		#if SGM_SIMILARITY == SGM_SIMILARITY_CENSUS
		leftData.imageColor.toGray(leftData.imageGray, cv::COLOR_BGR2GRAY, false, true);
		rightData.imageColor.toGray(rightData.imageGray, cv::COLOR_BGR2GRAY, false, true);
		#else
		leftData.imageColor.toGray(leftData.imageGray, cv::COLOR_BGR2GRAY, true, true);
		rightData.imageColor.toGray(rightData.imageGray, cv::COLOR_BGR2GRAY, true, true);
		#endif
		// compute scale used for the disparity estimation
		REAL scale(1);
		// 计算用来depth计算的scale 
		if (minResolution) {
			unsigned resolutionLevel(8);
			Image8U::computeMaxResolution(leftData.imageGray.width(), leftData.imageGray.height(), resolutionLevel, minResolution);
			scale = REAL(1)/MAXF(2,POWI(2,resolutionLevel));
		}
		// 如果scale不等于1就使用tSGM算法（金字塔）上述论文有介绍
		const bool tSGM(!ISEQUAL(scale, REAL(1)));
		DisparityMap leftDisparityMap, rightDisparityMap; AccumCostMap costMap;
		do {
			#if 0
			// export the intermediate disparity-maps
			if (!leftDisparityMap.empty()) {
				const REAL _scale(scale*0.5);
				Matrix3x3 _H(H); Matrix4x4 _Q(Q);
				Image::ScaleStereoRectification(_H, _Q, _scale);
				ExportDisparityDataRawFull(String::FormatString("%s_%d.dimap", pairName.c_str(), log2i(ROUND2INT(REAL(1)/_scale))), leftDisparityMap, costMap, Image8U::computeResize(leftImage.GetSize(), _scale), H, _Q, 1);
			}
			#endif
			// initialize
			// Step 3_2_2_2 视差图初始化
			const ViewData leftDataLevel(leftData.GetImage(scale));
			const ViewData rightDataLevel(rightData.GetImage(scale));
			const cv::Size size(leftDataLevel.imageGray.size());
			const cv::Size sizeValid(size.width-2*halfWindowSizeX, size.height-2*halfWindowSizeY);
			const bool bFirstLevel(leftDisparityMap.empty());//第一次计算标志
			if (bFirstLevel) {
				// initialize the disparity-map with a rough estimate based on the sparse point-cloud
				//TODO: remove DepthData::ViewData dependency
				// 初始用稀疏点云对depth进行初始化，与patchMatch初始化一样。
				Image leftImageLevel(leftImage.GetImage(scene.platforms, scale*0.5, false));
				DepthData::ViewData image;
				image.pImageData = &leftImageLevel; // used only for avgDepth
				image.image.create(leftImageLevel.GetSize());
				image.camera = leftImageLevel.camera;
				DepthMap depthMap;
				Depth dMin, dMax;
				TriangulatePoints2DepthMap(image, scene.pointcloud, points, depthMap, dMin, dMax);
				points.Release();
				Matrix3x3 H2(H); Matrix4x4 Q2(Q);
				Image::ScaleStereoRectification(H2, Q2, scale*0.5);
				const cv::Size sizeHalf(Image8U::computeResize(size, 0.5));
				// 有效视差去除边界不大于窗口的像素
				const cv::Size sizeValidHalf(sizeHalf.width-2*halfWindowSizeX, sizeHalf.height-2*halfWindowSizeY);
				leftDisparityMap.create(sizeValidHalf);
				//depth转视差图，H，Q介绍看上个函数极限矫正StereoRectifyImages
				Depth2DisparityMap(depthMap, H2.inv(), Q2.inv(), 1, leftDisparityMap);
				// resize masks
				cv::resize(leftMaskMap, leftMaskMap, size, 0, 0, cv::INTER_NEAREST);
				cv::resize(rightMaskMap, rightMaskMap, size, 0, 0, cv::INTER_NEAREST);
				const cv::Rect ROI(halfWindowSizeX,halfWindowSizeY, sizeValid.width,sizeValid.height);
				leftMaskMap(ROI).copyTo(leftMaskMap);
				rightMaskMap(ROI).copyTo(rightMaskMap);
			} else {
				// upscale masks
				UpscaleMask(leftMaskMap, sizeValid);
				UpscaleMask(rightMaskMap, sizeValid);
			}
			// Step 3_2_2_3 计算视差图estimate right-left disparity-map
			Index numCosts;
			if (tSGM) {
				// upscale the disparity-map from the previous level
				// 左右视差转换为右左视差
				FlipDirection(leftDisparityMap, rightDisparityMap);
				// 计算每个像素点的视差搜索范围（对应论文中SURE: Photogrammetric Surface Reconstruction from Imagery 2.2.2的介绍）
				numCosts = Disparity2RangeMap(rightDisparityMap, rightMaskMap, bFirstLevel?11:5, bFirstLevel?33:7);
			} else {
				// extract global min and max disparities
				// 标准的SGM，每个像素视差范围都一致，故计算一个全局的最小最大视差即可。
				Range range{std::numeric_limits<Disparity>::max(), std::numeric_limits<Disparity>::min()};
				ASSERT(leftDisparityMap.isContinuous());
				const Disparity* pd = leftDisparityMap.ptr<const Disparity>();
				const Disparity* const pde = pd+leftDisparityMap.area();
				do {
					const Disparity d(*pd);
					if (range.minDisp > d)
						range.minDisp = d;
					if (range.maxDisp < d)
						range.maxDisp = d;
				} while (++pd < pde);
				// set disparity search range to the global min/max range
				//? numDisp+16原因？
				const Disparity numDisp(range.numDisp()+16);
				const Disparity disp(range.minDisp+range.maxDisp);
				range.minDisp = disp-numDisp;//2*minDisp-16
				range.maxDisp = disp+numDisp;//2*maxDisp+16
				maxNumDisp = range.numDisp();
				numCosts = 0;//(size.width-2*halfWindowSizeX, size.height-2*halfWindowSizeY)*(range.maxDisp-range.minDisp)
				imagePixels.resize(sizeValid.area());
				for (PixelData& pixel: imagePixels) {
					pixel.range = range;
					pixel.idx = numCosts;
					numCosts += maxNumDisp;
				}
			}
			imageCosts.resize(numCosts);
			imageAccumCosts.resize(numCosts);
			//构建right-left 视差图
			Match(rightDataLevel, leftDataLevel, rightDisparityMap, costMap);
			// estimate left-right disparity-map
			if (tSGM) {
				numCosts = Disparity2RangeMap(leftDisparityMap, leftMaskMap, bFirstLevel?11:5, bFirstLevel?33:7);
				imageCosts.resize(numCosts);
				imageAccumCosts.resize(numCosts);
			} else {
				//图像左右，换了视差*-1
				for (PixelData& pixel: imagePixels) {
					const Disparity maxDisp(-pixel.range.minDisp);
					pixel.range.minDisp = -pixel.range.maxDisp;
					pixel.range.maxDisp = maxDisp;
				}
			}
			//构建left-right视差图
			Match(leftDataLevel, rightDataLevel, leftDisparityMap, costMap);
			// Step 3_2_2_4  视差一致性检查check disparity-map cross-consistency
			#if 0
			if (ISEQUAL(scale, REAL(1))) {
				cv::Ptr<cv::ximgproc::DisparityWLSFilter> filter = cv::ximgproc::createDisparityWLSFilterGeneric(true);
				const cv::Rect rcValid(halfWindowSizeX,halfWindowSizeY, sizeValid.width,sizeValid.height);
				Image32F leftDisparityMap32F, rightDisparityMap32F, filtered32F;
				leftDisparityMap.convertTo(leftDisparityMap32F, CV_32F, 16);
				rightDisparityMap.convertTo(rightDisparityMap32F, CV_32F, 16);
				filter->filter(leftDisparityMap32F, leftData.imageColor(rcValid), filtered32F, rightDisparityMap32F);
				filtered32F.convertTo(leftDisparityMap, CV_16S, 1.0/16, 0.5);
			} else
			#endif
			if (bFirstLevel) {
				// perform a rigorous filtering of the estimated disparity maps in order to
				// estimate the common region or interest and set the validity masks
				// 错误匹配剔除方法是左右一致性法,将左右影像互换位置，即左影像成为右影像，右影像成为左影像，再做一次立体匹配，得到另一张视差图，因为视差图中每个值所反映的是
				// 两个像素之间的对应关系，所以依据视差的唯一性约束，通过左影像的视差图，找到每个像素在右影像的同名点像素及该像素对应的视差值，这两个视差值之间的差值若小于一定
				// 阈值（一般为1个像素），则满足唯一性约束被保留，反之则不满足唯一性约束而被剔除
				ConsistencyCrossCheck(leftDisparityMap, rightDisparityMap);
				ConsistencyCrossCheck(rightDisparityMap, leftDisparityMap);
				// 视差后处理剔除speckles
				cv::filterSpeckles(leftDisparityMap, NO_DISP, OPTDENSE::nSpeckleSize, 5);
				cv::filterSpeckles(rightDisparityMap, NO_DISP, OPTDENSE::nSpeckleSize, 5);
				ExtractMask(leftDisparityMap, leftMaskMap);
				ExtractMask(rightDisparityMap, rightMaskMap);
			} else {
				// simply run a left-right consistency check
				// 左右一致性检查
				ConsistencyCrossCheck(leftDisparityMap, rightDisparityMap);
			}
		} while ((scale*=2) < REAL(1)+ZEROTOLERANCE<REAL>());
		#if 0
		// remove speckles
		if (OPTDENSE::nSpeckleSize > 0)
			cv::filterSpeckles(leftDisparityMap, NO_DISP, OPTDENSE::nSpeckleSize, 5);
		#endif
		// sub-pixel disparity-map estimation
		// Step 3_2_2_5 亚像素提取：采用二次曲线内插的方法获得子像素精度
		RefineDisparityMap(leftDisparityMap);
		#if 1
		// export disparity-map for the left image
		DEBUG_EXTRA("Disparity-map for images %3u and %3u: %dx%d (%s)", leftImage.ID, rightImage.ID,
			leftImage.width, leftImage.height, TD_TIMER_GET_FMT().c_str());
		ExportPointCloud(pairName+".ply", leftImage, leftDisparityMap, Q, subpixelSteps);
		ExportDisparityMap(pairName+".png", leftDisparityMap);
		ExportDisparityDataRawFull(pairName+".dimap", leftDisparityMap, costMap, leftImage.GetSize(), H, Q, subpixelSteps);
		#else
		// convert disparity-map to final depth-map for the left image
		//转换为深度图
		DepthMap depthMap(leftImage.image.size()); ConfidenceMap confMap;
		Disparity2DepthMap(leftDisparityMap, costMap, H, Q, subpixelSteps, depthMap, confMap);
		DEBUG_EXTRA("Depth-map for images %3u and %3u: %dx%d (%s)", leftImage.ID, rightImage.ID,
			depthMap.width(), depthMap.height(), TD_TIMER_GET_FMT().c_str());
		ExportDepthMap(pairName+".png", depthMap);
		MVS::ExportPointCloud(pairName+".ply", leftImage, depthMap, NormalMap());
		ExportDepthDataRaw(pairName+".dmap", leftImage.name, IIndexArr{leftImage.ID,rightImage.ID}, leftImage.GetSize(), leftImage.camera.K, leftImage.camera.R, leftImage.camera.C, 0, FLT_MAX, depthMap, NormalMap(), confMap);
		#endif
	}
}
// 暂时不用参考，该函数是融合视差的。
void SemiGlobalMatcher::Fuse(const Scene& scene, IIndex idxImage, IIndex numNeighbors, unsigned minViews, DepthMap& depthMap, ConfidenceMap& confMap)
{
	TD_TIMER_STARTD();
	const Image& leftImage = scene.images[idxImage];
	const float fMinScore(MAXF(leftImage.neighbors.front().score*(OPTDENSE::fViewMinScoreRatio*0.1f), OPTDENSE::fViewMinScore));
	struct PairData {
		DepthMap depthMap;
		DepthRangeMap depthRangeMap;
		ConfidenceMap confMap;
		PairData(const cv::Size& size) : depthMap(size) {}
	};
	// load and put in same space all disparity-maps containing this view
	CLISTDEFIDX(PairData,IIndex) pairs;
	pairs.reserve(numNeighbors);
	FOREACH(idxNeighbor, leftImage.neighbors) {
		const ViewScore& neighbor = leftImage.neighbors[idxNeighbor];
		// exclude neighbors that over the limit or too small score
		ASSERT(scene.images[neighbor.idx.ID].IsValid());
		if ((numNeighbors && idxNeighbor >= numNeighbors) ||
			(neighbor.score < fMinScore))
			break;
		// check if the disparity-map was estimated for this images pair
		const Image& rightImage = scene.images[neighbor.idx.ID];
		Disparity subpixelSteps;
		cv::Size imageSize; Matrix3x3 H; Matrix4x4 Q;
		DisparityMap disparityMap; AccumCostMap costMap;
		if (!ImportDisparityDataRawFull(MAKE_PATH(String::FormatString("%04u_%04u.dimap", leftImage.ID, rightImage.ID)), disparityMap, costMap, imageSize, H, Q, subpixelSteps)) {
			if (!ImportDisparityDataRawFull(MAKE_PATH(String::FormatString("%04u_%04u.dimap", rightImage.ID, leftImage.ID)), disparityMap, costMap, imageSize, H, Q, subpixelSteps)) {
				DEBUG("warning: no disparity-data file found for image pair (%u,%u)", leftImage.ID, rightImage.ID);
				continue;
			}
			// adapt Q to project the 3D point from right into the left-image
			RMatrix poseR;
			CMatrix poseC;
			ComputeRelativePose(rightImage.camera.R, rightImage.camera.C, leftImage.camera.R, leftImage.camera.C, poseR, poseC);
			Matrix4x4 P(Matrix4x4::IDENTITY);
			AssembleProjectionMatrix(leftImage.camera.K, poseR, poseC, reinterpret_cast<Matrix3x4&>(P));
			Matrix4x4 invK(Matrix4x4::IDENTITY);
			cv::Mat(rightImage.camera.GetInvK()).copyTo(cv::Mat(4,4,cv::DataType<Matrix4x4::Type>::type,invK.val)(cv::Rect(0,0,3,3)));
			Q = P*invK*Q;
		}
		// convert disparity-map to final depth-map for the left image
		PairData& pair = pairs.emplace_back(leftImage.image.size());
		if (!ProjectDisparity2DepthMap(disparityMap, costMap, Q, subpixelSteps, pair.depthMap, pair.depthRangeMap, pair.confMap)) {
			pairs.RemoveLast();
			continue;
		}
		#if TD_VERBOSE != TD_VERBOSE_OFF
		// save pair depth-map
		if (VERBOSITY_LEVEL > 3) {
			const String pairName(MAKE_PATH(String::FormatString("depth%04u_%04u", leftImage.ID, rightImage.ID)));
			ExportDepthMap(pairName+".png", pair.depthMap);
			MVS::ExportPointCloud(pairName+".ply", leftImage, pair.depthMap, NormalMap());
		}
		#endif
	}
	// fuse available depth-maps such that for each pixel set its depth as the average of the largest cluster of agreeing depths;
	// pixel depths values agree if their trust regions overlap
	depthMap.create(leftImage.image.size()); confMap.create(depthMap.size());
	for (int r=0; r<depthMap.rows; ++r) {
		for (int c=0; c<depthMap.cols; ++c) {
			struct Cluster {
				IIndexArr views;
				DepthRange range;
			};
			CLISTDEFIDX(Cluster,IIndex) clusters(0, pairs.size());
			FOREACH(p, pairs) {
				const PairData& pair = pairs[p];
				const Depth depth(pair.depthMap(r,c));
				if (depth <= 0)
					continue;
				const DepthRange& range(pair.depthRangeMap(r,c));
				unsigned numClusters(0);
				for (Cluster& cluster: clusters) {
					if (!ISINSIDE(depth, cluster.range.x, cluster.range.y))
						continue;
					cluster.views.push_back(p);
					if (cluster.range.x < range.x)
						cluster.range.x = range.x;
					if (cluster.range.y > range.y)
						cluster.range.y = range.y;
					++numClusters;
				}
				if (numClusters == 0)
					clusters.emplace_back(Cluster{IIndexArr{p}, range});
			}
			if (clusters.empty()) {
				depthMap(r,c) = Depth(0);
				confMap(r,c) = float(0);
				continue;
			}
			const Cluster& cluster = clusters.GetMax([](const Cluster& i, const Cluster& j) { return i.views.size() < j.views.size(); });
			if (cluster.views.size() < minViews) {
				depthMap(r,c) = Depth(0);
				confMap(r,c) = float(0);
				continue;
			}
			Depth& depth = depthMap(r,c); depth = 0;
			float& conf = confMap(r,c); conf = 0;
			unsigned numDepths(0);
			for (IIndex p: cluster.views) {
				const PairData& pair = pairs[p];
				depth += pair.depthMap(r,c);
				conf += pair.confMap(r,c);
				++numDepths;
			}
			depth /= numDepths;
			conf /= numDepths;
		}
	}
	DEBUG_EXTRA("Depth-map for image %3u fused: %dx%d (%s)", leftImage.ID,
		depthMap.width(), depthMap.height(), TD_TIMER_GET_FMT().c_str());
	#if TD_VERBOSE != TD_VERBOSE_OFF
	// save depth-map
	if (VERBOSITY_LEVEL > 2) {
		const String fileName(MAKE_PATH(String::FormatString("depth%04u", leftImage.ID)));
		ExportDepthMap(fileName+".png", depthMap);
		MVS::ExportPointCloud(fileName+".ply", leftImage, depthMap, NormalMap());
	}
	#endif
}


// Compute SGM stereo on the images
// SGM 立体匹配
void SemiGlobalMatcher::Match(const ViewData& leftImage, const ViewData& rightImage, DisparityMap& disparityMap, AccumCostMap& costMap)
{
	const cv::Size size(leftImage.imageGray.size());
	const cv::Size sizeValid(size.width-2*halfWindowSizeX, size.height-2*halfWindowSizeY);
	ASSERT(leftImage.imageColor.size() == size);

	#if 0
	// display search info (average disparity and range)
	DisplayState(sizeValid);
	#endif

	// compute costs
	//代价计算：一种census计算：Census变换是使用像素邻域内的局部灰度差异将像素灰度转换为二进制
	//一种是WZNCC与patchMatch相似
	{
	ASSERT(!imageCosts.empty());
	const float eps(1e-3f); // 用来抑制无纹理区域噪声影响used suppress the effect of noise in untextured regions
	auto pixel = [&](int idx, int r, int c) {
		// ignore pixel if not valid
		// 如果视差最小值大于最大值则无效
		const PixelData& pixel = imagePixels[idx];
		if (!pixel.range.isValid())
			return;
		#if SGM_SIMILARITY == SGM_SIMILARITY_CENSUS
		//census 汉明距离：两个比特串对应位不相同的个数
		struct Compute {
			static inline int HammingDistance(uint64_t c1, uint64_t c2) {
				return PopCnt(c1 ^ c2);
			}
		};
		// compute pixel cost
		// 代价=左右图的census的汉明距离
		// pixel.idx记录的是当前像素的cost在这整个cost指针变量中的起始地址
		Cost* costs = imageCosts.data()+pixel.idx;
		const Census lc(leftImage.imageCensus(r,c));
		for (int d=pixel.range.minDisp; d<pixel.range.maxDisp; ++d) {
			const ImageRef x(c+d,r);
			// 右图中没有这个点则直接cost给固定值255
			if (!rightImage.imageCensus.isInside(x)) {
				*costs++ = 255;
				continue;
			}
			// 计算汉明距离
			const Census rc(rightImage.imageCensus(x));
			*costs++ = Compute::HammingDistance(lc, rc)*4;
		}
		#else
		//wzncc跟patchMatch类似不再介绍
		struct Compute {
			static float NormL1Sq(const Pixel8U& a, const Pixel8U& b) {
				// |(a-b)| L1 norm of (a-b) when types are byte
				return float(
					SQUARE(unsigned(a[0]<b[0] ? b[0]-a[0] : a[0]-b[0])) +
					SQUARE(unsigned(a[1]<b[1] ? b[1]-a[1] : a[1]-b[1])) +
					SQUARE(unsigned(a[2]<b[2] ? b[2]-a[2] : a[2]-b[2])));
			}
			static float WeightColor(const Image8U3& image, const Pixel8U& center, const ImageRef& x) {
				// color weight [0..1]
				static const float sigmaColor(-1.f/(2.f*SQUARE(0.3f*255)));
				return Compute::NormL1Sq(image(x), center) * sigmaColor;
			}
			static float WeightSpatial(int x, int y) {
				// spatial weight [0..1]
				static const float sigmaSpatial(-1.f/(2.f*SQUARE(0.4f*MAXF<int>(windowSizeX,windowSizeY))));
				return float(SQUARE(x) + SQUARE(y)) * sigmaSpatial;
			}
		};
		const ImageRef u(c+halfWindowSizeX,r+halfWindowSizeY);
		// initialize pixel patch weights
		// 初始化像素块的权重
		WeightedPatch w;
		w.normSq0 = 0;
		w.sumWeights = 0;
		int n = 0;
		const Pixel8U& colCenter = leftImage.imageColor(u);
		for (int i=-halfWindowSizeY; i<=halfWindowSizeY; ++i) {
			for (int j=-halfWindowSizeX; j<=halfWindowSizeX; ++j) {
				const ImageRef x(u.x+j,u.y+i);
				WeightedPatch::Pixel& pw = w.weights[n++];
				w.normSq0 +=
					(pw.tempWeight = leftImage.imageGray(x)) *
					(pw.weight = EXP(Compute::WeightColor(leftImage.imageColor, colCenter, x)+Compute::WeightSpatial(j,i)));
				w.sumWeights += pw.weight;
			}
		}
		ASSERT(n == numTexels);
		const float tm(w.normSq0/w.sumWeights);
		w.normSq0 = 0;
		n = 0;
		do {
			WeightedPatch::Pixel& pw = w.weights[n];
			const float t(pw.tempWeight - tm);
			w.normSq0 += (pw.tempWeight = pw.weight * t) * t;
		} while (++n < numTexels);
		// compute pixel cost
		Cost* costs = imageCosts.data()+pixel.idx;
		for (int d=pixel.range.minDisp; d<pixel.range.maxDisp; ++d) {
			float sum(0), sumSq(0), nom(0);
			for (int i=-halfWindowSizeY, n=0; i<=halfWindowSizeY; ++i) {
				for (int j=-halfWindowSizeX; j<=halfWindowSizeX; ++j) {
					const ImageRef x(u.x+j+d,u.y+i);
					if (!rightImage.imageGray.isInside(x)) {
						*costs++ = 255;
						goto NEXT_COST;
					}
					const float f(rightImage.imageGray(x));
					const WeightedPatch::Pixel& pw = w.weights[n++];
					const float fw(f*pw.weight);
					sum += fw;
					sumSq += f*fw;
					nom += f*pw.tempWeight;
				}
			}
			{
			const float normSq1(sumSq-SQUARE(sum)/w.sumWeights);
			const float ncc(nom/SQRT(w.normSq0*normSq1+eps));
			*costs++ = (ncc <= 0 ? Cost(255) : (Cost)ROUND2INT((1.f-MINF(ncc,1.f))*255.f));
			}
			NEXT_COST:;
		}
		#endif
	};
	ASSERT(threads.IsEmpty());
	// 多线程并行处理像素
	if (!threads.empty()) {
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelProcess(sizeValid, idxPixel, pixel));
		WaitThreadWorkers(threads.size());
	} else
	for (int r=0; r<sizeValid.height; ++r)
		for (int c=0; c<sizeValid.width; ++c)
			pixel(r*sizeValid.width+c, r, c);
	}

	// accumulate costs
	//参考SURE: Photogrammetric Surface Reconstruction from Imagery公式6和8，该聚合
	//兼容了SGM和tSGM算法
	{
	ASSERT(!imageAccumCosts.empty());
	imageAccumCosts.Memset(0);
	#if SGM_SIMILARITY == SGM_SIMILARITY_CENSUS
	const ImageGray::Type Igray(127);
	#else
	const ImageGray::Type Igray(0.5f);
	#endif
	struct LineData {
		AccumCost* L;
		Range R;
		~LineData() { delete[] L; }
		AccumCost operator[] (int i) const { return L[i]; }
		AccumCost& operator[] (int i) { return L[i]; }
	};
	auto pixelAccum = [&](const Cost* costs, const LineData& Lp, LineData& Ls, AccumCost* accums, ImageGray::Type DI) {
		struct Compute {
			static inline void MINS(AccumCost& m, AccumCost v) { if (m > v) m = v; }
		};
		ASSERT(Ls.R.isValid());
		#if SGM_SIMILARITY == SGM_SIMILARITY_CENSUS
		const AccumCost P2(P2s[DI]);
		#else
		const AccumCost P2(P2s[ABS(ROUND2INT(255.f*DI))]);
		#endif
		// 计算Lp与Ls重合的视差范围
		const Disparity minDisp(MAXF(Lp.R.minDisp, Ls.R.minDisp));
		const Disparity maxDisp(MINF(Lp.R.maxDisp, Ls.R.maxDisp));
		
		if (minDisp >= maxDisp) {//无重合
			// the disparity ranges for the two pixels do not intersect;
			// fill all accumulated costs with L(d)=C(d)+P2
			// 参考公式8 如果与邻域没有代价重合则：L(d)=C(d)+P2
			const Disparity numDisp(Ls.R.numDisp());
			for (int idxDisp=0; idxDisp<numDisp; ++idxDisp)
				accums[idxDisp] += (Ls[idxDisp] = costs[idxDisp]+P2);
		} else {
			// accumulate cost as L(d)=C(d)+min(Lp(d)+V(d,dp))-min(Lp)
			// where V(d,dp) is:
			//  0  if d=dp
			//  P1 if |d-dp|=1
			//  P2 if |d-dp|>1
			AccumCost minLp(std::numeric_limits<AccumCost>::max());
			//计算lp与ls重合部分的视差最小值
			//该部分用的公式6
			// 计算聚合路径在重合视差范围内的最小代价
			for (const AccumCost *L=Lp.L+(minDisp-Lp.R.minDisp), *endL=L+(maxDisp-minDisp); L<endL; ++L)
				Compute::MINS(minLp, *L);
			for (Disparity d=Ls.R.minDisp; d<Ls.R.maxDisp; ++d) {
				const int idxDisp(d-Ls.R.minDisp);
				AccumCost& L = Ls[idxDisp];
				L = std::numeric_limits<AccumCost>::max();
				// 遍历了lp与ls重合的所有视差层，寻找最小
				// min(l(lp,d),l(lp,d-1)+P1,l(lp,d+1)+P1,l(lp,i)+P2),i指的是非d,d-1,d+1的其它值。
				for (Disparity dp=minDisp; dp<maxDisp; ++dp) {
					const int idxDispp(dp-Lp.R.minDisp);
					if (dp == d)
						Compute::MINS(L, Lp[idxDispp]);
					else if (dp == d-1 || dp == d+1)
						Compute::MINS(L, Lp[idxDispp]+P1);
					else
						Compute::MINS(L, Lp[idxDispp]+P2);
				}
				// 累加当前视差层在当前聚合路径的聚合代价
				accums[idxDisp] += (L = costs[idxDisp]+L-minLp);
			}
		}
	};
	ASSERT(threads.IsEmpty());
	if (!threads.empty()) {
		struct AccumLines {
			LineData linesBuffer[2];
			LineData* lines[2];
			AccumLines(Disparity maxNumDisp) {
				for (LineData& line: linesBuffer) {
					line.L = new AccumCost[maxNumDisp];
					memset(line.L, 0, sizeof(AccumCost)*maxNumDisp);
					line.R.minDisp = line.R.maxDisp = 0;
				}
				lines[0] = linesBuffer+0;
				lines[1] = linesBuffer+1;
			}
			void NextLine() { std::swap(lines[0], lines[1]); }
			const LineData& operator() (int i) const { return *lines[i]; }
			LineData& operator() (int i) { return *lines[i]; }
		};
		#define ACCUM_PIXELS(cond) \
			AccumLines lines(maxNumDisp); \
			ImageGray::Type Ip(Igray); \
			do { \
				const int idx(u.y*sizeValid.width+u.x); \
				const PixelData& pixel = imagePixels[idx]; \
				if (!pixel.range.isValid()) \
					continue; \
				const Cost* costs = imageCosts.cdata()+pixel.idx; \
				AccumCost* accums = imageAccumCosts.data()+pixel.idx; \
				const LineData& Lp = lines(0); \
				LineData& Ls = lines(1); \
				Ls.R = pixel.range; \
				const ImageGray::Type I(leftImage.imageGray(u)); \
				pixelAccum(costs, Lp, Ls, accums, I-Ip); \
				Ip = I; \
				lines.NextLine(); \
			} while (cond)
		{ // width-down
		auto pixels = [&](int x) {
			ImageRef u(x,0);
			ACCUM_PIXELS(++u.y < sizeValid.height);
		};
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumInc(sizeValid.width, idxPixel, pixels));
		WaitThreadWorkers(threads.size());
		}
		{ // height-right
		auto pixels = [&](int y) {
			ImageRef u(0,y);
			ACCUM_PIXELS(++u.x < sizeValid.width);
		};
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumInc(sizeValid.height, idxPixel, pixels));
		WaitThreadWorkers(threads.size());
		}
		{ // width-up
		auto pixels = [&](int x) {
			ImageRef u(x,sizeValid.height-1);
			ACCUM_PIXELS(--u.y >= 0);
		};
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumInc(sizeValid.width, idxPixel, pixels));
		WaitThreadWorkers(threads.size());
		}
		{ // height-left
		auto pixels = [&](int y) {
			ImageRef u(sizeValid.width-1,y);
			ACCUM_PIXELS(--u.x >= 0);
		};
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumInc(sizeValid.height, idxPixel, pixels));
		WaitThreadWorkers(threads.size());
		}
		if (numDirs == 4) {
		{ // width-right-down
		auto pixels = [&](int x) {
			ImageRef u(x,0);
			ACCUM_PIXELS(++u.x < sizeValid.width && ++u.y < sizeValid.height);
		};
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumInc(sizeValid.width, idxPixel, pixels));
		}
		{ // height-right-down
		auto pixels = [&](int y) {
			ImageRef u(0,y);
			ACCUM_PIXELS(++u.x < sizeValid.width && ++u.y < sizeValid.height);
		};
		volatile Thread::safe_t idxPixel(0);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumInc(sizeValid.height, idxPixel, pixels));
		}
		WaitThreadWorkers(threads.size()*2);
		{ // width-left-down
		auto pixels = [&](int x) {
			ImageRef u(x,0);
			ACCUM_PIXELS(--u.x >= 0  && ++u.y < sizeValid.height);
		};
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumInc(sizeValid.width-1, idxPixel, pixels));
		}
		{ // height-left-down
		auto pixels = [&](int y) {
			ImageRef u(sizeValid.width-1,y);
			ACCUM_PIXELS(--u.x >= 0 && ++u.y < sizeValid.height);
		};
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumInc(sizeValid.height, idxPixel, pixels));
		}
		WaitThreadWorkers(threads.size()*2);
		{ // width-right-up
		auto pixels = [&](int x) {
			ImageRef u(x,sizeValid.height-1);
			ACCUM_PIXELS(++u.x < sizeValid.width && --u.y >= 0);
		};
		volatile Thread::safe_t idxPixel(0);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumInc(sizeValid.width, idxPixel, pixels));
		}
		{ // height-right-up
		auto pixels = [&](int y) {
			ImageRef u(0,y);
			ACCUM_PIXELS(++u.x < sizeValid.width && --u.y >= 0);
		};
		volatile Thread::safe_t idxPixel(sizeValid.height);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumDec(idxPixel, pixels));
		}
		WaitThreadWorkers(threads.size()*2);
		{ // width-left-up
		auto pixels = [&](int x) {
			ImageRef u(x,sizeValid.height-1);
			ACCUM_PIXELS(--u.x >= 0 && --u.y >= 0);
		};
		volatile Thread::safe_t idxPixel(sizeValid.width);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumDec(idxPixel, pixels));
		}
		{ // height-left-up
		auto pixels = [&](int y) {
			ImageRef u(sizeValid.width-1,y);
			ACCUM_PIXELS(--u.x >= 0 && --u.y >= 0);
		};
		volatile Thread::safe_t idxPixel(sizeValid.height-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumDec(idxPixel, pixels));
		}
		WaitThreadWorkers(threads.size()*2);
		}
		#undef ACCUM_PIXELS
	} else {
	//8聚合路径
	const ImageRef dirs[] = {{-1,0}, {0,-1}, {-1,-1}, {1,-1}};
	struct AccumLines {
		const Disparity maxNumDisp; // int16
		LineData* linesBuffer;
		LineData* lines[numDirs][2];// [0]是路径上上个像素的聚合代价 [1]是当前像素的
		AccumLines(Disparity _maxNumDisp) : maxNumDisp(_maxNumDisp), linesBuffer(NULL) {}
		~AccumLines() { delete[] linesBuffer; }
		void Init(int w) {
			const int linewidth(w+2); // 加2避免边界溢出
			const int buffersize(2*linewidth*numDirs);
			if (linesBuffer == NULL) {
				linesBuffer = new LineData[buffersize];
				for (int i=0; i<buffersize; ++i)
					linesBuffer[i].L = new AccumCost[maxNumDisp];
				LineData* line(linesBuffer-linewidth+1);
				for (int idxDir=0; idxDir<numDirs; ++idxDir) {
					lines[idxDir][0] = (line+=linewidth);
					lines[idxDir][1] = (line+=linewidth);
				}
			}
			// 初始设为0
			for (int i=0; i<buffersize; ++i) {
				LineData& line = linesBuffer[i]; 
				memset(line.L, 0, sizeof(AccumCost)*maxNumDisp);
				line.R.minDisp = line.R.maxDisp = 0;
			}
		}
		//  重置上个像素的最小代价值和代价数组
		void NextLine() {
			for (int idxDir=0; idxDir<numDirs; ++idxDir)
				std::swap(lines[idxDir][0], lines[idxDir][1]);
		}
		const LineData& operator() (int idxDir, int r, int c) const { return lines[idxDir][r][c]; }
		LineData& operator() (int idxDir, int r, int c) { return lines[idxDir][r][c]; }
	};
	AccumLines lines(maxNumDisp);
	#define ACCUM_PIXELS(dx, dy, _x) \
		const int idx(r*sizeValid.width+c); \
		const PixelData& pixel = imagePixels[idx]; \
		if (!pixel.range.isValid()) \
			continue; \
		const Cost* costs = imageCosts.cdata()+pixel.idx; \
		AccumCost* accums = imageAccumCosts.data()+pixel.idx; \
		for (int idxDir=0; idxDir<numDirs; ++idxDir) { \
			const ImageRef& dir = dirs[idxDir]; \
			// 路径上的聚合路径
			const LineData& Lp = lines(idxDir,1+dir.y,_x+dir.x); \
			// ls 是当前像素的路径
			LineData& Ls = lines(idxDir,1,_x); \
			Ls.R = pixel.range; \
			const ImageRef xp(c+dx, r+dy); \
			const ImageGray::Type DI(leftImage.imageGray(r,c)-(leftImage.imageGray.isInside(xp)?leftImage.imageGray(xp):Igray)); \
			pixelAccum(costs, Lp, Ls, accums, DI); \
		}
	// 从左上方向开始聚合
	lines.Init(sizeValid.width);
	for (int r=0; r<sizeValid.height; ++r) {
		for (int c=0; c<sizeValid.width; ++c) {
			ACCUM_PIXELS(dir.x, dir.y, c);
		}
		// 循环到下一行时，将当前行计算的路径代价替换掉原来的上一行
		lines.NextLine();
	}
	// 同上，聚合路径是上面的中心对称方向（右下）
	lines.Init(sizeValid.width);
	for (int r=sizeValid.height; --r>=0; ) {
		for (int c=sizeValid.width; --c>=0; ) {
			ACCUM_PIXELS(-dir.x, -dir.y, sizeValid.width-1-c);
		}
		lines.NextLine();
	}
	#undef ACCUM_PIXELS
	}
	}

	// select best disparity and cost
	// 在SGM算法中，视差计算采用赢家通吃（WTA）算法，每个像素选择最小聚合代价值所对应的视差值作为最终视差
	{
	disparityMap.create(sizeValid);
	costMap.create(sizeValid);
	// 计算最佳代价
	auto pixel = [&](int idx) {
		const PixelData& pixel = imagePixels[idx];
		if (pixel.range.isValid()) {
			const AccumCost* accums = imageAccumCosts.cdata()+pixel.idx;
			const AccumCost* bestAccum = accums;
			// 遍历整个有效视差范围，寻找最小值对应的视差
			for (const AccumCost *accum=accums+1, *accumEnd=accums+pixel.range.numDisp(); accum<accumEnd; ++accum) {
				if (*bestAccum > *accum)
					bestAccum = accum;
			}
			disparityMap(idx) = pixel.range.minDisp+(Disparity)(bestAccum-accums);
			costMap(idx) = *bestAccum;
		} else {
			disparityMap(idx) = pixel.range.minDisp;
			costMap(idx) = NO_ACCUMCOST;
		}
	};
	ASSERT(threads.IsEmpty());
	if (!threads.empty()) {
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumInc(sizeValid.area(), idxPixel, pixel));
		WaitThreadWorkers(threads.size());
	} else
	for (int r=0; r<sizeValid.height; ++r)
		for (int c=0; c<sizeValid.width; ++c)
			pixel(r*sizeValid.width+c);
	}
}

#if SGM_SIMILARITY == SGM_SIMILARITY_CENSUS
// Compute the census bit-mask for all the pixels of the image
// 为图像的所有像素计算bit-mask
void SemiGlobalMatcher::CensusTransform(const Image8U& imageGray, CensusMap& imageCensus)
{
	ASSERT(!imageGray.empty());
	const cv::Size size(imageGray.size());
	const cv::Size sizeValid(size.width-2*halfWindowSizeX, size.height-2*halfWindowSizeY);
	// 初始化
	imageCensus.create(sizeValid);

	#if 0
	// pre-process input image to easy CENSUS task
	ImageGray image;
	PrefilterXSobel(imageGray, image);
	#else
	const Image8U& image = imageGray;
	#endif

	auto pixel = [&](int, int r, int c) {
		const ImageRef u(c+halfWindowSizeX, r+halfWindowSizeY);
		const uint8_t g(image(u));
		Census& cs = imageCensus(r,c);
		cs = 0;
		for (int i=-halfWindowSizeY; i<=halfWindowSizeY; ++i) {
			for (int j=-halfWindowSizeX; j<=halfWindowSizeX; ++j) {
				// 左移1位，移动时用0填充右边的空位
				cs <<= 1;
				// 如果不大于邻域当前位就为1
				if (g <= image(u.y+i, u.x+j))
					cs += 1;
			}
		}
	};
	ASSERT(threads.IsEmpty());
	// 多线程
	if (!threads.empty()) {
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelProcess(imageCensus.size(), idxPixel, pixel));
		WaitThreadWorkers(threads.size());
	} else
	for (int r=0; r<imageCensus.rows; ++r)
		for (int c=0; c<imageCensus.cols; ++c)
			pixel(-1, r, c);
}
#endif

// Compute search range from the given disparity-map and setup pixel-map at twice the scale;
// the validity mask-map is considered as well and upscaled in the same time;
// return the total size of the disparities searched
// 计算每个像素点的视差搜索范围（对应论文中SURE: Photogrammetric Surface Reconstruction from Imagery 2.2.2的介绍）
// 因为我们这个视差范围计算是给下一个level使用的，所以我们计算的视差范围imagePixels的大小是传入的视差图的2倍
// 在根据视差图每个像素点计算范围时候，会同时进行插值得到resize后的imagePixels。
SemiGlobalMatcher::Index SemiGlobalMatcher::Disparity2RangeMap(const DisparityMap& disparityMap, const MaskMap& maskMap, Disparity minNumDisp, Disparity minNumDispInvalid)
{
	ASSERT(!disparityMap.empty() && disparityMap.width()<maskMap.width() && disparityMap.height()<maskMap.height());
	// 这个size是disparityMap的大小*2后的size
	const cv::Size size2x(maskMap.size());
	imagePixels.resize(size2x.area());
	Index numCosts(0);
	maxNumDisp = 0;
	CLISTDEF0IDX(Disparity,Disparity) disps(31*31);
	for (int r=0; r<disparityMap.rows; ++r) {
		const int r2(r == 0 ? 0 : r*2+halfWindowSizeY);
		ASSERT(r2 < size2x.height);
		// 用在插值resize后的坐标的
		const int offset(r2*size2x.width);
		int c2e(halfWindowSizeX);
		// 标记的是哪些像素坐标可以计算视差图
		const Mask* pm(maskMap.ptr<const Mask>(r*2+halfWindowSizeY, halfWindowSizeX));
		for (int c=0, c2=0; c<disparityMap.cols; ++c, pm+=2) {
			Disparity numDisp; Range range;
			if (*pm == INVALID) {
				// set empty range
				// 设置空的视差范围
				range = Range{NO_DISP,NO_DISP};
				numDisp = 0;
			} else {
				// set range based on the estimates around this location
				// 根据这个位置周围的估计值设置范围
				const bool bInvalid(disparityMap(r,c) == NO_DISP);
				// search range around 41x41 or 7x7 window
				//如果视差无效，则搜索窗口41*41(扩大搜索)反之7*7
				disps.Empty();
				const int hw(bInvalid ? 20 : 3);
				for (int i=-hw; i<=hw; ++i) {
					for (int j=-hw; j<=hw; ++j) {
						const ImageRef u(c+j,r+i);
						if (disparityMap.isInside(u)) {
							const Disparity d(disparityMap(u));
							if (d != NO_DISP)
								disps.push_back(d);
						}
					}
				}
				// set search range
				// 设置搜索范围
				if (disps.size() < 3) {
					range.maxDisp = MINF((Disparity)(disparityMap.width()*2/3), minNumDispInvalid);
					range.minDisp = -range.maxDisp;
					numDisp = range.numDisp();
				} else {
					//*2是给下一个level准备的视差相关值故需scale 2
					const Disparity disp(disps.GetMedian()*2);
					const auto minmax(disps.GetMinMax());
					numDisp = (minmax.second-minmax.first)*2;
					//为了保证有最小视差搜索个数默认是3
					if (numDisp < minNumDisp) {
						numDisp = minNumDisp;
						range.minDisp = disp-numDisp/2;
						range.maxDisp = disp+(numDisp+1)/2;
					} else {
						//限制视差最大搜索区间是无效的是64，有效32，不能超过这个范围。
						const Disparity maxNumDisp(bInvalid ? 64 : 32);
						if (numDisp > maxNumDisp) {
							range.minDisp = disp-(maxNumDisp*(disp-minmax.first*2)+1)/numDisp;
							range.maxDisp = disp+(maxNumDisp*(minmax.second*2+1-disp)+1)/numDisp;
							numDisp = range.numDisp();
						} else {
							range.minDisp = disp-numDisp/2;
							range.maxDisp = disp+(numDisp+1)/2;
						}
					}
				}
				ASSERT(range.numDisp() == numDisp);
				if (maxNumDisp < numDisp)
					maxNumDisp = numDisp;
			}
			c2e += 2;
			do {
				ASSERT(c2 < size2x.width);
				PixelData& pixel = imagePixels[offset+c2];
				pixel.range = range;
				pixel.idx = numCosts;
				numCosts += numDisp;
			} while (++c2 < c2e);
		}
		ASSERT(c2e < size2x.width);
		//插值resize后的图像上点的视差范围，统计numcosts所搜索的视差的总个数
		do {
			const PixelData& pixel = imagePixels[offset+c2e-1];
			PixelData& _pixel = imagePixels[offset+c2e];
			_pixel.range = pixel.range;
			_pixel.idx = numCosts;
			numCosts += pixel.range.numDisp();
		} while (++c2e < size2x.width);
		const int _offsete((r+1 == disparityMap.rows ? size2x.height : r*2+halfWindowSizeY+2)*size2x.width);
		for (int _offset=offset+size2x.width; _offset<_offsete; _offset+=size2x.width) {
			for (int c2=0; c2<size2x.width; ++c2) {
				const PixelData& pixel = imagePixels[offset+c2];
				PixelData& _pixel = imagePixels[_offset+c2];
				_pixel.range = pixel.range;
				_pixel.idx = numCosts;
				numCosts += pixel.range.numDisp();
			}
		}
	}
	return numCosts;
}

// Check for consistency between a left-to-right and right-to-left pair of stereo results;
// the results are expected to be opposite in sign but equal in magnitude;
// the valid disparities are returned in the left map
// 视差的唯一性约束，一致性检查：主要是判断左右的视差是否相同，剔除错误的视差
// 通过左的视差图，找到每个像素在右影像的同名点像素及该像素对应的视差值，这两个视差值之间的差值若小于一定阈值（一般为1个像素），则满足唯一性约束被保留，反之则不满足唯一性约束而被剔除
void SemiGlobalMatcher::ConsistencyCrossCheck(DisparityMap& l2r, const DisparityMap& r2l, Disparity thCross)
{
	ASSERT(thCross >= 0);
	ASSERT(!l2r.empty() && !r2l.empty());
	ASSERT(l2r.height() == r2l.height());

	auto pixel = [&](int, int r, int c) {
		Disparity& ld = l2r(r,c);
		if (ld == NO_DISP)
			return;
		// compute the corresponding disparity pixel according to the disparity value
		// 根据视差值计算对应的像素坐标
		const ImageRef v(c+ld,r);
		// check image bounds
		// 确认是否超出图像边界
		if (v.x < 0 || v.x >= r2l.width()) {
			ld = NO_DISP;
			return;
		}
		// check right disparity is valid
		// 确认右视差是否有效
		const Disparity rd = r2l(v);
		if (r2l(v) == NO_DISP) {
			ld = NO_DISP;
			return;
		}
		// check disparity consistency:
		//   since the left and right disparities are opposite in sign,
		//   we determine their similarity by *summing* them, rather
		//   than differencing them as you might expect
		// 因为左右视差本身是相反的d和-d ,所以计算两个视差的查是ld+rd，大于阈值thCross丢弃
		if (ABS(ld + rd) > thCross)
			ld = NO_DISP;
	};
	ASSERT(threads.IsEmpty());
	if (!threads.empty()) {
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelProcess(l2r.size(), idxPixel, pixel));
		WaitThreadWorkers(threads.size());
	} else
	for (int r=0; r<l2r.rows; ++r)
		for (int c=0; c<l2r.cols; ++c)
			pixel(-1, r, c);
}

// Discard disparities that have a hight similarity score
// 剔除score值比较高的视差
void SemiGlobalMatcher::FilterByCost(DisparityMap& disparityMap, const AccumCostMap& costMap, AccumCost th)
{
	ASSERT(th > 0);
	ASSERT(!disparityMap.empty() && disparityMap.size() == costMap.size());

	auto pixel = [&](int, int r, int c) {
		Disparity& d = disparityMap(r,c);
		if (d == NO_DISP)
			return;
		if (costMap(r,c) > th)
			d = NO_DISP;
	};
	ASSERT(threads.IsEmpty());
	if (!threads.empty()) {
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelProcess(disparityMap.size(), idxPixel, pixel));
		WaitThreadWorkers(threads.size());
	} else
	for (int r=0; r<disparityMap.rows; ++r)
		for (int c=0; c<disparityMap.cols; ++c)
			pixel(-1, r, c);
}

// Mark empty regions on the border of the disparity-map as invalid;
//  thValid is the number of valid disparities encountered to consider the valid region starts
void SemiGlobalMatcher::ExtractMask(const DisparityMap& disparityMap, MaskMap& maskMap, int thValid)
{
	ASSERT(!disparityMap.empty());
	ASSERT(maskMap.empty() || disparityMap.size() == maskMap.size());
	if (disparityMap.size() != maskMap.size()) {
		maskMap.create(disparityMap.size());
		maskMap.setTo(VALID);
	}

	#define MASK_PIXEL() \
		Mask& m = maskMap(r,c); \
		if (m == INVALID) \
			continue; \
		m = INVALID; \
		if (disparityMap(r,c) == NO_DISP) \
			continue; \
		if (++numValid >= thValid) \
			break

	// left-right direction
	{
	auto pixel = [&](int r) {
		int numValid(0);
		for (int c=0; c<disparityMap.cols; ++c) {
			MASK_PIXEL();
		}
	};
	ASSERT(threads.IsEmpty());
	if (!threads.empty()) {
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumInc(disparityMap.height(), idxPixel, pixel));
		WaitThreadWorkers(threads.size());
	} else
	for (int r=0; r<disparityMap.rows; ++r)
		pixel(r);
	}

	// right-left direction
	{
	auto pixel = [&](int r) {
		int numValid(0);
		for (int c=disparityMap.cols; --c>=0; ) {
			MASK_PIXEL();
		}
	};
	ASSERT(threads.IsEmpty());
	if (!threads.empty()) {
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumInc(disparityMap.height(), idxPixel, pixel));
		WaitThreadWorkers(threads.size());
	} else
	for (int r=0; r<disparityMap.rows; ++r)
		pixel(r);
	}

	#undef MASK_PIXEL

	#if 0
	#define MASK_PIXEL() \
		Mask& m = maskMap(r,c); \
		if (m == INVALID) \
			continue; \
		if (disparityMap(r,c) != NO_DISP) \
			break; \
		m = INVALID

	// top-bottom direction
	{
	auto pixel = [&](int c) {
		for (int r=0; r<disparityMap.rows; ++r) {
			MASK_PIXEL();
		}
	};
	ASSERT(threads.IsEmpty());
	if (!threads.empty()) {
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumInc(disparityMap.width(), idxPixel, pixel));
		WaitThreadWorkers(threads.size());
	} else
	for (int c=0; c<disparityMap.cols; ++c)
		pixel(c);
	}

	// bottom-top direction
	{
	auto pixel = [&](int c) {
		for (int r=disparityMap.rows; --r>=0; ) {
			MASK_PIXEL();
		}
	};
	ASSERT(threads.IsEmpty());
	if (!threads.empty()) {
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumInc(disparityMap.width(), idxPixel, pixel));
		WaitThreadWorkers(threads.size());
	} else
	for (int c=0; c<disparityMap.cols; ++c)
		pixel(c);
	}

	#undef MASK_PIXEL
	#endif
}

// Translate disparity-map between left-to-right and right-to-left stereo pair
// by translating to the x coordinated dictated by the disparity and negating the sign;
// the sub-pixel steps is assumed to be one and
// the input disparity-map has been cross-checked for consistency
// 左右视差转换为右左视差
void SemiGlobalMatcher::FlipDirection(const DisparityMap& l2r, DisparityMap& r2l)
{
	ASSERT(!l2r.empty());
	ASSERT(r2l.empty() || l2r.height() == r2l.height());
	if (r2l.empty())
		r2l.create(l2r.size());
	r2l.setTo(NO_DISP);

	auto pixel = [&](int r) {
		for (int c=0; c<l2r.cols; ++c) {
			const Disparity d = l2r(r,c);
			if (d == NO_DISP)
				continue;
			// compute the corresponding disparity pixel according to the disparity value and set right disparity value
			// 根据视差值计算相应的视差像素，并设置正确的视差值
			for (int x=MAXF(c+d-1,0), xe=MINF(c+d+2,r2l.width()); x<xe; ++x)
				r2l(r,x) = -d;
		}
	};
	ASSERT(threads.IsEmpty());
	if (!threads.empty()) {
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumInc(l2r.rows, idxPixel, pixel));
		WaitThreadWorkers(threads.size());
	} else
	for (int r=0; r<l2r.rows; ++r)
		pixel(r);
}

// Translate disparity-map between left-to-right and right-to-left stereo pair
// by translating to the x coordinated dictated by the disparity and negating the sign;
// the input disparity-map has been cross-checked for consistency
// 上采样 mask*2
void SemiGlobalMatcher::UpscaleMask(MaskMap& maskMap, const cv::Size& size2x)
{
	ASSERT(!maskMap.empty());
	MaskMap maskMap2x(size2x, INVALID);

	auto pixel = [&](int, int r, int c) {
		const int r2(r*2+halfWindowSizeY), c2(c*2+halfWindowSizeX);
		const Mask m(maskMap(r,c));
		for (int i=0; i<2; ++i) {
			for (int j=0; j<2; ++j) {
				const ImageRef u(c2+j,r2+i);
				if (maskMap2x.isInside(u))
					maskMap2x(u) = m;
			}
		}
	};
	ASSERT(threads.IsEmpty());
	// 多线程操作
	if (!threads.empty()) {
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelProcess(maskMap.size(), idxPixel, pixel));
		WaitThreadWorkers(threads.size());
	} else
	for (int r=0; r<maskMap.rows; ++r)
		for (int c=0; c<maskMap.cols; ++c)
			pixel(-1, r, c);

	cv::swap(maskMap, maskMap2x);
}

// Sub-pixel disparity-map refinement based on the accumulated-cost volume
// SGM采用二次曲线内插的方法获得子像素精度，对最优视差的代价值以及前后两个视差的代价值进行二次曲线拟合，曲线的极值点所对应的视差值即为新的子像素视差值
void SemiGlobalMatcher::RefineDisparityMap(DisparityMap& disparityMap) const
{
	ASSERT(!disparityMap.empty());
	if (subpixelSteps <= 1)
		return;
	if (subpixelMode == SUBPIXEL_NA) {
		// simply multiply disparity to the sub-pixel steps
		// 简单乘一个系数到视差上，作为亚像素视差
		auto pixel = [&](int, int r, int c) {
			Disparity& d = disparityMap(r,c);
			if (d == NO_DISP)
				return;
			ASSERT((int)d*subpixelSteps > (int)std::numeric_limits<Disparity>::min());
			ASSERT((int)d*subpixelSteps < (int)std::numeric_limits<Disparity>::max());
			d *= subpixelSteps;
		};
		ASSERT(threads.IsEmpty());
		if (!threads.empty()) {
			volatile Thread::safe_t idxPixel(-1);
			FOREACH(i, threads)
				threads.AddEvent(new EVTPixelProcess(disparityMap.size(), idxPixel, pixel));
			WaitThreadWorkers(threads.size());
		} else
		for (int r=0; r<disparityMap.rows; ++r)
			for (int c=0; c<disparityMap.cols; ++c)
				pixel(-1, r, c);
		return;
	}
	// proposed subpixelMode algorithms
	typedef float real;
	struct Fit {
		static real linear(real x) {
			return x/real(2);
		}
		static real poly4(real x) {
			return (x*x*x*x + x)/real(4);
		}
		static real parabola(real x) {
			return x/(x+real(1));
		}
		static real sine(real x) {
			return real(0.5) * (SIN((x-real(1))*real(HALF_PI)) + real(1));
		}
		static real cosine(real x) {
			return (real(1) - COS(x*(real)(PI/3.0)));
		}
		static real lcBlend(real x) {
			const real factor(real(1.195) - COS(x*(real)(PI/2.3)));
			return cosine(x)*factor + linear(x)*(real(1)-factor);
		}
		// subpixelMode interpolation when only two values are available
		// returns fraction of distance from the primary to the other value
		// 当两者值不同时直接返回当前cost到其他cost的分数，计算分数=0.5*primary/other
		static real semisubpixel(AccumCost primary, AccumCost other) {
			return real(0.5)*(static_cast<real>(primary) / static_cast<real>(other));
		}
		// compute the offset to be added to the integer disparity to get the final result
		// 计算加到整数视差上的偏移量
		static real subpixelMode(AccumCost prev, AccumCost center, AccumCost next, SgmSubpixelMode subpixelMode) {
			ASSERT(prev != NO_ACCUMCOST && center != NO_ACCUMCOST && next != NO_ACCUMCOST);
			// use a lower quality two value interpolation if only two values are available
			// 如果只有两个值可用，则使用低质量的两值插值
			if (prev == center)
				return center == next ? real(0) : semisubpixel(center, next);
			if (center == next)
				return prev == center ? real(0) : -semisubpixel(center, prev);
			// pick which direction to interpolate in
			// 选择插值方向
			const AccumCost ld(prev-center);
			const AccumCost rd(next-center);
			real x, mult;
			if (ld < rd) {
				x = static_cast<real>(ld) / static_cast<real>(rd);
				mult = real(1);
			} else {
				x = static_cast<real>(rd) / static_cast<real>(ld);
				mult = real(-1);
			}
			// use the selected subpixelMode function
			// 选择插值函数
			real value(0);
			switch (subpixelMode) {
			case SUBPIXEL_LINEAR:   value = linear(x); break;
			case SUBPIXEL_POLY4:    value = poly4(x); break;
			case SUBPIXEL_PARABOLA: value = parabola(x); break;
			case SUBPIXEL_SINE:     value = sine(x); break;
			case SUBPIXEL_COSINE:   value = cosine(x); break;
			case SUBPIXEL_LC_BLEND: value = lcBlend(x); break;
			};
			// complete computation
			// 计算插值的偏移量
			return (value - real(0.5))*mult;
		}
	};
	// estimate sub-pixel disparity based on the cost values
	// 基于代价计算亚像素视差
	auto pixel = [&](int idx) {
		const PixelData& pixel = imagePixels[idx];
		if (pixel.range.numDisp() < 2)
			return;
		Disparity& d = disparityMap(idx);
		if (d == NO_DISP)
			return;
		const AccumCost* accums = imageAccumCosts.cdata()+pixel.idx;
		const int idxDisp(d-pixel.range.minDisp);
		real disparity((real)d);
		// 相邻像素拟合一个视差
		if (d == pixel.range.minDisp)
			disparity += Fit::semisubpixel(accums[idxDisp], accums[idxDisp+1]);
		else if (d+1 == pixel.range.maxDisp)
			disparity -= Fit::semisubpixel(accums[idxDisp], accums[idxDisp-1]);
		else
			disparity += Fit::subpixelMode(accums[idxDisp-1], accums[idxDisp], accums[idxDisp+1], subpixelMode);
		ASSERT(ROUND2INT(disparity*subpixelSteps) > (int)std::numeric_limits<Disparity>::min());
		ASSERT(ROUND2INT(disparity*subpixelSteps) < (int)std::numeric_limits<Disparity>::max());
		d = (Disparity)ROUND2INT(disparity*subpixelSteps);
	};
	ASSERT(threads.IsEmpty());
	if (!threads.empty()) {
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelAccumInc(disparityMap.size().area(), idxPixel, pixel));
		WaitThreadWorkers(threads.size());
	} else
	for (int r=0; r<disparityMap.rows; ++r)
		for (int c=0; c<disparityMap.cols; ++c)
			pixel(r*disparityMap.cols+c);
}


// extract disparity and range from the pixel-map
void SemiGlobalMatcher::DisplayState(const cv::Size& size) const
{
	Image8U disparity(size);
	Image8U range(size);
	for (int idx=0; idx<size.area(); ++idx) {
		const PixelData& pixel = imagePixels[idx];
		disparity(idx) = (uint8_t)(CLAMP(pixel.range.avgDisp(), Disparity(-128), Disparity(127))+Disparity(128));
		range(idx) = (uint8_t)CLAMP(pixel.range.numDisp(), Disparity(0), Disparity(255));
	}

	disparity.Show("Disparity", -1, false);
	range.Show("Range", -1, false);

	char c = 'a';
	while (std::tolower(c) != 'q')
		c = cv::waitKey();

	cv::destroyAllWindows();
}

// Compute the disparity-map for the rectified image from the given depth-map of the un-rectified image;
// the disparity map needs to be already constructed at the desired size (valid size, excluding the border)
/**
 * @brief 根据原图的深度图计算校正后的图像的视差图，视差图事先根据有效尺寸分配好。
 * 
 * @param[in] depthMap        原图的深度图
 * @param[in] invH 			  转换矩阵：把校正图的像素坐标转换到原图	
 * @param[in] invQ            转换矩阵把[x*z y*z z 1]*w in original image coordinates（z即为depth） 转到[x' y' disparity 1] in rectified coordinates 
 * @param[in] subpixelSteps   亚像素精度，如果是4则对于精度是0.25，主要是视差值存储的都是整型。所以得到的视差会乘这个值转成整型。具体转回float真值时会除掉这个数
 * @param[in] disparityMap    视差图
 */
void SemiGlobalMatcher::Depth2DisparityMap(const DepthMap& depthMap, const Matrix3x3& invH, const Matrix4x4& invQ, Disparity subpixelSteps, DisparityMap& disparityMap)
{
	auto pixel = [&](int, int r, int c) {
		// rc加half窗口的原因是视差图是有效像素开始的起始是从（halfWindowSizeX，halfWindowSizeY）开始的而非（0，0）
		const ImageRef x(c+halfWindowSizeX,r+halfWindowSizeY); Point2f u;
		//把校正图像上的像素坐标转到原图坐标上
		ProjectVertex_3x3_2_2(invH.val, x.ptr(), u.ptr());
		float depth, disparity;
		//取深度值depth并转到视差上
		if (!depthMap.sampleSafe(depth, u, [](Depth d) { return d > 0; }) || !Image::Depth2Disparity(invQ, u, depth, disparity))
			disparityMap(r,c) = NO_DISP;
		else
			disparityMap(r,c) = (Disparity)ROUND2INT(disparity*subpixelSteps);
	};
	ASSERT(threads.IsEmpty());
	if (!threads.empty()) {
		volatile Thread::safe_t idxPixel(-1);
		FOREACH(i, threads)
			threads.AddEvent(new EVTPixelProcess(disparityMap.size(), idxPixel, pixel));
		WaitThreadWorkers(threads.size());
	} else
	for (int r=0; r<disparityMap.rows; ++r)
		for (int c=0; c<disparityMap.cols; ++c)
			pixel(-1, r, c);
}

// Compute the depth-map for the un-rectified image from the given disparity-map of the rectified image
// 同上反向即可
void SemiGlobalMatcher::Disparity2DepthMap(const DisparityMap& disparityMap, const AccumCostMap& costMap, const Matrix3x3& H, const Matrix4x4& Q, Disparity subpixelSteps, DepthMap& depthMap, ConfidenceMap& confMap)
{
	ASSERT(costMap.empty() || costMap.size() == disparityMap.size());
	ASSERT(!depthMap.empty());
	ASSERT(confMap.empty() || confMap.size() == depthMap.size());

	if (!costMap.empty()) {
		confMap.create(depthMap.size());
		auto pixel = [&](int, int r, int c) {
			const ImageRef x(c,r); Point2f u;
			ProjectVertex_3x3_2_2(H.val, x.ptr(), u.ptr());
			u.x -= halfWindowSizeX; u.y -= halfWindowSizeY;
			float disparity;
			if (!disparityMap.sampleSafe(disparity, u, [](Disparity d) { return d != NO_DISP; })) {
				depthMap(x) = 0;
				confMap(x) = 0;
				return;
			}
			float cost;
			costMap.sampleSafe(cost, u, [](AccumCost c) { return c != NO_ACCUMCOST; });
			depthMap(x) = Image::Disparity2Depth(Q, u, disparity/subpixelSteps);
			confMap(x) = 1.f/(cost+1);
		};
		ASSERT(threads.IsEmpty());
		if (!threads.empty()) {
			volatile Thread::safe_t idxPixel(-1);
			FOREACH(i, threads)
				threads.AddEvent(new EVTPixelProcess(depthMap.size(), idxPixel, pixel));
			WaitThreadWorkers(threads.size());
		} else
		for (int r=0; r<depthMap.rows; ++r)
			for (int c=0; c<depthMap.cols; ++c)
				pixel(-1, r, c);
	} else {
		auto pixel = [&](int, int r, int c) {
			const ImageRef x(c,r); Point2f u;
			ProjectVertex_3x3_2_2(H.val, x.ptr(), u.ptr());
			u.x -= halfWindowSizeX; u.y -= halfWindowSizeY;
			float disparity;
			if (!disparityMap.sampleSafe(disparity, u, [](Disparity d) { return d != NO_DISP; }))
				depthMap(x) = 0;
			else
				depthMap(x) = Image::Disparity2Depth(Q, u, disparity/subpixelSteps);
		};
		ASSERT(threads.IsEmpty());
		if (!threads.empty()) {
			volatile Thread::safe_t idxPixel(-1);
			FOREACH(i, threads)
				threads.AddEvent(new EVTPixelProcess(depthMap.size(), idxPixel, pixel));
			WaitThreadWorkers(threads.size());
		} else
		for (int r=0; r<depthMap.rows; ++r)
			for (int c=0; c<depthMap.cols; ++c)
				pixel(-1, r, c);
	}
}

// Compute the depth-map for the un-rectified image from the given disparity-map of the rectified image
// by projecting the point-cloud to the image and setting the pixel depth as the average of the closest 4 points;
// return false if the disparity map is completely empty
bool SemiGlobalMatcher::ProjectDisparity2DepthMap(const DisparityMap& disparityMap, const AccumCostMap& costMap, const Matrix4x4& Q, Disparity subpixelSteps, DepthMap& depthMap, DepthRangeMap& depthRangeMap, ConfidenceMap& confMap)
{
	ASSERT(costMap.empty() || costMap.size() == disparityMap.size());
	ASSERT(!depthMap.empty());
	ASSERT(confMap.empty() || confMap.size() == depthMap.size());

	// store 4 depths per pixel, the closest one from each side, ordered as:
	//  - left, right, top, bottom (x pointing right and y pointing down)
	// with a small overlap border
	const float overlapBorder(0.5f+0.25f);
	struct DepthData {
		Image32F distMap;
		DepthMap depthMap;
		DepthRangeMap depthRangeMap;
		ConfidenceMap confMap;
		void Init(const cv::Size& size) {
			distMap.create(size);
			depthMap.create(size);
			depthRangeMap.create(size);
			confMap.create(size);
			depthMap.memset(0);
		}
	} depthDatas[4];
	for (int i=0; i<4; ++i)
		depthDatas[i].Init(depthMap.size());
	for (int r=0; r<disparityMap.rows; ++r) {
		for (int c=0; c<disparityMap.cols; ++c) {
			const Disparity disparityInt(disparityMap(r,c));
			if (disparityInt == NO_DISP)
				continue;
			const float disparity((float)disparityInt/subpixelSteps);
			Point2f u, v;
			const ImageRef dx(c+halfWindowSizeX,r+halfWindowSizeY);
			const Depth depth(Image::Disparity2Depth(Q, dx, disparity, u));
			if (depth <= 0)
				continue;
			const Disparity disparityCenter((Disparity)FLOOR2INT(disparity));
			const DepthRange depthRange(
				Image::Disparity2Depth(Q, dx, (float)(disparityCenter-1)),
				Image::Disparity2Depth(Q, dx, (float)(disparityCenter+1))
			);
			ASSERT(ISINSIDE(depth, depthRange.x, depthRange.y));
			const float cost(costMap.empty()?0.f:1.f/(costMap(r,c)+1));
			const ImageRef x(FLOOR2INT(u));
			u.x -= 0.5f; u.y -= 0.5f;
			for (int i=-1; i<=1; ++i) {
				for (int j=-1; j<=1; ++j) {
					const ImageRef nx(x.x+j,x.y+i);
					if (!depthMap.isInside(nx))
						continue;
					const Point2f dist((float)nx.x-u.x,(float)nx.y-u.y);
					const Point2f absDist(ABS(dist));
					if (absDist.x > overlapBorder || absDist.y > overlapBorder)
						continue;
					const int idx((dist.x<0?1:0)+(dist.y<0?2:0));
					DepthData& depthData = depthDatas[idx];
					const float deistSq(normSq(dist));
					float& ndeistSq = depthData.distMap(nx);
					Depth& ndepth = depthData.depthMap(nx);
					if (ndepth > 0 && ndeistSq <= deistSq)
						continue;
					ndeistSq = deistSq;
					ndepth = depth;
					depthData.depthRangeMap(nx) = depthRange;
					depthData.confMap(nx) = cost;
				}
			}
		}
	}
	typedef TAccumulator<Vec4f> ValueAccumulator;
	depthRangeMap.create(depthMap.size());
	confMap.create(depthMap.size());
	const float thDist(SQUARE(0.75f));
	const Depth thDepth(0.02f);
	unsigned numDepths(0);
	for (int r=0; r<depthMap.rows; ++r) {
		for (int c=0; c<depthMap.cols; ++c) {
			float distCenter(std::numeric_limits<float>::max());
			Depth depthCenter;
			for (int i=0; i<4; ++i) {
				const DepthData& depthData = depthDatas[i];
				const Depth depth(depthData.depthMap(r,c));
				if (depth <= 0)
					continue;
				const float dist(depthData.distMap(r,c));
				if (distCenter > dist) {
					distCenter = dist;
					depthCenter = depth;
				}
			}
			if (distCenter > thDist) {
				depthMap(r,c) = Depth(0);
				continue;
			}
			ValueAccumulator acc(Vec4f::ZERO, 0.f);
			for (int i=0; i<4; ++i) {
				const DepthData& depthData = depthDatas[i];
				const Depth depth(depthData.depthMap(r,c));
				if (depth <= 0 || !IsDepthSimilar(depthCenter, depth, thDepth))
					continue;
				const DepthRange& depthRange(depthData.depthRangeMap(r,c));
				acc.Add(Vec4f((float)depth,depthRange.x,depthRange.y,depthData.confMap(r,c)), SQRT(depthData.distMap(r,c)));
			}
			ASSERT(!acc.IsEmpty());
			const Vec4f value(acc.Normalized());
			depthMap(r,c) = value(0);
			depthRangeMap(r,c) = DepthRange(value(1),value(2));
			confMap(r,c) = value(3);
			++numDepths;
		}
	}
	if (costMap.empty())
		confMap.release();
	return numDepths > 0;
}


EventThreadPool SemiGlobalMatcher::threads;
Semaphore SemiGlobalMatcher::sem;

// start worker threads
void SemiGlobalMatcher::CreateThreads(unsigned nMaxThreads)
{
	ASSERT(nMaxThreads > 0);
	ASSERT(threads.IsEmpty() && threads.empty());
	if (nMaxThreads > 1) {
		threads.resize(nMaxThreads);
		threads.start(ThreadWorker);
	}
}
// destroy worker threads
void SemiGlobalMatcher::DestroyThreads()
{
	ASSERT(threads.IsEmpty());
	if (!threads.empty()) {
		FOREACH(i, threads)
			threads.AddEvent(new EVTClose());
		threads.Release();
	}
}

void* SemiGlobalMatcher::ThreadWorker(void*) {
	while (true) {
		CAutoPtr<Event> evt(threads.GetEvent());
		switch (evt->GetID()) {
		case EVT_JOB:
			evt->Run();
			break;
		case EVT_CLOSE:
			return NULL;
		default:
			ASSERT("Should not happen!" == NULL);
		}
		sem.Signal();
	}
	return NULL;
}
void SemiGlobalMatcher::WaitThreadWorkers(unsigned nJobs)
{
	while (nJobs-- > 0)
		sem.Wait();
	ASSERT(threads.IsEmpty());
}
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

bool SemiGlobalMatcher::ExportDisparityDataRaw(const String& fileName, const DisparityMap& disparityMap, const AccumCostMap& costMap, const cv::Size& imageSize, const Matrix3x3& H, const Matrix4x4& Q, Disparity subpixelSteps)
{
	ASSERT(!disparityMap.empty());
	ASSERT(costMap.empty() || disparityMap.size() == costMap.size());

	FILE *f = fopen(fileName, "wb");
	if (f == NULL)
		return false;

	// write info
	fwrite(&imageSize.width, sizeof(int), 2, f);
	fwrite(H.val, sizeof(REAL), 9, f);
	fwrite(Q.val, sizeof(REAL), 16, f);
	fwrite(&subpixelSteps, sizeof(Disparity), 1, f);

	// write resolution
	fwrite(&disparityMap.cols, sizeof(int), 1, f);
	fwrite(&disparityMap.rows, sizeof(int), 1, f);

	// write disparity-map
	fwrite(disparityMap.getData(), sizeof(Disparity), disparityMap.area(), f);

	// write cost-map
	if (!costMap.empty())
		fwrite(costMap.getData(), sizeof(AccumCost), costMap.area(), f);

	fclose(f);
	return true;
} // ExportDisparityDataRaw
// same as above, but exports also the empty border
bool SemiGlobalMatcher::ExportDisparityDataRawFull(const String& fileName, const DisparityMap& disparityMap, const AccumCostMap& costMap, const cv::Size& imageSize, const Matrix3x3& H, const Matrix4x4& Q, Disparity subpixelSteps)
{
	ASSERT(!disparityMap.empty());
	ASSERT(costMap.empty() || disparityMap.size() == costMap.size());

	const cv::Size size(disparityMap.width()+2*halfWindowSizeX,disparityMap.height()+2*halfWindowSizeY);
	const cv::Rect ROI(halfWindowSizeX,halfWindowSizeY, disparityMap.width(),disparityMap.height());
	DisparityMap disparityMapFull(size, NO_DISP);
	disparityMap.copyTo(disparityMapFull(ROI));
	if (costMap.empty())
		return ExportDisparityDataRaw(fileName, disparityMapFull, costMap, imageSize, H, Q, subpixelSteps);
	AccumCostMap costMapFull(size, NO_ACCUMCOST);
	costMap.copyTo(costMapFull(ROI));
	return ExportDisparityDataRaw(fileName, disparityMapFull, costMapFull, imageSize, H, Q, subpixelSteps);
} // ExportDisparityDataRawFull

bool SemiGlobalMatcher::ImportDisparityDataRaw(const String& fileName, DisparityMap& disparityMap, AccumCostMap& costMap, cv::Size& imageSize, Matrix3x3& H, Matrix4x4& Q, Disparity& subpixelSteps)
{
	FILE *f = fopen(fileName, "rb");
	if (f == NULL)
		return false;

	// read info
	fread(&imageSize.width, sizeof(int), 2, f);
	fread(H.val, sizeof(REAL), 9, f);
	fread(Q.val, sizeof(REAL), 16, f);
	fread(&subpixelSteps, sizeof(Disparity), 1, f);
	ASSERT(imageSize.width > 0 && imageSize.height > 0);

	// read resolution
	int w, h;
	fread(&w, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	ASSERT(w > 0 && h > 0);

	// read disparity-map
	disparityMap.create(h,w);
	fread(disparityMap.getData(), sizeof(Disparity), w*h, f);

	// read cost-map
	if (fgetc(f) != EOF) {
		fseek(f, -1, SEEK_CUR);
		costMap.create(h,w);
		fread(costMap.getData(), sizeof(AccumCost), w*h, f);
	}

	fclose(f);
	return true;
} // ImportDisparityDataRaw
// same as above, but imports also the empty border
bool SemiGlobalMatcher::ImportDisparityDataRawFull(const String& fileName, DisparityMap& disparityMap, AccumCostMap& costMap, cv::Size& imageSize, Matrix3x3& H, Matrix4x4& Q, Disparity& subpixelSteps)
{
	if (!ImportDisparityDataRaw(fileName, disparityMap, costMap, imageSize, H, Q, subpixelSteps))
		return false;
	const cv::Size sizeValid(disparityMap.width()-2*halfWindowSizeX,disparityMap.height()-2*halfWindowSizeY);
	const cv::Rect ROI(halfWindowSizeX,halfWindowSizeY, sizeValid.width,sizeValid.height);
	disparityMap = disparityMap(ROI).clone();
	if (!costMap.empty())
		costMap = costMap(ROI).clone();
	return true;
} // ImportDisparityDataRawFull


// export disparity-map as an image (red - maximum disparity, blue - minimum disparity)
Image8U3 SemiGlobalMatcher::DisparityMap2Image(const DisparityMap& disparityMap, Disparity minDisparity, Disparity maxDisparity)
{
	ASSERT(!disparityMap.empty());
	// find min and max values
	if (minDisparity == NO_DISP || maxDisparity == NO_DISP) {
		CLISTDEF0(Disparity) disparities(0, disparityMap.area());
		for (int i=disparityMap.area(); i-- > 0; ) {
			const Disparity disparity = disparityMap[i];
			if (disparity != NO_DISP)
				disparities.emplace_back(disparity);
		}
		if (!disparities.empty()) {
			const std::pair<float,float> th(ComputeX84Threshold<Disparity,float>(disparities.data(), disparities.size(), 5.2f));
			minDisparity = (Disparity)ROUND2INT(th.first-th.second);
			maxDisparity = (Disparity)ROUND2INT(th.first+th.second);
		}
		DEBUG_ULTIMATE("\tdisparity range: [%d, %d]", minDisparity, maxDisparity);
	}
	const float sclDepth(1.f/(float)(maxDisparity - minDisparity));
	// create color image
	Image8U3 img(cv::Size(disparityMap.width()+2*halfWindowSizeX,disparityMap.height()+2*halfWindowSizeY), Pixel8U::BLACK);
	for (int r=0; r<disparityMap.rows; ++r) {
		for (int c=0; c<disparityMap.cols; ++c) {
			const Disparity disparity(disparityMap(r,c));
			if (disparity == NO_DISP)
				continue;
			img(r+halfWindowSizeY,c+halfWindowSizeX) = Pixel8U::gray2color(CLAMP((float)(maxDisparity-disparity)*sclDepth, 0.f, 1.f));
		}
	}
	return img;
} // DisparityMap2Image
bool SemiGlobalMatcher::ExportDisparityMap(const String& fileName, const DisparityMap& disparityMap, Disparity minDisparity, Disparity maxDisparity)
{
	if (disparityMap.empty())
		return false;
	return DisparityMap2Image(disparityMap, minDisparity, maxDisparity).Save(fileName);
} // ExportDisparityMap


// export point cloud
bool SemiGlobalMatcher::ExportPointCloud(const String& fileName, const Image& imageData, const DisparityMap& disparityMap, const Matrix4x4& Q, Disparity subpixelSteps)
{
	ASSERT(!disparityMap.empty());

	// vertex definition
	struct Vertex {
		float x,y,z;
		uint8_t r,g,b;
	};
	// list of property information for a vertex
	static PLY::PlyProperty vert_props[] = {
		{"x", PLY::Float32, PLY::Float32, offsetof(Vertex,x), 0, 0, 0, 0},
		{"y", PLY::Float32, PLY::Float32, offsetof(Vertex,y), 0, 0, 0, 0},
		{"z", PLY::Float32, PLY::Float32, offsetof(Vertex,z), 0, 0, 0, 0},
		{"red", PLY::Uint8, PLY::Uint8, offsetof(Vertex,r), 0, 0, 0, 0},
		{"green", PLY::Uint8, PLY::Uint8, offsetof(Vertex,g), 0, 0, 0, 0},
		{"blue", PLY::Uint8, PLY::Uint8, offsetof(Vertex,b), 0, 0, 0, 0},
	};
	// list of the kinds of elements in the PLY
	static const char* elem_names[] = {
		"vertex"
	};

	// create PLY object
	ASSERT(!fileName.empty());
	Util::ensureFolder(fileName);
	const size_t memBufferSize(disparityMap.area()*(8*3/*pos*/+3*3/*color*/+7/*space*/+2/*eol*/) + 2048/*extra size*/);
	PLY ply;
	if (!ply.write(fileName, 1, elem_names, PLY::BINARY_LE, memBufferSize))
		return false;

	// describe what properties go into the vertex elements
	ply.describe_property("vertex", 6, vert_props);

	// export the array of 3D points
	Vertex vertex;
	for (int r=0; r<disparityMap.rows; ++r) {
		for (int c=0; c<disparityMap.cols; ++c) {
			const Disparity& disparity = disparityMap(r,c);
			if (disparity == NO_DISP)
				continue;
			Point2f u;
			Depth depth(Image::Disparity2Depth(Q, ImageRef(c+halfWindowSizeX,r+halfWindowSizeY), (float)disparity/subpixelSteps, u));
			if (depth <= 0)
				continue;
			if (!imageData.image.isInsideWithBorder<float,1>(u))
				continue;
			const Point3f X(imageData.camera.TransformPointI2W(Point3(u,depth)));
			vertex.x = X.x; vertex.y = X.y; vertex.z = X.z;
			const Pixel8U C(imageData.image.empty() ? Pixel8U::WHITE : imageData.image.sample(u));
			vertex.r = C.r; vertex.g = C.g; vertex.b = C.b;
			ply.put_element(&vertex);
		}
	}
	if (ply.get_current_element_count() == 0)
		return false;

	// write to file
	return ply.header_complete();
} // ExportPointCloud

// imports a DIMAP file and converts it to point-cloud
bool SemiGlobalMatcher::ImportPointCloud(const String& fileName, const ImageArr& images, PointCloud& pointcloud)
{
	// load disparity-map
	Disparity subpixelSteps;
	cv::Size imageSize; Matrix3x3 H; Matrix4x4 Q;
	DisparityMap disparityMap; AccumCostMap costMap;
	if (!ImportDisparityDataRawFull(fileName, disparityMap, costMap, imageSize, H, Q, subpixelSteps))
		return false;
	// parse image index from the file name
	const String name(Util::getFileName(fileName));
	IIndex idxImage(NO_ID), idxImagePair(NO_ID);
	if (sscanf(name, "%u_%u", &idxImage, &idxImagePair) != 2 || idxImage == NO_ID || idxImagePair == NO_ID)
		return false;
	const Image& imageData = images[idxImage];
	ASSERT(imageData.image.size() == imageSize);
	// import the array of 3D points
	for (int r=0; r<disparityMap.rows; ++r) {
		for (int c=0; c<disparityMap.cols; ++c) {
			const Disparity& disparity = disparityMap(r,c);
			if (disparity == NO_DISP)
				continue;
			Point2f u;
			Depth depth(Image::Disparity2Depth(Q, ImageRef(c+halfWindowSizeX,r+halfWindowSizeY), (float)disparity/subpixelSteps, u));
			if (depth <= 0)
				continue;
			if (!imageData.image.isInsideWithBorder<float,1>(u))
				continue;
			pointcloud.points.emplace_back(Cast<PointCloud::Point::Type>(imageData.camera.TransformPointI2W(Point3(u,depth))));
			pointcloud.colors.emplace_back(imageData.image.empty() ? Pixel8U::WHITE : imageData.image.sample(u));
		}
	}
	return true;
} // ImportPointCloud
/*----------------------------------------------------------------*/


bool MVS::STEREO::ExportCamerasEngin(const Scene& scene, const String& fileName)
{
	ASSERT(!scene.IsEmpty());

	File f(fileName, File::WRITE, File::CREATE | File::TRUNCATE);
	if (!f.isOpen())
		return false;

	// write header
	f.print("n_cameras %u\n", scene.images.size());
	f.print("n_points %u\n", 0);

	// write cameras
	for (const Image& image: scene.images) {
		if (!image.IsValid())
			continue;
		const Point3 t(image.camera.GetT());
		f.print("%u %u %u %s "
			"%g %g %g %g "
			"%g %g %g %g %g %g %g %g %g "
			"%g %g %g "
			"%g %g "
			"%u",
			image.ID, image.width, image.height, image.name.c_str(),
			image.camera.K(0,0), image.camera.K(1,1), image.camera.K(0,2), image.camera.K(1,2),
			image.camera.R(0,0), image.camera.R(0,1), image.camera.R(0,2),
			image.camera.R(1,0), image.camera.R(1,1), image.camera.R(1,2),
			image.camera.R(2,0), image.camera.R(2,1), image.camera.R(2,2),
			t.x, t.y, t.z,
			0, 0,
			image.neighbors.size()
		);
		for (const auto& neighbor: image.neighbors)
			f.print(" %u", neighbor.idx.ID);
		f.print("\n");
	}

	return true;
} // ExportCamerasEngin
/*----------------------------------------------------------------*/
