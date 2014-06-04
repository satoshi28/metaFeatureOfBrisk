#ifndef CLUSTER_OF_FEATURE
#define CLUSTER_OF_FEATURE

////////////////////////////////////////////////////////////////////
#include <opencv2/opencv.hpp>

/**
 * Store the image data and computed descriptors of target pattern
 */
struct ClusterOfFeature
{
	int maxClusterSize;
	std::vector<int>	rankingList;
	cv::Mat			metaDescriptors;
	cv::Mat			singleDescriptors;
#if _DEBUG
  std::vector<cv::KeyPoint> metaKeypoints;
  std::vector<cv::KeyPoint> singleKeypoints;
#endif

};
#endif