#ifndef EXAMPLE_MARKERLESS_AR_PATTERN_HPP
#define EXAMPLE_MARKERLESS_AR_PATTERN_HPP

////////////////////////////////////////////////////////////////////
#include <opencv2/opencv.hpp>

#include "Gps.h"

/**
 * Store the image data and computed descriptors of target pattern
 */
struct Pattern
{
  cv::Mat                   image;

  std::vector<cv::KeyPoint> keypoints;
  cv::Mat                   descriptors;
  Gps						gps;
  int						numberOfDB;
  std::vector<std::pair<bool, int>>			paramOfKeypoints;	//first:true=meta,false=single,second:‚Ç‚Ì‰æ‘œ‚©‚ç—ˆ‚½“Á’¥“_‚©‚ðŽ¦‚·
};
#endif