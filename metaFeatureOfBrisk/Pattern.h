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
};
#endif