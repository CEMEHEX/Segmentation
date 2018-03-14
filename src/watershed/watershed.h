#ifndef WATERSHED_H
#define WATERSHED_H

#include <opencv2/core/utility.hpp>

cv::Mat runWatershed(const cv::Mat& img0, const cv::Mat& markerMask);

#endif
