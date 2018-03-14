#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include "opencv2/imgproc.hpp"

void recolorImg(cv::Mat& m, const std::vector<cv::Vec3b>& from, const std::vector<cv::Vec3b>& to);

#endif
