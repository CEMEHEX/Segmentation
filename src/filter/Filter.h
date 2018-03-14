#ifndef FILTER_H
#define FILTER_H

void invalidColorFilter(cv::Mat& img, std::unordered_set<CvScalar>& validColors, int winSize = 10);

#endif
