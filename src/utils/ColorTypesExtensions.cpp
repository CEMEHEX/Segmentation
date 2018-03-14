#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include <cstdio>
#include <iostream>
#include <unistd.h>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <thread>

bool operator==(const CvScalar& lhs, const CvScalar& rhs)
{
    return lhs.val[0] == rhs.val[0] &&
            lhs.val[1] == rhs.val[1] &&
            lhs.val[2] == rhs.val[2] &&
            lhs.val[3] == rhs.val[3];
}

CvScalar getColor(cv::Mat& img, int i, int j) {
    auto Vec3bCol = img.at<cv::Vec3b>(i, j);
    auto r = Vec3bCol[2];
    auto g = Vec3bCol[1];
    auto b = Vec3bCol[0];
    return CV_RGB(r, g, b);
}

cv::Vec3b cvScalar2Vec3b(const CvScalar& sc) {
    auto r = sc.val[0];
    auto g = sc.val[1];
    auto b = sc.val[2];
    return cv::Vec3b(r, g, b);
}
