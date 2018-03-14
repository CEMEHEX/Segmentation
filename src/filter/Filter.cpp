#include "opencv2/imgproc.hpp"

#include <unordered_set>
#include <unordered_map>
#include <iostream>

#include"ColorTypesExtensions.h"

void processWindow(cv::Mat& img, std::unordered_set<CvScalar>& validColors, int winSize, int x, int y) {
    std::unordered_map<CvScalar, int> colorsCnt;

    int sizeX = std::min(winSize, img.cols - x);
    int sizeY = std::min(winSize, img.rows - y);

    for (size_t i = x; i < x + sizeX; ++i) {
        for (size_t j = y; j < y + sizeY; ++j) {
            if (i >= img.cols || j >= img.rows) {
                std::cerr << "Point (" << i << ", " << j << ") doesn't belong to image" << std::endl << std::endl;
                return;
            }
            auto curPixelCol = getColor(img, j, i);

            if (validColors.find(curPixelCol) != validColors.end()) {
                colorsCnt[curPixelCol]++;
            }
        }
    }

    if (colorsCnt.empty()) {
        std::cerr << "No valid colors in window (" << x << ", " << y <<
                ") - (" << x + sizeX << ", " << y + sizeY << "),\n skipping" << std::endl;
        return;
    }

    auto mostRecentColor = std::max_element(colorsCnt.begin(), colorsCnt.end(),
                                            [](const std::pair<CvScalar, int>& p1, const std::pair<CvScalar, int>& p2) {
        return p1.second < p2.second; })->first;

    for (size_t i = x; i < x + sizeX; ++i) {
        for (size_t j = y; j < y + sizeY; ++j) {
            auto curPixelCol = getColor(img, j, i);
            if (validColors.find(curPixelCol) == validColors.end()) {
                //                std::cerr << curPixelCol.val[2] << std::endl;
                //                std::cerr << curPixelCol.val[1] << std::endl;
                //                std::cerr << curPixelCol.val[0] << std::endl << std::endl;
                img.at<cv::Vec3b>(j, i) = cvScalar2Vec3b(mostRecentColor);
            }
        }
    }
}

void invalidColorFilter(cv::Mat& img, std::unordered_set<CvScalar>& validColors, int winSize = 10) {
    if (img.cols <= winSize || img.rows <= winSize || winSize <= 0) {
        std::cerr << "Bad window size" << std::endl;
        return;
    }

    for (size_t i = 0; i < img.cols ; i += winSize) {
        for (size_t j = 0; j < img.rows; j += winSize) {
            processWindow(img, validColors, winSize, i, j);
        }
    }
}
