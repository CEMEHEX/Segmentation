#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include "ImageUtils.h"

cv::Mat runThresholdBasedMethod(const cv::Mat& src, const cv::Mat& curMask) {
    if (curMask.empty()) {
        return cv::Mat();
    }

    std::vector<cv::Mat> channels;
    cv::Mat image_hsv;

    cv::cvtColor(src, image_hsv, CV_BGR2HSV);
    cv::split(image_hsv, channels);

    cv::Mat dst;
    //    cv::threshold(channels[0], dst, 131, 255, cv::THRESH_BINARY);
    //    cv::adaptiveThreshold(channels[0], dst, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 3, 2);
    cv::threshold(channels[0], dst, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    int morph_elem = cv::MORPH_RECT;
    int morph_size = 1;
    auto element = getStructuringElement(morph_elem, cv::Size(2*morph_size + 1, 2*morph_size + 1), cv::Point(morph_size, morph_size));
    morphologyEx(dst, dst, cv::MORPH_OPEN, element);

    cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);

    std::vector<cv::Vec3b> from = {cv::Vec3b(0, 0, 0), cv::Vec3b(255, 255, 255)};
    std::vector<cv::Vec3b> to = {cv::Vec3b(255, 0, 255), cv::Vec3b(255, 255, 0)};
    recolorImg(dst, from, to);

    return dst;
}
