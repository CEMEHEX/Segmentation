#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

void recolorImg(cv::Mat& m, const std::vector<cv::Vec3b>& from, const std::vector<cv::Vec3b>& to)
{
    assert(from.size() == to.size());
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            auto cur_col = m.at<cv::Vec3b>(i, j);

            for (size_t k = 0; k < from.size(); ++k) {
                if (cur_col == from[k]) {
                    m.at<cv::Vec3b>(i, j) = to[k];
                    break;
                }
            }
        }
    }
}
