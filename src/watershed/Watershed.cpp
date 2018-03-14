#include "opencv2/imgproc.hpp"

cv::Mat runWatershed(const cv::Mat& img0, const cv::Mat& markerMask) {
    int i, j, compCount = 0;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    findContours(markerMask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    if( contours.empty() )
        return cv::Mat();

    cv::Mat markers(markerMask.size(), CV_32S);
    markers = cv::Scalar::all(0);
    int idx = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0], compCount++ )
        drawContours(markers, contours, idx, cv::Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);

    if( compCount == 0 )
        return cv::Mat();

    std::vector<cv::Vec3b> colorTab;
    for( i = 0; i < compCount; i++ )
    {
        int b = cv::theRNG().uniform(0, 255);
        int g = cv::theRNG().uniform(0, 255);
        int r = cv::theRNG().uniform(0, 255);

        colorTab.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    double t = (double)cv::getTickCount();
    cv::watershed( img0, markers );
    t = (double)cv::getTickCount() - t;
    printf( "execution time = %gms\n", t*1000./cv::getTickFrequency() );

    cv::Mat wshed(markers.size(), CV_8UC3);

    // paint the watershed image
    for( i = 0; i < markers.rows; i++ ) {
        for( j = 0; j < markers.cols; j++ )
        {
            int index = markers.at<int>(i,j);
            if( index == -1 )
                wshed.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,255);
            else if( index <= 0 || index > compCount )
                wshed.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
            else
                wshed.at<cv::Vec3b>(i,j) = colorTab[index - 1];
        }
    }

    return wshed;
}

