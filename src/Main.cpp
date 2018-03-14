#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <cstdio>
#include <iostream>
#include <unistd.h>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <thread>

#include "watershed/Watershed.h"
#include "utils/ColorTypesExtensions.h"
#include "threshold/HueThreshold.h"
#include "ImageUtils.h"

using namespace cv;
using namespace std;

static void help()
{
    cout << "\nThis program demonstrates the famous watershed segmentation algorithm in OpenCV: watershed()\n"
            "Usage:\n"
            "./watershed [image_name -- default is ../data/fruits.jpg]\n" << endl;


    cout << "Hot keys: \n"
            "\tESC - quit the program\n"
            "\tr - restore the original image\n"
            "\tw or SPACE - run watershed segmentation algorithm\n"
            "\t\t(before running it, *roughly* mark the areas to segment on the image)\n"
            "\t  (before that, roughly outline several markers on the image)\n"
            "\tm - switch on/off color selecting mode\n"
            "\tz - save mask\n"
            "\tl - load mask\n"
            "\tf - apply filter\n"
            "\t1-9 - set brush thickness\n"
            "\th - refresh image" << endl;
}
Mat markerMask, img, img0, curMask;
CvScalar curColor = CV_RGB(0, 0, 0);
Point prevPt(-1, -1);
int curThickness = 5;

// terrain
const CvScalar justTerrainColor = CV_RGB(102, 51, 0);
const CvScalar snowColor = CV_RGB(204, 255, 255);
const CvScalar sandColor = CV_RGB(255, 255, 51);
const CvScalar forestColor = CV_RGB(0, 102, 0);
const CvScalar grassColor = CV_RGB(51, 255, 51);

const CvScalar roadsColor = CV_RGB(160, 160, 160);

const CvScalar buildingsColor = CV_RGB(96, 96, 96);

const CvScalar waterColor = CV_RGB(0, 128, 255);

const CvScalar cloudsColor = CV_RGB(224, 224, 224);

const CvScalar unknownColor = CV_RGB(0, 0, 0);

const CvScalar notSpecifiedColor = CV_RGB(255, 0, 0);

const string IMAGE_WINDOW_NAME("image");
const string WATERSHED_TRANS_WINDOW_NAME("watershed transform");
const string MASK_WINDOW_NAME("mask");

void initColorSet(unordered_set<CvScalar>& colors) {
    colors.insert(justTerrainColor);
    colors.insert(snowColor);
    colors.insert(sandColor);
    colors.insert(forestColor);
    colors.insert(grassColor);
    colors.insert(roadsColor);
    colors.insert(waterColor);
    colors.insert(buildingsColor);
    colors.insert(cloudsColor);
    colors.insert(unknownColor);
}

void mark(Mat src_, CvPoint seed, CvScalar color=CV_RGB(255, 0, 0))
{
    IplImage* src = new IplImage(src_);
    CvConnectedComp comp;

    cvFloodFill( src, seed, color,
                 cvScalarAll(10), // минимальная разность
                 cvScalarAll(10), // максимальная разность
                 &comp,
                 CV_FLOODFILL_FIXED_RANGE + 8,
                 0);
}

inline void refreshMainImg() {
    img0.copyTo(img);

    for (size_t i = 0 ; i < img.cols; ++i) {
        for (size_t j = 0; j < img.rows; ++j) {
            if (markerMask.at<uchar>(j, i) == 255) {
                img.at<Vec3b>(j, i) = cvScalar2Vec3b(CV_RGB(255, 0, 0));
            }
        }
    }

    imshow( IMAGE_WINDOW_NAME, img );
}

inline int blackBrushThickness(int redBrushThickness) {
    return redBrushThickness * redBrushThickness;
}

static void onMouse( int event, int x, int y, int flags, void* )
{
    if( x < 0 || x >= img.cols || y < 0 || y >= img.rows ) {
        return;
    }

    if( event == EVENT_RBUTTONDOWN ) {
        prevPt = Point(x,y);
    }
    else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_RBUTTON) && !(flags & EVENT_FLAG_CTRLKEY) )
    {
        Point pt(x, y);
        if( prevPt.x < 0 )
            prevPt = pt;
        line( markerMask, prevPt, pt, Scalar::all(255), curThickness, 8, 0 );
        line( img, prevPt, pt, Scalar(0, 0, 255), curThickness, 8, 0 );
        prevPt = pt;
        imshow(IMAGE_WINDOW_NAME, img);
    }
    else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_RBUTTON) && (flags & EVENT_FLAG_CTRLKEY) )
    {
        Point pt(x, y);
        if( prevPt.x < 0 )
            prevPt = pt;
        line( markerMask, prevPt, pt, Scalar::all(0), blackBrushThickness(curThickness), 8, 0 );
        line( img, prevPt, pt, Scalar::all(0), blackBrushThickness(curThickness), 8, 0 );
        prevPt = pt;
        imshow(IMAGE_WINDOW_NAME, img);
    } else {
        if (event == EVENT_RBUTTONUP /*&& (flags & EVENT_FLAG_CTRLKEY)*/) { // uncomment for speeding up
            refreshMainImg();
        }
        prevPt = Point(-1,-1);
    }
}

static void onMouse_Mask( int event, int x, int y, int flags, void* ) {
    switch( event ) {
    case CV_EVENT_MOUSEMOVE:
        break;

    case CV_EVENT_RBUTTONDOWN:
        mark(curMask, cvPoint(x, y), curColor);
        imshow(MASK_WINDOW_NAME, curMask);
        break;

    case CV_EVENT_RBUTTONUP:
        break;
    }
}

inline bool file_exists(const string& name) {
    return ( access( name.c_str(), F_OK ) != -1 );
}

inline string removeExtention(const string& filename) {
    const std::string ext(".jpg");
    if ( filename != ext &&
         filename.size() > ext.size() &&
         filename.substr(filename.size() - ext.size()) == ".jpg" )
    {
        return filename.substr(0, filename.size() - ext.size());
    }

    cerr << "Can't remove extention" << endl;
    return "";
}

inline string genMaskFileName(const string& filename) {
    string pureFilename = removeExtention(filename);
    if (pureFilename.empty()) {
        return "";
    }

    return pureFilename + "_mask.png";
}

inline void saveMask(const string& maskFilename) {
    if (!(curMask.rows > 0 && curMask.cols > 0)) {
        cerr << "No mask to save" << endl;
        return;
    }

    if ( !maskFilename.empty() )
    {
        // Uncomment for asking before saving

        //        if (file_exists(maskFilename))  {
        //            cerr << "Warning! File " << maskFilename << " already exists.\n Are you sure you want to overwrite it (y/n)?" << endl;
        //            char c = (char)waitKey(0);
        //            while (c != 'y' && c != 'n') {
        //                cout << "Please enter y or n" << endl;
        //                c = (char)waitKey(0);
        //            }

        //            if (c == 'n') {
        //                cout << "File won't be saved" << endl;
        //                return;
        //            }
        //        }

        cout << "Saving mask to " << maskFilename << endl;
        imwrite(maskFilename, curMask);
        cout << "Saved successfully!" << endl;
    } else {
        cerr << "Something went wrong, can't generate name for mask" << endl;
    }
}

const int IMG_WIDTH = 1200;
const int IMG_HEIGHT = 900;

inline void createMaskWindow() {
    namedWindow( MASK_WINDOW_NAME, cv::WINDOW_NORMAL | CV_GUI_NORMAL);
    resizeWindow(MASK_WINDOW_NAME, IMG_WIDTH, IMG_HEIGHT);
    setMouseCallback( MASK_WINDOW_NAME, onMouse_Mask, 0 );
}

inline void loadMask(const string& maskFileName) {
    if (!file_exists(maskFileName)) {
        cerr << "No mask file to load!" << endl;
        return;
    }

    cout << "Loading mask..." << endl;

    createMaskWindow();
    curMask = imread(maskFileName, 1);
    imshow(MASK_WINDOW_NAME, curMask);

    cout << "Done!" << endl;
}

void processWindow(Mat& img, unordered_set<CvScalar>& validColors, int winSize, int x, int y) {
    unordered_map<CvScalar, int> colorsCnt;

    int sizeX = min(winSize, img.cols - x);
    int sizeY = min(winSize, img.rows - y);

    for (size_t i = x; i < x + sizeX; ++i) {
        for (size_t j = y; j < y + sizeY; ++j) {
            if (i >= img.cols || j >= img.rows) {
                cerr << "Point (" << i << ", " << j << ") doesn't belong to image" << endl << endl;
                return;
            }
            auto curPixelCol = getColor(img, j, i);

            if (validColors.find(curPixelCol) != validColors.end()) {
                colorsCnt[curPixelCol]++;
            }
        }
    }

    if (colorsCnt.empty()) {
        cerr << "No valid colors in window (" << x << ", " << y <<
                ") - (" << x + sizeX << ", " << y + sizeY << "),\n skipping" << endl;
        return;
    }

    auto mostRecentColor = std::max_element(colorsCnt.begin(), colorsCnt.end(),
                                            [](const pair<CvScalar, int>& p1, const pair<CvScalar, int>& p2) {
        return p1.second < p2.second; })->first;

    for (size_t i = x; i < x + sizeX; ++i) {
        for (size_t j = y; j < y + sizeY; ++j) {
            auto curPixelCol = getColor(img, j, i);
            if (validColors.find(curPixelCol) == validColors.end()) {
                //                cerr << curPixelCol.val[2] << endl;
                //                cerr << curPixelCol.val[1] << endl;
                //                cerr << curPixelCol.val[0] << endl << endl;
                img.at<Vec3b>(j, i) = cvScalar2Vec3b(mostRecentColor);
            }
        }
    }
}

void invalidColorFilter(Mat& img, unordered_set<CvScalar>& validColors, int winSize) {
    if (img.cols <= winSize || img.rows <= winSize || winSize <= 0) {
        cerr << "Bad window size" << endl;
        return;
    }

    for (size_t i = 0; i < img.cols ; i += winSize) {
        for (size_t j = 0; j < img.rows; j += winSize) {
            processWindow(img, validColors, winSize, i, j);
        }
    }
}

void filter(unordered_set<CvScalar>& validColors, int ksize = 10) {
    if (!(curMask.rows > 0 && curMask.cols > 0)) {
        cerr << "Mask is not created yet!" << endl;
        return;
    }

    cout << "Applying filter to mask" << endl;
    invalidColorFilter( curMask, validColors, ksize );
    cout << "done!" << endl;
    imshow(MASK_WINDOW_NAME, curMask);
}

inline string genMarkersFileName(const string& filename) {
    string pureFilename = removeExtention(filename);
    if (pureFilename.empty()) {
        return "";
    }

    return pureFilename + "_zMarkers.png";
}

inline void saveMarkers(const string& filename) {
    if (!(markerMask.rows > 0 && markerMask.cols > 0)) {
        cerr << "No markers to save" << endl;
        return;
    }

    if ( !filename.empty() )
    {
        cout << "Saving markers to " << filename << endl;
        imwrite(filename, markerMask);
        cout << "Saved successfully!" << endl;
    } else {
        cerr << "Something went wrong, can't generate name for markers" << endl;
    }
}

inline void loadMarkers(const string& filename) {
    if (!file_exists(filename)) {
        cerr << "No markers file to load!" << endl;
        return;
    }

    cout << "Loading markers..." << endl;

    markerMask = imread(filename, 1);
    cvtColor(markerMask, markerMask, CV_RGB2GRAY);
    refreshMainImg();

    cout << "Done!" << endl;
}

void mergeMasks(Mat& src, const Mat& dst) {
    if (src.rows != dst.rows || src.cols != dst.cols) {
        cerr << "Can't merge masks, incompatible sizes" << endl;
        return;
    }

    for (size_t i = 0; i < src.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            if (src.at<Vec3b>(i, j) == cvScalar2Vec3b(notSpecifiedColor)) {
                src.at<Vec3b>(i, j) = dst.at<Vec3b>(i, j);
            }
        }
    }
}

int main( int argc, char** argv )
{
    cv::CommandLineParser parser(argc, argv, "{help h | | }{ @input | ../data/fruits.jpg | }");
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    string filename = parser.get<string>("@input");
    img0 = imread(filename, 1);
    Mat imgGray;

    if( img0.empty() )
    {
        cout << "Couldn'g open image " << filename << ". Usage: watershed <image_name>\n";
        return 0;
    }
    help();

    unordered_set<CvScalar> validColors;
    initColorSet(validColors);

    namedWindow( IMAGE_WINDOW_NAME, WINDOW_NORMAL | CV_GUI_NORMAL);

    img0.copyTo(img);
    cvtColor(img, markerMask, COLOR_BGR2GRAY);
    cvtColor(markerMask, imgGray, COLOR_GRAY2BGR);
    markerMask = Scalar::all(0);
    imshow( IMAGE_WINDOW_NAME, img );
    resizeWindow(IMAGE_WINDOW_NAME, IMG_WIDTH, IMG_HEIGHT);
    setMouseCallback( IMAGE_WINDOW_NAME, onMouse, 0 );

    bool isColorSelectMode = false;
    for(;;)
    {
        char c = (char)waitKey(0);

        if( c == 27 ) {
            break;
        }

        if (c == 'm') {
            if (isColorSelectMode) {
                isColorSelectMode = false;
                cout << "Exiting color selecting mode" << endl;
            } else {
                isColorSelectMode = true;
                cout << "Please select color. Possible colors: \n"
                        // terrain
                        "\tt - brown(just terrain)\n"
                        "\tw - light-blue(snow)\n"
                        "\ty - yellow(sand)\n"
                        "\tg - dark-green(forest)\n"
                        "\tp - light-green(grass)\n"
                        "\tr - gray(roads)\n"
                        "\td - dark-gray(buildings)\n"
                        "\tb - blue(water)\n"
                        "\tc - light-gray(clouds)\n"
                        "\tu - black(unknown)\n"
                        "\tx - red(not specified)\n"
                        "Type '-' or '=' to replace violet or cyan color with selected color respectively" << endl;
            }
            continue;
        }

        switch (c) {
        case 'z':
            saveMask(genMaskFileName(filename));
            saveMarkers(genMarkersFileName(filename));
            break;
        case 'l':
            loadMask(genMaskFileName(filename));
            loadMarkers(genMarkersFileName(filename)) ;
            break;
        case ' ': {
            Mat wshed = runWatershed(img0, markerMask);

            if (wshed.empty()) {
                break;
            }

            curMask = wshed.clone();
            wshed = wshed*0.5 + imgGray*0.5;

            namedWindow( WATERSHED_TRANS_WINDOW_NAME, cv::WINDOW_NORMAL | CV_GUI_NORMAL);
            imshow( WATERSHED_TRANS_WINDOW_NAME, wshed );
            resizeWindow(WATERSHED_TRANS_WINDOW_NAME, IMG_WIDTH, IMG_HEIGHT);
            break;
        }
        case 13: { // Enter
            if(curMask.empty()) {
                cerr << "Mask is not created yet!" << endl;
                break;
            }

            Mat dst = runThresholdBasedMethod(img0);

            mergeMasks(curMask, dst);
            createMaskWindow();
            imshow(MASK_WINDOW_NAME, curMask);
            break;
        }
        case 's':
            if (curMask.empty()) {
                cerr << "Mask is not created yet!" << endl;
                continue;
            }
            createMaskWindow();
            imshow(MASK_WINDOW_NAME, curMask);
        default :
            if (!isColorSelectMode) {
                if( c == 'r' )
                {
                    markerMask = Scalar::all(0);
                    img0.copyTo(img);
                    imshow( IMAGE_WINDOW_NAME, img );
                    cout << "Main image and markers has been cleared" << endl;
                }
                else if (c == 'f') {
                    // TODO: do it in separate thread
                    filter(validColors);
                } else if (c >= '1' && c <= '9') {
                    cout << "Setting brush thikness to " << c - '0' << endl;
                    curThickness = c - '0';
                } else if (c == 'h') {
                    refreshMainImg();
                    cout << "Main image has been refreshed!" << endl;
                }
            } else {
                vector<Vec3b> from, to = {cvScalar2Vec3b(curColor)};
                switch (c) {
                case 't':
                    cout << "Selecting brown color(just terrain)" << endl;
                    curColor = justTerrainColor;
                    break;
                case 'w':
                    cout << "Selecting light-blue color(snow)" << endl;
                    curColor = snowColor;
                    break;
                case 'y':
                    cout << "Selecting yellow color(sand)" << endl;
                    curColor = sandColor;
                    break;
                case 'g':
                    cout << "Selecting dark-green color(forest)" << endl;
                    curColor = forestColor;
                    break;
                case 'p':
                    cout << "Selecting light-green color(grass)" << endl;
                    curColor = grassColor;
                    break;

                case 'r':
                    cout << "Selecting gray color(roads)" << endl;
                    curColor = roadsColor;
                    break;

                case 'd':
                    cout << "Selecting dark-gray color(buildings)" << endl;
                    curColor = buildingsColor;
                    break;

                case 'b':
                    cout << "Selecting blue color(water)" << endl;
                    curColor = waterColor;
                    break;

                case 'c':
                    cout << "Selecting light-gray color(clouds)" << endl;
                    curColor = cloudsColor;
                    break;

                case 'x':
                    cout << "Selecting red color(not specified)" << endl;
                    curColor = notSpecifiedColor;
                    break;

                case 'u':
                    cout << "Selecting black color(unknown)" << endl;
                    curColor = unknownColor;
                    break;

                case '-':
                    cerr << "Replacing violet color with selected color" << endl;

                    from = {Vec3b(255, 0, 255)};
                    recolorImg(curMask, {Vec3b(255, 0, 255)}, to);

                    createMaskWindow();
                    imshow(MASK_WINDOW_NAME, curMask);
                    break;

                case '=':
                    cerr << "Replacing cyan color with selected color" << endl;

                    from = {Vec3b(255, 255, 0)};
                    recolorImg(curMask, from, to);

                    createMaskWindow();
                    imshow(MASK_WINDOW_NAME, curMask);
                    break;

                case 9: // tab
                    break;

                case -23: // alt
                    break;

                default:
                    break;
                }
            }
        }
    }

    return 0;
}
