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
            "\tz - save mask(don't ask me why z)\n"
            "\tl - load mask\n"
            "\tf - apply filter\n"
            "\t1-9 - set brush thickness" << endl;
}
Mat markerMask, img, curMask;
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

const string IMAGE_WINDOW_NAME("image");
const string WATERSHED_TRANS_WINDOW_NAME("watershed transform");
const string MASK_WINDOW_NAME("mask");

namespace std {
template<>
class hash<CvScalar> {
public:
    size_t operator()(const CvScalar &cvScalar) const
    {
        const size_t prime = 31;
        size_t res = 1;

        size_t h1 = std::hash<double>()(cvScalar.val[0]);
        size_t h2 = std::hash<double>()(cvScalar.val[1]);
        size_t h3 = std::hash<double>()(cvScalar.val[2]);
        size_t h4 = std::hash<double>()(cvScalar.val[3]);

        res = prime * res + (h1 ^ (h1 >> 32));
        res = prime * res + (h2 ^ (h2 >> 32));
        res = prime * res + (h3 ^ (h3 >> 32));
        res = prime * res + (h4 ^ (h4 >> 32));

        return res;
    }
};
}

inline bool operator==(const CvScalar& lhs, const CvScalar& rhs)
{
    return lhs.val[0] == rhs.val[0] &&
            lhs.val[1] == rhs.val[1] &&
            lhs.val[2] == rhs.val[2] &&
            lhs.val[3] == rhs.val[3];
}

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
        line( img, prevPt, pt, Scalar::all(255), curThickness, 8, 0 );
        prevPt = pt;
        imshow(IMAGE_WINDOW_NAME, img);
    }
    else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_RBUTTON) && (flags & EVENT_FLAG_CTRLKEY) )
    {
        Point pt(x, y);
        if( prevPt.x < 0 )
            prevPt = pt;
        line( markerMask, prevPt, pt, Scalar::all(0), 3 * curThickness, 8, 0 );
        line( img, prevPt, pt, Scalar::all(0), 3 * curThickness, 8, 0 );
        prevPt = pt;
        imshow(IMAGE_WINDOW_NAME, img);
    } else {
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

inline string genMaskFileName(const string& filename) {
    const std::string ext(".jpg");
    if ( filename != ext &&
         filename.size() > ext.size() &&
         filename.substr(filename.size() - ext.size()) == ".jpg" )
    {
        return filename.substr(0, filename.size() - ext.size()) + "_mask.png";
    }

    return "";
}

inline void saveMask(const string& maskFilename) {
    if ( !maskFilename.empty() )
    {
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
    } else {
        cerr << "Something went wrong, can't generate name for mask" << endl;
    }
}

inline void loadMask(const string& maskFileName) {
    if (!file_exists(maskFileName)) {
        cerr << "No file to load!" << endl;
        return;
    }

    curMask = imread(maskFileName, 1);
    imshow(MASK_WINDOW_NAME, curMask);
}

const int IMG_WIDTH = 1200;
const int IMG_HEIGHT = 900;

inline void createMaskWindow() {
    namedWindow( MASK_WINDOW_NAME, cv::WINDOW_NORMAL | CV_GUI_NORMAL);
    resizeWindow(MASK_WINDOW_NAME, IMG_WIDTH, IMG_HEIGHT);
    setMouseCallback( MASK_WINDOW_NAME, onMouse_Mask, 0 );
}

inline CvScalar getColor(Mat& img, int i, int j) {
    auto vec3bCol = img.at<Vec3b>(i, j);
    auto r = vec3bCol[2];
    auto g = vec3bCol[1];
    auto b = vec3bCol[0];
    return CV_RGB(r, g, b);
}

inline Vec3b cvScalar2Vec3b(CvScalar& sc) {
    auto r = sc.val[0];
    auto g = sc.val[1];
    auto b = sc.val[2];
    return Vec3b(r, g, b);
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
        cerr << "No valid colors in this window, skipping" << endl;
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

    for (size_t i = 0; i < img.cols / winSize; ++i) {
        for (size_t j = 0; j < img.rows / winSize; ++j) {
            processWindow(img, validColors, winSize, i * winSize, j * winSize);
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

int main( int argc, char** argv )
{
    cv::CommandLineParser parser(argc, argv, "{help h | | }{ @input | ../data/fruits.jpg | }");
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    string filename = parser.get<string>("@input");
    Mat img0 = imread(filename, 1), imgGray;

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
                        "\tw - white(snow)\n"
                        "\ty - yellow(sand)\n"
                        "\tg - dark-green(forest)\n"
                        "\tp - light-green(grass)\n"
                        "\tr - gray(roads)\n"
                        "\td - dark-gray(buildings)\n"
                        "\tb - blue(water)\n"
                        "\tc - light-gray(clouds)" << endl;
            }
            continue;
        }

        if (!isColorSelectMode) {
            if( c == 'r' )
            {
                markerMask = Scalar::all(0);
                img0.copyTo(img);
                imshow( IMAGE_WINDOW_NAME, img );
            }
            else if( c == 'w' || c == ' ' )
            {
                int i, j, compCount = 0;
                vector<vector<Point> > contours;
                vector<Vec4i> hierarchy;

                findContours(markerMask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

                if( contours.empty() )
                    continue;
                Mat markers(markerMask.size(), CV_32S);
                markers = Scalar::all(0);
                int idx = 0;
                for( ; idx >= 0; idx = hierarchy[idx][0], compCount++ )
                    drawContours(markers, contours, idx, Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);

                if( compCount == 0 )
                    continue;

                vector<Vec3b> colorTab;
                for( i = 0; i < compCount; i++ )
                {
                    int b = theRNG().uniform(0, 255);
                    int g = theRNG().uniform(0, 255);
                    int r = theRNG().uniform(0, 255);

                    colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
                }

                double t = (double)getTickCount();
                watershed( img0, markers );
                t = (double)getTickCount() - t;
                printf( "execution time = %gms\n", t*1000./getTickFrequency() );

                Mat wshed(markers.size(), CV_8UC3);

                // paint the watershed image
                for( i = 0; i < markers.rows; i++ )
                    for( j = 0; j < markers.cols; j++ )
                    {
                        int index = markers.at<int>(i,j);
                        if( index == -1 )
                            wshed.at<Vec3b>(i,j) = Vec3b(255,255,255);
                        else if( index <= 0 || index > compCount )
                            wshed.at<Vec3b>(i,j) = Vec3b(0,0,0);
                        else
                            wshed.at<Vec3b>(i,j) = colorTab[index - 1];
                    }

                curMask = wshed.clone();
                wshed = wshed*0.5 + imgGray*0.5;

                namedWindow( WATERSHED_TRANS_WINDOW_NAME, cv::WINDOW_NORMAL | CV_GUI_NORMAL);
                imshow( WATERSHED_TRANS_WINDOW_NAME, wshed );
                resizeWindow(WATERSHED_TRANS_WINDOW_NAME, IMG_WIDTH, IMG_HEIGHT);
            } else if (c == 's') {
                if (!(curMask.rows > 0 && curMask.cols > 0)) {
                    cerr << "Mask is not created yet!" << endl;
                    continue;
                }
                createMaskWindow();
                imshow(MASK_WINDOW_NAME, curMask);
            } else if (c == 'z') {
                saveMask(genMaskFileName(filename));
            } else if (c == 'l') {
                createMaskWindow();
                loadMask(genMaskFileName(filename));
            } else if (c == 'f') {
                // TODO: do it in separate thread
                filter(validColors);
            } else if (c >= '1' && c <= '9') {
                cout << "Setting brush thikness to " << curThickness << endl;
                curThickness = c - '0';
            }
        } else {
            switch (c) {
            case 't':
                cout << "Selecting brown color(just terrain)" << endl;
                curColor = justTerrainColor;
                break;
            case 'w':
                cout << "Selecting white color(snow)" << endl;
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
            case 'm':
                isColorSelectMode = false;
                cout << "Exiting color selecting mode" << endl;
                break;

            default:
                cout << "Unknown color" << endl;
                curColor = unknownColor;
            }
        }
    }

    return 0;
}
