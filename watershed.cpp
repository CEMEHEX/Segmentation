#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <cstdio>
#include <iostream>

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
            "\tm - switch on/off color selecting mode" << endl;
}
Mat markerMask, img, curMask;
CvScalar curColor = CV_RGB(0, 0, 0);
Point prevPt(-1, -1);

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

    if( event == EVENT_LBUTTONDOWN ) {
        prevPt = Point(x,y);
    }
    else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON) )
    {
        Point pt(x, y);
        if( prevPt.x < 0 )
            prevPt = pt;
        line( markerMask, prevPt, pt, Scalar::all(255), 5, 8, 0 );
        line( img, prevPt, pt, Scalar::all(255), 5, 8, 0 );
        prevPt = pt;
        imshow("image", img);
    } else if( event == EVENT_RBUTTONUP || !(flags & EVENT_FLAG_RBUTTON) ) {
        prevPt = Point(-1,-1);
    }
    else if( event == EVENT_RBUTTONDOWN ) {
        prevPt = Point(x,y);
    }
    else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_RBUTTON) )
    {
        Point pt(x, y);
        if( prevPt.x < 0 )
            prevPt = pt;
        line( markerMask, prevPt, pt, Scalar::all(0), 15, 8, 0 );
        line( img, prevPt, pt, Scalar::all(0), 15, 8, 0 );
        prevPt = pt;
        imshow("image", img);
    } else {
        prevPt = Point(-1,-1);
    }
}

static void onMouse_Mask( int event, int x, int y, int flags, void* ) {
    switch( event ) {
    case CV_EVENT_MOUSEMOVE:
        break;

    case CV_EVENT_LBUTTONDOWN:
        mark(curMask, cvPoint(x, y), curColor);
        imshow("mask", curMask);
        break;

    case CV_EVENT_LBUTTONUP:
        break;
    }
}

const int IMG_WIDTH = 1200;
const int IMG_HEIGHT = 900;

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
    namedWindow( "image", WINDOW_NORMAL | CV_GUI_NORMAL);

    img0.copyTo(img);
    cvtColor(img, markerMask, COLOR_BGR2GRAY);
    cvtColor(markerMask, imgGray, COLOR_GRAY2BGR);
    markerMask = Scalar::all(0);
    imshow( "image", img );
    resizeWindow("image", IMG_WIDTH, IMG_HEIGHT);
    setMouseCallback( "image", onMouse, 0 );

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
                imshow( "image", img );
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

                namedWindow( "watershed transform", cv::WINDOW_NORMAL);
                imshow( "watershed transform", wshed );
                resizeWindow("watershed transform", IMG_WIDTH, IMG_HEIGHT);
            } else if (c == 's') {
                if (!(curMask.rows > 0 && curMask.cols > 0)) {
                    cerr << "Mask is not created yet!" << endl;
                    continue;
                }

                namedWindow( "mask", cv::WINDOW_NORMAL);
                resizeWindow("mask", IMG_WIDTH, IMG_HEIGHT);
                imshow("mask", curMask);
                setMouseCallback( "mask", onMouse_Mask, 0 );
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
