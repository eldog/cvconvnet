#include <iostream>

#include "opencv2/highgui/highgui.hpp"

#include "cnn.h"


int main(int argc, char* argv[])
{
    using namespace std;

    if (argc <= 2)
    {
        cerr << "Usage: " << endl ;
        cerr << "\tfacedetect <network.xml> <cascade.xml>" << endl;
        return 1;
    } // if

    Cnn cnn;

    if (!cnn.loadConvNet(argv[1]))
    {
        cerr << "Unable to load neural network" << endl;
        return 1;
    } // if

    if (!cnn.loadCascade(argv[2]))
    {
        cerr << "Unable to load cascade" << endl;
        return 1;
    } // if

    CvCapture* capture;
    cv::Mat frame;
    vector<cv::Rect> faces;
    capture = cvCaptureFromCAM(-1);
    if (capture)
    {
        while(true)
        {
            frame = cvQueryFrame(capture);

            if (!frame.empty())
            {
                faces = cnn.findFaces(frame);
                cnn.drawRectangles(faces, frame);
                imshow("Face Detect", frame);
                
            } // if
            else
            {
                cerr << "Empty frame!" << endl;
                break;
            } // else

            int c = cv::waitKey(10);
            if (((char) c == 'c'))
            {
                break;
            } // if
            else if (((char) c == 'f'))
            {
                for (int faceIndex = 0; faceIndex < faces.size(); faceIndex++)
                {
                    cv::Mat roi = cnn.cropFrame(frame, faces[faceIndex]);
                    double score = cnn.runConvNet(roi);
                    cerr << "Score: " << score << endl;
                } // for
            } // else if
        } // while
    } //if

} // main

