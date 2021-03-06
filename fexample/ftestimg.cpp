/*****************************************************************************
 IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING. By
downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install, copy or
use the software.

Contributors License Agreement

Copyright© 2007, Akhmed Umyarov. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.
- The name of Contributor may not be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
All information provided related to future Intel products and plans is
preliminary and subject to change at any time, without notice.
*****************************************************************************/

/*!\file
 * \brief Sample program using CvConvNet object
 * \author Akhmed Umyarov & Lloyd Henning
 * \date 2012
 */
#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/program_options.hpp>
#include <opencv/highgui.h> 

#include "cvconvnet.h"


using namespace std;

/*! The program takes filenames from commandline parameters
 * and runs the network over the files.
 * The program displays the result in a small-window
 * where predicted value is displayed in green.
 * Usage of the program is the following:
 * $ ./testimg [net.xml] [file1] [file2] [file3] ...
 * 
 * net.xml is XML description of the network
 * files must be 128x128
 * no error checking is done!
 */
int main(int argc, char *argv[])
{
    if (argc <=2 )
    {
        cerr << "Usage: " << endl 
             << "\ttestimg <network.xml> <imagefile(s)>" << endl;
        return 1;
    }

    // Create empty net object
    CvConvNet net;

    // Source featuremap size
    CvSize inputsz = cvSize(128, 128);

    // Load mnist.xml file into a std::string called xml
    ifstream ifs(argv[1]);
    string xml ( (istreambuf_iterator<char> (ifs)) , istreambuf_iterator<char>() );
    
    // Create network from XML string
    if ( !net.fromString(xml) )
    {
            cerr << "*** ERROR: Can't load net from XML" << endl << "Check file "<< argv[1] << endl;
            return 1;
    }

    // create some GUI
    cvNamedWindow("Image", CV_WINDOW_AUTOSIZE); 
    cvMoveWindow("Image", inputsz.height, inputsz.width);
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.0, 1.0);

    // Grayscale img pointer
    IplImage* img;

    // Also create a color image (for display)
    IplImage *colorimg = cvCreateImage( inputsz, IPL_DEPTH_8U, 3 );

    // Cycle over input images
    for (int i=2; i<argc; i++)
    {
            // Load the image
            if ((img = cvLoadImage( argv[i], CV_LOAD_IMAGE_GRAYSCALE )) == NULL)
            {
                    cerr << "ERROR: Bad image file: " << argv[i] << endl; 
                    break;
            }

            //CvScalar s;
            //s = cvGet2D(img, 3, 3);
            //cout << "s val: " << s.val[0] << endl;

            cv::Mat matImg(img);
            double sourcePlaneFeatureMapValues[] = 
                        {
                            1, 1, 2, 2,
                            1, 1, 2, 2,
                            3, 3, 4, 4,
                            3, 3, 4, 4
                        };
            
            // FOR TESTING
            //CvMat cvmat = cvMat(4, 
            //                            4, 
            //                            CV_64FC1, 
            //                            sourcePlaneFeatureMapValues);


            CvMat *cvmat = cvCreateMat(img->height, img->width, CV_32FC1);

            cvConvert(img, cvmat);
            cv::Mat floatMat = cvmat;
            floatMat -= 108.08409242f;
            floatMat /= 255.0f;
            //floatMat -= 109.799621582;
            //floatMat /= 234.0f;
            //
            // normalise it
            //matImg = (matImg - 107.609341755);// / 255.0;
            // cout << matImg << endl;
            IplImage newImage = matImg;


            cout << setprecision( 8 ) << right << fixed;
            cout << (double) cvmGet(cvmat, 0, 0) << endl;
            //cout << (double) cvmGet(cvmat, 127, 127) << endl;
            //for ( int row = 0; row < 128; ++ row )
           // {
            //    for ( int col = 0; col < 128; ++ col )
             //   {
             //       cout << setw( 5 ) 
             //            << (double)cvmGet( cvmat, row, col ) 
             //            << " ";
             //   }
             //  cout << endl;
            //}
            // Forward propagate the grayscale (8 bit) image and get the value
            ostringstream val;
            val << (float) net.fprop(cvmat);
            cout << val.str() << endl;
            // Make image colorful
            cvCvtColor(&newImage,colorimg,CV_GRAY2RGB);

            // Draw green text for the recognized number on top of the image
            cvPutText(colorimg, val.str().c_str(), cvPoint(0,inputsz.height/2), &font, CV_RGB(0,255,0));

            // show the image
            cvShowImage("Image", colorimg );
            cvWaitKey(1000);
            
            cvReleaseImage(&img);
    }
    // Free buffers
    cvReleaseImage(&colorimg);

    return 0;
}
