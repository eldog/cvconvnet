#include "cnn.h"

Cnn::Cnn()
{

} // Cnn

Cnn::~Cnn()
{
} // ~Cnn

bool Cnn::loadCascade(char* const filePath)
{
    return mFaceCascade.load(filePath);
} // Cnn::loadCascade

bool Cnn::loadConvNet(char* const filePath)
{
    using namespace std;
    ifstream ifs(filePath);
    const string xmlString((istreambuf_iterator<char> (ifs)), 
                           istreambuf_iterator<char>());
    return mConvNet.fromString(xmlString);
} // Cnn::loadConvNet

std::vector<cv::Rect> Cnn::findFaces(cv::Mat const inputImage)
{
    std::vector<cv::Rect> faces;
    cv::Mat greyInputImage;
    cv::cvtColor(inputImage, greyInputImage, CV_BGR2GRAY);
    cv::equalizeHist(greyInputImage, greyInputImage);
    mFaceCascade.detectMultiScale(greyInputImage, faces, 1.1, 2, 2, cv::Size(10, 10));
    return faces;
} // Cnn::findFaces

void Cnn::drawRectangles(std::vector<cv::Rect> rectangles, cv::Mat frame)
{
    for (int rectIndex = 1; rectIndex < rectangles.size(); rectIndex++)
    {
        cv::Point topLeft(rectangles[rectIndex].x, rectangles[rectIndex].y);
        cv::Point bottomRight(rectangles[rectIndex].x + rectangles[rectIndex].width, 
                              rectangles[rectIndex].y + rectangles[rectIndex].height);
        cv::rectangle(frame, 
                      topLeft, 
                      bottomRight,
                      cv::Scalar(0, 255, 0, 0), /* the color; green */
                      1, /* thickness */
                      8, /* line type */
                      0 /* shift */);
    } // for
} // Cnn::drawRectangles

cv::Mat Cnn::cropFrame(cv::Mat inputFrame, cv::Rect roi)
{
    cv::Mat scaled;
    cv::resize(inputFrame(roi), scaled, cv::Size(128, 128));
    return scaled;
} // Cnn::cropFrame

double Cnn::runConvNet(cv::Mat const frame)
{   
    cv::Mat greyInputImage;
    cv::cvtColor(frame, greyInputImage, CV_BGR2GRAY);
    IplImage image = greyInputImage;
    return mConvNet.fprop(&image);
} // Cnn::runConvNet

