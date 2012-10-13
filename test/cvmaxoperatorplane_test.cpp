#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE cvmaxoperatorplane test
#define CHECK_MESSAGE(a, b) {\
                                BOOST_CHECK_MESSAGE(a == b,\
                                                  "target:" << b <<\
                                        " result:" << a);\
}

#include <vector>
#include <boost/test/unit_test.hpp>
#include <opencv/cv.h>

#include "cvconvolutionplane.h"
#include "cvgenericplane.h"
#include "cvmaxoperatorplane.h"
#include "cvregressionplane.h"
#include "cvsourceplane.h"

CvSourcePlane createTestCvSourcePlane()
{
    CvSize featureMapSize = cvSize(8, 8);
    return CvSourcePlane("test_source_plane", featureMapSize);
} // createTestCvSourcePlane

CvMaxOperatorPlane createTestMaxOperatorPlane(CvSize featureMapSize,
                                              CvSize neuronSize)
{
    return CvMaxOperatorPlane("test_max", featureMapSize, neuronSize);
} // createTestMaxOperatorPlane


BOOST_AUTO_TEST_CASE( full_fprop_test )
{
    CvSourcePlane sourcePlane = createTestCvSourcePlane();

    // test the basic forward propagation
    double sourcePlaneFeatureMapValues[] =
                        {
                              0,   1,   2,   3,   4,   5,   6,   7,
                              8,   9,  10,  11,  12,  13,  14,  15,
                             16,  17,  18,  19,  20,  21,  22,  23,
                             24,  25,  26,  27,  28,  29,  30,  31,
                             32,  33,  34,  35,  36,  37,  38,  39,
                             40,  41,  42,  43,  44,  45,  46,  47,
                             48,  49,  50,  51,  52,  53,  54,  55,
                             56,  57,  58,  59,  60,  61,  62,  63
                        };
    CvMat sourcePlaneFeatureMap = cvMat(8,
                                        8,
                                        CV_64FC1,
                                        sourcePlaneFeatureMapValues);

    CHECK_MESSAGE(sourcePlane.setfmap(&sourcePlaneFeatureMap), 1);

    std::vector<CvGenericPlane *> parentPlanes;
    parentPlanes.push_back(&sourcePlane);

    CvConvolutionPlane* convolutionPlane= new CvConvolutionPlane("test_conv",
                                                                cvSize(6, 6),
                                                                cvSize(3, 3));
     // Don't forget the bias
    double convolutionPlaneWeights[] = {
                                          0.1,
                                          0.1,   0.1,   0.1,
                                          0.1,   0.1,   0.1,
                                          0.1,   0.1,   0.1
                                      };
    std::vector<double> weights(convolutionPlaneWeights,
                                convolutionPlaneWeights
                                +
                                sizeof(convolutionPlaneWeights)
                                / sizeof(double));

    CHECK_MESSAGE(convolutionPlane->connto(parentPlanes), 1);
    CHECK_MESSAGE(convolutionPlane->setweight(weights), 1);

    std::vector<CvGenericPlane*> maxParentPlanes;
    maxParentPlanes.push_back(convolutionPlane);

    CvMaxOperatorPlane* maxOperatorPlane =
        new CvMaxOperatorPlane("test_max_plane", cvSize(3, 3), cvSize(2, 2));
    CHECK_MESSAGE(maxOperatorPlane->connto(maxParentPlanes), 1);

    std::vector<CvGenericPlane*> regressionParentPlanes;
    regressionParentPlanes.push_back(maxOperatorPlane);

    double regressionPlaneWeights[] = {
                                              1,
                                              1,   1,   1,
                                              1,   1,   1,
                                              1,   1,   1
                                      };
    std::vector<double> regWeights(regressionPlaneWeights,
                                    regressionPlaneWeights
                                    +
                                    sizeof(regressionPlaneWeights)
                                    / sizeof(double));


    CvRegressionPlane* regressionPlane =
        new CvRegressionPlane("test_reg_plane", cvSize(3, 3));
    CHECK_MESSAGE(regressionPlane->connto(regressionParentPlanes), 1);
    CHECK_MESSAGE(regressionPlane->setweight(regWeights), 1);

    CvMat* fprop1 = convolutionPlane->fprop();
    CHECK_MESSAGE(cvmGet(fprop1, 0, 0), tanh(8.2));
    CHECK_MESSAGE(cvmGet(fprop1, 0, 1), tanh(9.1));
    CHECK_MESSAGE(cvmGet(fprop1, 1, 0), tanh(15.4));
    CHECK_MESSAGE(cvmGet(fprop1, 1, 1), tanh(16.3));
    CHECK_MESSAGE(cvmGet(fprop1, 5, 5), tanh(48.7));

    CvMat* maxfprop = maxOperatorPlane->fprop();
    CHECK_MESSAGE(cvmGet(maxfprop, 0, 0), 16.3);
    CHECK_MESSAGE(cvmGet(maxfprop, 0, 1), 18.1);
    CHECK_MESSAGE(cvmGet(maxfprop, 0, 2), 19.9);
    CHECK_MESSAGE(cvmGet(maxfprop, 1, 0), 30.7);
    CHECK_MESSAGE(cvmGet(maxfprop, 1, 1), 32.5);
    CHECK_MESSAGE(cvmGet(maxfprop, 1, 2), 34.3);
    CHECK_MESSAGE(cvmGet(maxfprop, 2, 0), 45.1);
    CHECK_MESSAGE(cvmGet(maxfprop, 2, 1), 46.9);
    CHECK_MESSAGE(cvmGet(maxfprop, 2, 2), 48.7);

    CvMat* regfprop = regressionPlane->fprop();
    CHECK_MESSAGE(cvmGet(regfprop, 0, 0), 293.5);
} // BOOST_AUTO_TEST_CASE

BOOST_AUTO_TEST_CASE( cvconvolutionplane_test )
{
    CvSourcePlane sourcePlane = createTestCvSourcePlane();

    // test the basic forward propagation
    double sourcePlaneFeatureMapValues[] =
                        {
                              0,   1,   2,   3,   4,   5,   6,   7,
                              8,   9,  10,  11,  12,  13,  14,  15,
                             16,  17,  18,  19,  20,  21,  22,  23,
                             24,  25,  26,  27,  28,  29,  30,  31,
                             32,  33,  34,  35,  36,  37,  38,  39,
                             40,  41,  42,  43,  44,  45,  46,  47,
                             48,  49,  50,  51,  52,  53,  54,  55,
                             56,  57,  58,  59,  60,  61,  62,  63
                        };
    CvMat sourcePlaneFeatureMap = cvMat(8,
                                        8,
                                        CV_64FC1,
                                        sourcePlaneFeatureMapValues);

    CHECK_MESSAGE(sourcePlane.setfmap(&sourcePlaneFeatureMap), 1);

    std::vector<CvGenericPlane *> parentPlanes;
    parentPlanes.push_back(&sourcePlane);

    CvConvolutionPlane* convolutionPlane= new CvConvolutionPlane("test_conv",
                                                                cvSize(6, 6),
                                                                cvSize(3, 3));
     // Don't forget the bias
    double convolutionPlaneWeights[] = {
                                          0.1,
                                          0.1,   0.1,   0.1,
                                          0.1,   0.1,   0.1,
                                          0.1,   0.1,   0.1
                                      };
    std::vector<double> weights(convolutionPlaneWeights,
                                convolutionPlaneWeights
                                +
                                sizeof(convolutionPlaneWeights)
                                / sizeof(double));

    CHECK_MESSAGE(convolutionPlane->connto(parentPlanes), 1);
    CHECK_MESSAGE(convolutionPlane->setweight(weights), 1)

    CvMat* fprop1 = convolutionPlane->fprop();
    CHECK_MESSAGE(cvmGet(fprop1, 0, 0), tanh(8.2));
    CHECK_MESSAGE(cvmGet(fprop1, 0, 1), tanh(9.1));
    CHECK_MESSAGE(cvmGet(fprop1, 1, 0), tanh(15.4));
    CHECK_MESSAGE(cvmGet(fprop1, 1, 1), tanh(16.3));
    CHECK_MESSAGE(cvmGet(fprop1, 5, 5), tanh(48.7));

} // BOOST_AUTO_TEST_CASE

BOOST_AUTO_TEST_CASE( cvregressionplane_test )
{
    std::vector<CvGenericPlane *> parentPlanes;
    for (int i = 0; i < 1; i++)
    {
        CvSourcePlane* sourcePlane = new CvSourcePlane("test_source_plane",
                                                       cvSize(3, 3));
        // test the basic forward propagation
        double sourcePlaneFeatureMapValues[] = {
                                                  1,   1,   1,
                                                  1,   1,   1,
                                                  1,   1,   1
                                               };

        CvMat sourcePlaneFeatureMap = cvMat(3,
                                            3,
                                            CV_64FC1,
                                            sourcePlaneFeatureMapValues);

        CHECK_MESSAGE(sourcePlane->setfmap(&sourcePlaneFeatureMap), 1);
        parentPlanes.push_back(sourcePlane);

        // Don't forget the bias
        double regressionPlaneWeights[] = {
                                              1,
                                              1,   1,   1,
                                              1,   1,   1,
                                              1,   1,   1
                                          };
        std::vector<double> weights(regressionPlaneWeights,
                                    regressionPlaneWeights
                                    +
                                    sizeof(regressionPlaneWeights)
                                    / sizeof(double));

        CvRegressionPlane* regressionPlane
                    = new CvRegressionPlane("test_regression_plane",
                                            cvSize(3, 3));
        CHECK_MESSAGE(regressionPlane->connto(parentPlanes),  1);
        CHECK_MESSAGE(regressionPlane->setweight(weights), 1);
        CHECK_MESSAGE(cvmGet(regressionPlane->fprop(), 0, 0), 10);

        parentPlanes.push_back(sourcePlane);
        // Don't forget the bias
        double regressionPlaneWeights1[] = {
                                              1,
                                              1,   1,   1,
                                              1,   1,   1,
                                              1,   1,   1,
                                              1,   1,   1,
                                              1,   1,   1,
                                              1,   1,   1
                                          };
        std::vector<double> weights1(regressionPlaneWeights1,
                                    regressionPlaneWeights1
                                    +
                                    sizeof(regressionPlaneWeights1)
                                    / sizeof(double));

        CvRegressionPlane* regressionPlane1
                    = new CvRegressionPlane("test_regrssion_plane2",
                                            cvSize(3, 3));
        CHECK_MESSAGE(regressionPlane1->connto(parentPlanes), 1);
        CHECK_MESSAGE(regressionPlane1->setweight(weights1), 1);
        CHECK_MESSAGE(cvmGet(regressionPlane1->fprop(), 0, 0), 19);




    } // for

    // Test different valued weigths etc with one parent.
    CvSourcePlane* multiValSrcPlane = new CvSourcePlane("m_v_sp", cvSize(2, 2));
    double multiValSrcPlaneVal[] = {
                                    1.0, 2.0,
                                    3.0, 4.0
                                 };
    CvMat multiValSrcPlaneValMat = cvMat(2,
                                         2,
                                         CV_64FC1,
                                         multiValSrcPlaneVal);
    CHECK_MESSAGE(multiValSrcPlane->setfmap(&multiValSrcPlaneValMat), 1);

    std::vector<CvGenericPlane*> multiValRegPlaneParents;
    multiValRegPlaneParents.push_back(multiValSrcPlane);

    double multiValRegPlaneWeightVals[] = {
                                            2.0,
                                            6.0, 7.0,
                                            8.0, 9.0
                                        };
    std::vector<double> multiValRegPlaneWeights(multiValRegPlaneWeightVals,
                                                multiValRegPlaneWeightVals
                                                +
                                                sizeof(multiValRegPlaneWeightVals)
                                                / sizeof(double));
    CvRegressionPlane* multiValRegPlane = new CvRegressionPlane("m_v_rp",
                                                                cvSize(2, 2));
    CHECK_MESSAGE(multiValRegPlane->connto(multiValRegPlaneParents), 1);
    CHECK_MESSAGE(multiValRegPlane->setweight(multiValRegPlaneWeights), 1);
    CHECK_MESSAGE(cvmGet(multiValRegPlane->fprop(), 0, 0), 82);

    // test with multiple parents
    CvSourcePlane* multiValSrcPlane2 = new CvSourcePlane("m_v_sp_2", cvSize(2, 2));
    double multiValSrcPlaneVal2[] = {
                                      2.0, 4.0,
                                      6.0, 8.0
                                    };
    CvMat multiValSrcPlaneValMat2 = cvMat(2,
                                         2,
                                         CV_64FC1,
                                         multiValSrcPlaneVal2);
    CHECK_MESSAGE(multiValSrcPlane2->setfmap(&multiValSrcPlaneValMat2), 1);
    multiValRegPlaneParents.push_back(multiValSrcPlane2);

    double multiValRegPlaneWeightVals2[] = {
                                            2.0,
                                            6.0, 7.0,
                                            8.0, 9.0,
                                            11.0, 12.0,
                                            13.0, 14.0
                                            };
    std::vector<double> multiValRegPlaneWeights2(multiValRegPlaneWeightVals2,
                                                multiValRegPlaneWeightVals2
                                                +
                                                sizeof(multiValRegPlaneWeightVals2)
                                                / sizeof(double));
    CvRegressionPlane* multiValRegPlane2= new CvRegressionPlane("m_v_rp2",
                                                                cvSize(2, 2));
    CHECK_MESSAGE(multiValRegPlane2->connto(multiValRegPlaneParents), 1);
    CHECK_MESSAGE(multiValRegPlane2->setweight(multiValRegPlaneWeights2), 1);
    CHECK_MESSAGE(cvmGet(multiValRegPlane2->fprop(), 0, 0), 342);

    //CvRegressionPlane regressionPlane
    //   = new CvRegressionPlane("test_regression_plane",
    //                            cvSize()
} // BOOST_AUTO_TEST_CASE

BOOST_AUTO_TEST_CASE( cvmaxoperatorplane_test )
{
    CvSourcePlane sourcePlane = createTestCvSourcePlane();

    // test the basic forward propagation
    double sourcePlaneFeatureMapValues[] =
                        {
                              0,   1,   2,   3,   4,   5,   6,   7,
                              8,   9,  10,  11,  12,  13,  14,  15,
                             16,  17,  18,  19,  20,  21,  22,  23,
                             24,  25,  26,  27,  28,  29,  30,  31,
                             32,  33,  34,  35,  36,  37,  38,  39,
                             40,  41,  42,  43,  44,  45,  46,  47,
                             48,  49,  50,  51,  52,  53,  54,  55,
                             56,  57,  58,  59,  60,  61,  62,  63
                        };
    CvMat sourcePlaneFeatureMap = cvMat(8,
                                        8,
                                        CV_64FC1,
                                        sourcePlaneFeatureMapValues);

    CHECK_MESSAGE(sourcePlane.setfmap(&sourcePlaneFeatureMap), 1);

    std::vector<CvGenericPlane *> parentPlanes;
    parentPlanes.push_back(&sourcePlane);

    // Test a 1 x 1 output
    CvMaxOperatorPlane maxOperatorPlane =
        createTestMaxOperatorPlane(cvSize(1, 1), cvSize(8, 8));
    maxOperatorPlane.connto(parentPlanes);
    CHECK_MESSAGE(cvmGet(maxOperatorPlane.fprop(), 0, 0), 63);

    // Test a 2 x 2 output
    CvMaxOperatorPlane maxOperatorPlane2 =
        createTestMaxOperatorPlane(cvSize(2, 2), cvSize(4, 4));
    maxOperatorPlane2.connto(parentPlanes);
    CvMat* fprop1 = maxOperatorPlane2.fprop();
    BOOST_CHECK(fprop1 != 0);
    CHECK_MESSAGE(cvmGet(fprop1, 0, 0), 27);
    CHECK_MESSAGE(cvmGet(fprop1, 0, 1), 31);
    CHECK_MESSAGE(cvmGet(fprop1, 1, 0), 59);
    CHECK_MESSAGE(cvmGet(fprop1, 1, 1), 63);

    CvMaxOperatorPlane maxOperatorPlane3 =
        createTestMaxOperatorPlane(cvSize(4, 4), cvSize(2, 2));
    maxOperatorPlane3.connto(parentPlanes);
    CvMat * fprop2 = maxOperatorPlane3.fprop();
    BOOST_CHECK(fprop2 != 0);
    CHECK_MESSAGE(cvmGet(fprop2, 0, 0), 9);
    CHECK_MESSAGE(cvmGet(fprop2, 0, 1), 11);
    CHECK_MESSAGE(cvmGet(fprop2, 0, 2), 13);
    CHECK_MESSAGE(cvmGet(fprop2, 0, 3), 15);
    CHECK_MESSAGE(cvmGet(fprop2, 1, 0), 25);
    CHECK_MESSAGE(cvmGet(fprop2, 1, 1), 27);
    CHECK_MESSAGE(cvmGet(fprop2, 1, 2), 29);
    CHECK_MESSAGE(cvmGet(fprop2, 1, 3), 31);
    CHECK_MESSAGE(cvmGet(fprop2, 2, 0), 41);
    CHECK_MESSAGE(cvmGet(fprop2, 2, 1), 43);
    CHECK_MESSAGE(cvmGet(fprop2, 2, 2), 45);
    CHECK_MESSAGE(cvmGet(fprop2, 2, 3), 47);
    CHECK_MESSAGE(cvmGet(fprop2, 3, 0), 57);
    CHECK_MESSAGE(cvmGet(fprop2, 3, 1), 59);
    CHECK_MESSAGE(cvmGet(fprop2, 3, 2), 61);
    CHECK_MESSAGE(cvmGet(fprop2, 3, 3), 63);

} // BOOST_AUTO_TEST_CASE

BOOST_AUTO_TEST_CASE( cvsourceplane_test )
{
    CvSourcePlane sourcePlane = createTestCvSourcePlane();
    // Create our source plane to test
    int testFeatureMapValues[] = {
                                      0,   1,   2,   3,   4,   5,   6,   7,
                                      8,   9,  10,  11,  12,  13,  14,  15,
                                     16,  17,  18,  19,  20,  21,  22,  23,
                                     24,  25,  26,  27,  28,  29,  30,  31,
                                     32,  33,  34,  35,  36,  37,  38,  39,
                                     40,  41,  42,  43,  44,  45,  46,  47,
                                     48,  49,  50,  51,  52,  53,  54,  55,
                                     56,  57,  58,  59,  60,  61,  62,  63
                                 };
    CvMat testFeatureMap = cvMat(8, 8, CV_8UC1, testFeatureMapValues);
    CHECK_MESSAGE(sourcePlane.setfmap(&testFeatureMap), 1);
} // BOOST_AUTO_TEST_CASE

