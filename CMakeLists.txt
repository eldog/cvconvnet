cmake_minimum_required(VERSION 2.8)

PROJECT (CvConvolutionalNet)

# IF() ENDIF() statements
SET(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

# Specify build-type as Debug if not specified already
#IF (NOT CMAKE_BUILD_TYPE)
#	SET(CMAKE_BUILD_TYPE Debug)
        SET(CMAKE_BUILD_TYPE Release)
        #ENDIF ()

# Produce verbose makefiles
# SET(CMAKE_VERBOSE_MAKEFILE ON)

# Sources for library
SET(CVCONVNET_SRCS
	src/cvconvnet.cpp
	src/cvconvnetparser.cpp
	src/cvconvolutionplane.cpp
	src/cvfastsigmoid.cpp
	src/cvgenericplane.cpp
        src/cvmaxoperatorplane.cpp
	src/cvmaxplane.cpp
	src/cvrbfplane.cpp
        src/cvregressionplane.cpp
	src/cvsubsamplingplane.cpp
	src/cvsourceplane.cpp
)

# Sources for tests
SET(TEST_SRCS
    test/cvmaxoperatorplane_test.cpp
)

# Sources for example files
SET (EXAMPLEIMG_SRCS example/testimg.cpp)
SET (EXAMPLEMNIST_SRCS example/testmnist.cpp)
SET (FEXAMPLEIMG_SRCS fexample/ftestimg.cpp)

SET (FACEDETECT_SRCS
        fexample/facedetect.cpp
        fexample/cnn.cpp)

# Ouptut directory for binaries
SET (EXECUTABLE_DIR bin)

# Here are common paths (in addition to default paths)
SET (INCLUDE_SEARCH_PATH
	/usr/local/include
        /usr/include
        /usr/local/include/boost
        /usr/local/include/opencv/
	/usr/include/opencv/
        /usr/local/include/opencv2/
        c:/program\ files/opencv/include
        /usr/lib/jvm/default-java/include
        /usr/lib/jvm/default-java/include/linux
)

SET (LIBRARY_SEARCH_PATH
        /usr/lib/
	/usr/local/lib/
)

# Find OpenCV and Expat
FIND_PATH(CV_H NAMES cv.h PATHS ${INCLUDE_SEARCH_PATH} )
FIND_PATH(HIGHGUI_H NAMES highgui.h PATHS ${INCLUDE_SEARCH_PATH} )
FIND_PATH(EXPAT_H NAMES expat.h	PATHS ${INCLUDE_SEARCH_PATH} )
FIND_PATH(JNI_H NAMES jni.h PATHS ${INCLUDE_SEARCH_PATH} )
FIND_PATH(JNI_MD_H NAMES jni_m.h PATHS ${INCLUDE_SEARCH_PATH} )
# Boost
FIND_PATH(BOOST_H NAMES unit_test.h PATHS ${INCLUDE_SEARCH_PATH} )

FIND_LIBRARY(LIBCV NAMES libopencv_core.so PATHS ${LIBRARY_SEARCH_PATH} )
FIND_LIBRARY(LIBHIGHGUI NAMES libopencv_highgui.so PATHS ${LIBRARY_SEARCH_PATH})
FIND_LIBRARY(LIBEXPAT NAMES expat PATHS ${LIBRARY_SEARCH_PATH} )
FIND_LIBRARY(LIBOBJDETECT NAMES libopencv_objdetect.so PATHS ${LIBRARY_SEARCH_PATH})
FIND_LIBRARY(LIBIMGPROC NAMES libopencv_imgproc.so PATHS ${LIBRARY_SEARCH_PATH})
FIND_LIBRARY(LIB_BOOST_TEST
                NAMES libboost_unit_test_framework.so
                PATHS ${LIBRARY_SEARCH_PATH})

INCLUDE_DIRECTORIES(include/
                    jni_fexample/
                    fexample/
                    ${CV_H}
                    ${HIGHGUI_H}
                    ${EXPAT_H}
                    ${JNI_H}
                    ${JNI_M_H}
)

# Here is out library
ADD_LIBRARY(cvconvnet SHARED ${CVCONVNET_SRCS})
ADD_LIBRARY(cvconvnet_static STATIC ${CVCONVNET_SRCS})
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# Here are our test programs
ADD_EXECUTABLE(testimg ${EXAMPLEIMG_SRCS})
ADD_EXECUTABLE(testmnist ${EXAMPLEMNIST_SRCS})
ADD_EXECUTABLE(ftestimg ${FEXAMPLEIMG_SRCS})
ADD_EXECUTABLE(facedetect ${FACEDETECT_SRCS})
ADD_EXECUTABLE(test_cvmaxoperatorplane ${TEST_SRCS})
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Compiler options are different for Release and Debug
IF (CMAKE_BUILD_TYPE MATCHES Release)
	# Highly optimized + cancel all assert()s
	ADD_DEFINITIONS(-O3 -DNDEBUG)
ELSE ()
	# Include debug info, profiling info, some text output
	ADD_DEFINITIONS(-O -pg -g -DDEBUG)
	# Set profiling for linker too
	SET_TARGET_PROPERTIES(testmnist PROPERTIES LINK_FLAGS "-pg")
ENDIF ()

# fpic
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC"  )
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC"  )

# We should link our test programs to libraries
TARGET_LINK_LIBRARIES(
    facedetect
    cvconvnet
    ${LIBCV}
    ${LIBHIGHGUI}
    ${LIBEXPAT}
    ${LIBOBJDETECT}
    ${LIBIMGPROC}
)
TARGET_LINK_LIBRARIES(
    testimg
    cvconvnet
    ${LIBCV}
    ${LIBHIGHGUI}
    ${LIBEXPAT}
    ${LIBIMGPROC}
)
TARGET_LINK_LIBRARIES(
    testmnist
    cvconvnet
    ${LIBCV}
    ${LIBHIGHGUI}
    ${LIBEXPAT}
)
TARGET_LINK_LIBRARIES(
    ftestimg
    cvconvnet
    ${LIBCV}
    ${LIBHIGHGUI}
    ${LIBEXPAT}
    ${LIBIMGPROC}
)
TARGET_LINK_LIBRARIES(
    test_cvmaxoperatorplane
    cvconvnet
    ${LIBCV}
    ${LIB_BOOST_TEST}
    ${LIBEXPAT}
)
FILE(GLOB files "include/*.h")
INSTALL(FILES ${files} DESTINATION /usr/local/include)
INSTALL(TARGETS cvconvnet
        LIBRARY         DESTINATION /usr/local/lib
)

