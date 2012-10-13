class FaceDetectTest
{
    static native long loadFaceDetector(String cascadeFile, String cnnFile);
    static native long findFaces(int width, int height, byte[] yuv, int[] rgba);

    static
    {
        System.loadLibrary("FaceDetectTest");
    } // static
}
