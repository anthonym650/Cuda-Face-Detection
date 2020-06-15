
// CUDA C++ program to detects face in a video 

// Include required header files from OpenCV directory 
// Include required header files from CUDA directory

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream> 
#include <ctime>
#include <ratio>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;

string cascadeName, nestedCascadeName;


void FaceDetect(unsigned int frame[], Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale, int frame_width, int frame_height, size_t frame_size, int type,int filter);
void SerialGaussianFilter(const unsigned int d_src[], unsigned int d_dst[], int r_width, int r_height, int width, int height, int x, int y, int filter);



__global__ void CudaGaussianFilter(const unsigned int d_src[], unsigned int d_dst[], int r_width, int r_height, int width, int height, int x, int y, int filter);
__global__ void linearizeDataKernel(const unsigned char d_src[], unsigned int d_dst[], int width, int height, size_t frame_size, int channels);
__global__ void copyAndDisplayKernel(unsigned char d_dst[], const unsigned int d_src[], int width, int height, size_t frame_size, int channels);


__global__ void CudaGaussianFilter(const unsigned int d_src[], unsigned int d_dst[], int r_width, int r_height, int width, int height, int x, int y, int filter)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

        if ( (i >= y && j >= x) && (i < y + r_height - 2 && j < x + r_width - 2) && (i > 2 && j > 2)) {
            //3x3 inside
            uchar3 rgb; // (i)(j)
            rgb.x = d_src[i * width + j];
            rgb.y = d_src[(height + i) * width + j];
            rgb.z = d_src[(height * 2 + i) * width + j];

            uchar3 rgb1; //(i-1)(j-1)
            rgb1.x = d_src[(i - 1) * width + (j - 1)];
            rgb1.y = d_src[(height + (i - 1)) * width + (j - 1)];
            rgb1.z = d_src[(height * 2 + (i - 1)) * width + (j - 1)];

            uchar3 rgb2; //(i-1)(j)
            rgb2.x = d_src[(i - 1) * width + j];
            rgb2.y = d_src[(height + (i - 1)) * width + j];
            rgb2.z = d_src[(height * 2 + (i - 1)) * width + j];

            uchar3 rgb3; //(i-1)(j+1)
            rgb3.x = d_src[(i - 1) * width + (j + 1)];
            rgb3.y = d_src[(height + (i - 1)) * width + (j + 1)];
            rgb3.z = d_src[(height * 2 + (i - 1)) * width + (j + 1)];

            uchar3 rgb4; //(i)(j-1)
            rgb4.x = d_src[(i)*width + (j - 1)];
            rgb4.y = d_src[(height + (i)) * width + (j - 1)];
            rgb4.z = d_src[(height * 2 + (i)) * width + (j - 1)];

            uchar3 rgb5; //(i)(j+1)
            rgb5.x = d_src[i * width + (j + 1)];
            rgb5.y = d_src[(height + i) * width + (j + 1)];
            rgb5.z = d_src[(height * 2 + i) * width + (j + 1)];

            uchar3 rgb6; //(i+1)(j-1)
            rgb6.x = d_src[(i + 1) * width + (j - 1)];
            rgb6.y = d_src[(height + (i + 1)) * width + (j - 1)];
            rgb6.z = d_src[(height * 2 + (i + 1)) * width + (j - 1)];

            uchar3 rgb7; //(i+1)(j)
            rgb7.x = d_src[(i + 1) * width + j];
            rgb7.y = d_src[(height + (i + 1)) * width + j];
            rgb7.z = d_src[(height * 2 + (i + 1)) * width + j];

            uchar3 rgb8; //(i+1)(j+1)
            rgb8.x = d_src[(i + 1) * width + (j + 1)];
            rgb8.y = d_src[(height + (i + 1)) * width + (j + 1)];
            rgb8.z = d_src[(height * 2 + (i + 1)) * width + (j + 1)];

            //adding 16 outside points for 5x5 stencil in order to achieve greater blurr

            //top row:
            uchar3 rgb9; //(i-2)(j-2)
            rgb9.x = d_src[(i - 2) * width + (j - 2)];
            rgb9.y = d_src[(height + (i - 2)) * width + (j - 2)];
            rgb9.z = d_src[(height * 2 + (i - 2)) * width + (j - 2)];

            uchar3 rgb10; //(i-2)(j-1)
            rgb10.x = d_src[(i - 2) * width + (j - 1)];
            rgb10.y = d_src[(height + (i - 2)) * width + (j - 1)];
            rgb10.z = d_src[(height * 2 + (i - 2)) * width + (j - 1)];

            uchar3 rgb11; //(i-2)(j)
            rgb11.x = d_src[(i - 2) * width + (j)];
            rgb11.y = d_src[(height + (i - 2)) * width + (j)];
            rgb11.z = d_src[(height * 2 + (i - 2)) * width + (j)];

            uchar3 rgb12; //(i-2)(j+1)
            rgb12.x = d_src[(i - 2) * width + (j + 1)];
            rgb12.y = d_src[(height + (i - 2)) * width + (j + 1)];
            rgb12.z = d_src[(height * 2 + (i - 2)) * width + (j + 1)];

            uchar3 rgb13; //(i-2)(j+2)
            rgb13.x = d_src[(i - 2) * width + (j + 2)];
            rgb13.y = d_src[(height + (i - 2)) * width + (j + 2)];
            rgb13.z = d_src[(height * 2 + (i - 2)) * width + (j + 2)];

            //side left:
            uchar3 rgb14; //(i-1)(j-2)
            rgb14.x = d_src[(i - 1) * width + (j - 2)];
            rgb14.y = d_src[(height + (i - 1)) * width + (j - 2)];
            rgb14.z = d_src[(height * 2 + (i - 1)) * width + (j - 2)];

            uchar3 rgb15; //(i)(j-2)
            rgb15.x = d_src[(i)*width + (j - 2)];
            rgb15.y = d_src[(height + (i)) * width + (j - 2)];
            rgb15.z = d_src[(height * 2 + (i)) * width + (j - 2)];

            uchar3 rgb16; //(i+1)(j-2)
            rgb16.x = d_src[(i + 1) * width + (j - 2)];
            rgb16.y = d_src[(height + (i + 1)) * width + (j - 2)];
            rgb16.z = d_src[(height * 2 + (i + 1)) * width + (j - 2)];

            //side right:
            uchar3 rgb17; //(i-1)(j+2)
            rgb17.x = d_src[(i - 1) * width + (j + 2)];
            rgb17.y = d_src[(height + (i - 1)) * width + (j + 2)];
            rgb17.z = d_src[(height * 2 + (i - 1)) * width + (j + 2)];

            uchar3 rgb18; //(i)(j+2)
            rgb18.x = d_src[(i)*width + (j + 2)];
            rgb18.y = d_src[(height + (i)) * width + (j + 2)];
            rgb18.z = d_src[(height * 2 + (i)) * width + (j + 2)];

            uchar3 rgb19; //(i+1)(j+2)
            rgb19.x = d_src[(i + 1) * width + (j + 2)];
            rgb19.y = d_src[(height + (i + 1)) * width + (j + 2)];
            rgb19.z = d_src[(height * 2 + (i + 1)) * width + (j + 2)];

            //bottom row:
            uchar3 rgb20; //(i+2)(j-2)
            rgb20.x = d_src[(i + 2) * width + (j - 2)];
            rgb20.y = d_src[(height + (i + 2)) * width + (j - 2)];
            rgb20.z = d_src[(height * 2 + (i + 2)) * width + (j - 2)];

            uchar3 rgb21; //(i+2)(j-1)
            rgb21.x = d_src[(i + 2) * width + (j - 1)];
            rgb21.y = d_src[(height + (i + 2)) * width + (j - 1)];
            rgb21.z = d_src[(height * 2 + (i + 2)) * width + (j - 1)];

            uchar3 rgb22; //(i+2)(j)
            rgb22.x = d_src[(i + 2) * width + (j)];
            rgb22.y = d_src[(height + (i + 2)) * width + (j)];
            rgb22.z = d_src[(height * 2 + (i + 2)) * width + (j)];

            uchar3 rgb23; //(i+2)(j+1)
            rgb23.x = d_src[(i + 2) * width + (j + 1)];
            rgb23.y = d_src[(height + (i + 2)) * width + (j + 1)];
            rgb23.z = d_src[(height * 2 + (i + 2)) * width + (j + 1)];

            uchar3 rgb24; //(i+2)(j+2)
            rgb24.x = d_src[(i + 2) * width + (j + 2)];
            rgb24.y = d_src[(height + (i + 2)) * width + (j + 2)];
            rgb24.z = d_src[(height * 2 + (i + 2)) * width + (j + 2)];



            if (filter == 0) {
                //Gaussian blurr 5x5

                unsigned int blurr_rgbx = (unsigned int)(rgb.x * (0.140625f) + (rgb5.x + rgb4.x + rgb7.x + rgb2.x) * (0.09375f) + (rgb1.x + rgb3.x + rgb6.x + rgb8.x) * (0.0625f) + (rgb9.x + rgb13.x + rgb20.x + rgb24.x) * (0.00390625f) + (rgb10.x + rgb12.x + rgb14.x + rgb16.x + rgb17.x + rgb19.x + rgb21.x + rgb23.x) * (0.015625f) + (rgb11.x + rgb15.x + rgb18.x + rgb22.x) * (0.0234375f));
                unsigned int blurr_rgby = (unsigned int)(rgb.y * (0.140625f) + (rgb5.y + rgb4.y + rgb7.y + rgb2.y) * (0.09375f) + (rgb1.y + rgb3.y + rgb6.y + rgb8.y) * (0.0625f) + (rgb9.y + rgb13.y + rgb20.y + rgb24.y) * (0.00390625f) + (rgb10.y + rgb12.y + rgb14.y + rgb16.y + rgb17.y + rgb19.y + rgb21.y + rgb23.y) * (0.015625f) + (rgb11.y + rgb15.y + rgb18.y + rgb22.y) * (0.0234375f));
                unsigned int blurr_rgbz = (unsigned int)(rgb.z * (0.140625f) + (rgb5.z + rgb4.z + rgb7.z + rgb2.z) * (0.09375f) + (rgb1.z + rgb3.z + rgb6.z + rgb8.z) * (0.0625f) + (rgb9.z + rgb13.z + rgb20.z + rgb24.z) * (0.00390625f) + (rgb10.z + rgb12.z + rgb14.z + rgb16.z + rgb17.z + rgb19.z + rgb21.z + rgb23.z) * (0.015625f) + (rgb11.z + rgb15.z + rgb18.z + rgb22.z) * (0.0234375f));
                
                d_dst[i * width + j] = (unsigned char)(blurr_rgbx > 255 ? 255 : blurr_rgbx);
                d_dst[(height + i) * width + j] = (unsigned char)(blurr_rgby > 255 ? 255 : blurr_rgby);
                d_dst[(height * 2 + i) * width + j] = (unsigned char)(blurr_rgbz > 255 ? 255 : blurr_rgbz);
            
            }
            if (filter == 1) {
                //Increase brightness filter

                unsigned int blurr_rgbx = (unsigned int)(rgb.x * (0.04f) + (rgb5.x + rgb4.x + rgb7.x + rgb2.x) * (0.16f) + (rgb1.x + rgb3.x + rgb6.x + rgb8.x) * (0.16f) + (rgb9.x + rgb13.x + rgb20.x + rgb24.x) * (0.16f) + (rgb10.x + rgb12.x + rgb14.x + rgb16.x + rgb17.x + rgb19.x + rgb21.x + rgb23.x) * (0.32f) + (rgb11.x + rgb15.x + rgb18.x + rgb22.x) * (0.16f));
                unsigned int blurr_rgby = (unsigned int)(rgb.y * (0.04f) + (rgb5.y + rgb4.y + rgb7.y + rgb2.y) * (0.16f) + (rgb1.y + rgb3.y + rgb6.y + rgb8.y) * (0.16f) + (rgb9.y + rgb13.y + rgb20.y + rgb24.y) * (0.16f) + (rgb10.y + rgb12.y + rgb14.y + rgb16.y + rgb17.y + rgb19.y + rgb21.y + rgb23.y) * (0.32f) + (rgb11.y + rgb15.y + rgb18.y + rgb22.y) * (0.16f));
                unsigned int blurr_rgbz = (unsigned int)(rgb.z * (0.04f) + (rgb5.z + rgb4.z + rgb7.z + rgb2.z) * (0.16f) + (rgb1.z + rgb3.z + rgb6.z + rgb8.z) * (0.16f) + (rgb9.z + rgb13.z + rgb20.z + rgb24.z) * (0.16f) + (rgb10.z + rgb12.z + rgb14.z + rgb16.z + rgb17.z + rgb19.z + rgb21.z + rgb23.z) * (0.32f) + (rgb11.z + rgb15.z + rgb18.z + rgb22.z) * (0.16f));

                d_dst[i * width + j] = (unsigned char)(blurr_rgbx > 255 ? 255 : blurr_rgbx);
                d_dst[(height + i) * width + j] = (unsigned char)(blurr_rgby > 255 ? 255 : blurr_rgby);
                d_dst[(height * 2 + i) * width + j] = (unsigned char)(blurr_rgbz > 255 ? 255 : blurr_rgbz);
            }

            if (filter == 2) {
                //Averaging filter
                unsigned int blurr_rgbx = (unsigned int)(rgb.x * (0.04f) + (rgb5.x + rgb4.x + rgb7.x + rgb2.x) * (0.04f) + (rgb1.x + rgb3.x + rgb6.x + rgb8.x) * (0.04f) + (rgb9.x + rgb13.x + rgb20.x + rgb24.x) * (0.04f) + (rgb10.x + rgb12.x + rgb14.x + rgb16.x + rgb17.x + rgb19.x + rgb21.x + rgb23.x) * (0.04f) + (rgb11.x + rgb15.x + rgb18.x + rgb22.x) * (0.04f));
                unsigned int blurr_rgby = (unsigned int)(rgb.y * (0.04f) + (rgb5.y + rgb4.y + rgb7.y + rgb2.y) * (0.04f) + (rgb1.y + rgb3.y + rgb6.y + rgb8.y) * (0.04f) + (rgb9.y + rgb13.y + rgb20.y + rgb24.y) * (0.04f) + (rgb10.y + rgb12.y + rgb14.y + rgb16.y + rgb17.y + rgb19.y + rgb21.y + rgb23.y) * (0.04f) + (rgb11.y + rgb15.y + rgb18.y + rgb22.y) * (0.04f));
                unsigned int blurr_rgbz = (unsigned int)(rgb.z * (0.04f) + (rgb5.z + rgb4.z + rgb7.z + rgb2.z) * (0.04f) + (rgb1.z + rgb3.z + rgb6.z + rgb8.z) * (0.04f) + (rgb9.z + rgb13.z + rgb20.z + rgb24.z) * (0.04f) + (rgb10.z + rgb12.z + rgb14.z + rgb16.z + rgb17.z + rgb19.z + rgb21.z + rgb23.z) * (0.04f) + (rgb11.z + rgb15.z + rgb18.z + rgb22.z) * (0.04f));

                d_dst[i * width + j] = (unsigned char)(blurr_rgbx > 255 ? 255 : blurr_rgbx);
                d_dst[(height + i) * width + j] = (unsigned char)(blurr_rgby > 255 ? 255 : blurr_rgby);
                d_dst[(height * 2 + i) * width + j] = (unsigned char)(blurr_rgbz > 255 ? 255 : blurr_rgbz);
            }

        }
        else // set remaining pixels outside of wanted region to the same pixels
        {
        d_dst[i * width + j] = (unsigned char)d_src[i * width + j];
        d_dst[(height + i) * width + j] = (unsigned char)d_src[(height + i) * width + j];
        d_dst[(height * 2 + i) * width + j] = (unsigned char)d_src[(height * 2 + i) * width + j];
        }

    
}
__global__ void linearizeDataKernel(const unsigned char d_src[], unsigned int d_dst[], int width, int height, size_t frame_size, int channels) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    d_dst[i * width + j] = (unsigned int)d_src[channels * (width * i + j) + 0]; // blue 

    d_dst[(height + i) * width + j] = (unsigned int)d_src[channels * (width * i + j) + 1]; // green

    d_dst[(height * 2 + i) * width + j] = (unsigned int)d_src[channels * (width * i + j) + 2]; //red

}
__global__ void copyAndDisplayKernel(unsigned char d_dst[], const unsigned int d_src[], int width, int height, size_t frame_size, int channels) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    d_dst[channels * (width * i + j) + 0] = (unsigned char)d_src[i * width + j];
    //d_dst[i * width + j] = (unsigned int)d_src[channels * (width * i + j) + 0]; // blue 

    d_dst[channels * (width * i + j) + 1] = (unsigned char)d_src[(height + i) * width + j];
    //d_dst[(height + i) * width + j] = (unsigned int)d_src[channels * (width * i + j) + 1]; // green

    d_dst[channels * (width * i + j) + 2] = (unsigned char)d_src[(height * 2 + i) * width + j];
    //d_dst[(height * 2 + i) * width + j] = (unsigned int)d_src[channels * (width * i + j) + 2]; //red

}


void SerialGaussianFilter(const unsigned int d_src[], unsigned int d_dst[], int r_width, int r_height, int width, int height, int x, int y, int filter) {

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if ((i >= y && j >= x) && (i < y + r_height - 2 && j < x + r_width - 2) && (i > 2 && j > 2)) {
                //3x3 inside
                uchar3 rgb; // (i)(j)
                rgb.x = d_src[i * width + j];
                rgb.y = d_src[(height + i) * width + j];
                rgb.z = d_src[(height * 2 + i) * width + j];

                uchar3 rgb1; //(i-1)(j-1)
                rgb1.x = d_src[(i - 1) * width + (j - 1)];
                rgb1.y = d_src[(height + (i - 1)) * width + (j - 1)];
                rgb1.z = d_src[(height * 2 + (i - 1)) * width + (j - 1)];

                uchar3 rgb2; //(i-1)(j)
                rgb2.x = d_src[(i - 1) * width + j];
                rgb2.y = d_src[(height + (i - 1)) * width + j];
                rgb2.z = d_src[(height * 2 + (i - 1)) * width + j];

                uchar3 rgb3; //(i-1)(j+1)
                rgb3.x = d_src[(i - 1) * width + (j + 1)];
                rgb3.y = d_src[(height + (i - 1)) * width + (j + 1)];
                rgb3.z = d_src[(height * 2 + (i - 1)) * width + (j + 1)];

                uchar3 rgb4; //(i)(j-1)
                rgb4.x = d_src[(i)*width + (j - 1)];
                rgb4.y = d_src[(height + (i)) * width + (j - 1)];
                rgb4.z = d_src[(height * 2 + (i)) * width + (j - 1)];

                uchar3 rgb5; //(i)(j+1)
                rgb5.x = d_src[i * width + (j + 1)];
                rgb5.y = d_src[(height + i) * width + (j + 1)];
                rgb5.z = d_src[(height * 2 + i) * width + (j + 1)];

                uchar3 rgb6; //(i+1)(j-1)
                rgb6.x = d_src[(i + 1) * width + (j - 1)];
                rgb6.y = d_src[(height + (i + 1)) * width + (j - 1)];
                rgb6.z = d_src[(height * 2 + (i + 1)) * width + (j - 1)];

                uchar3 rgb7; //(i+1)(j)
                rgb7.x = d_src[(i + 1) * width + j];
                rgb7.y = d_src[(height + (i + 1)) * width + j];
                rgb7.z = d_src[(height * 2 + (i + 1)) * width + j];

                uchar3 rgb8; //(i+1)(j+1)
                rgb8.x = d_src[(i + 1) * width + (j + 1)];
                rgb8.y = d_src[(height + (i + 1)) * width + (j + 1)];
                rgb8.z = d_src[(height * 2 + (i + 1)) * width + (j + 1)];

                //adding 16 outside points for 5x5 stencil in order to achieve greater blurr

                //top row:
                uchar3 rgb9; //(i-2)(j-2)
                rgb9.x = d_src[(i - 2) * width + (j - 2)];
                rgb9.y = d_src[(height + (i - 2)) * width + (j - 2)];
                rgb9.z = d_src[(height * 2 + (i - 2)) * width + (j - 2)];

                uchar3 rgb10; //(i-2)(j-1)
                rgb10.x = d_src[(i - 2) * width + (j - 1)];
                rgb10.y = d_src[(height + (i - 2)) * width + (j - 1)];
                rgb10.z = d_src[(height * 2 + (i - 2)) * width + (j - 1)];

                uchar3 rgb11; //(i-2)(j)
                rgb11.x = d_src[(i - 2) * width + (j)];
                rgb11.y = d_src[(height + (i - 2)) * width + (j)];
                rgb11.z = d_src[(height * 2 + (i - 2)) * width + (j)];

                uchar3 rgb12; //(i-2)(j+1)
                rgb12.x = d_src[(i - 2) * width + (j + 1)];
                rgb12.y = d_src[(height + (i - 2)) * width + (j + 1)];
                rgb12.z = d_src[(height * 2 + (i - 2)) * width + (j + 1)];

                uchar3 rgb13; //(i-2)(j+2)
                rgb13.x = d_src[(i - 2) * width + (j + 2)];
                rgb13.y = d_src[(height + (i - 2)) * width + (j + 2)];
                rgb13.z = d_src[(height * 2 + (i - 2)) * width + (j + 2)];

                //side left:
                uchar3 rgb14; //(i-1)(j-2)
                rgb14.x = d_src[(i - 1) * width + (j - 2)];
                rgb14.y = d_src[(height + (i - 1)) * width + (j - 2)];
                rgb14.z = d_src[(height * 2 + (i - 1)) * width + (j - 2)];

                uchar3 rgb15; //(i)(j-2)
                rgb15.x = d_src[(i)*width + (j - 2)];
                rgb15.y = d_src[(height + (i)) * width + (j - 2)];
                rgb15.z = d_src[(height * 2 + (i)) * width + (j - 2)];

                uchar3 rgb16; //(i+1)(j-2)
                rgb16.x = d_src[(i + 1) * width + (j - 2)];
                rgb16.y = d_src[(height + (i + 1)) * width + (j - 2)];
                rgb16.z = d_src[(height * 2 + (i + 1)) * width + (j - 2)];

                //side right:
                uchar3 rgb17; //(i-1)(j+2)
                rgb17.x = d_src[(i - 1) * width + (j + 2)];
                rgb17.y = d_src[(height + (i - 1)) * width + (j + 2)];
                rgb17.z = d_src[(height * 2 + (i - 1)) * width + (j + 2)];

                uchar3 rgb18; //(i)(j+2)
                rgb18.x = d_src[(i)*width + (j + 2)];
                rgb18.y = d_src[(height + (i)) * width + (j + 2)];
                rgb18.z = d_src[(height * 2 + (i)) * width + (j + 2)];

                uchar3 rgb19; //(i+1)(j+2)
                rgb19.x = d_src[(i + 1) * width + (j + 2)];
                rgb19.y = d_src[(height + (i + 1)) * width + (j + 2)];
                rgb19.z = d_src[(height * 2 + (i + 1)) * width + (j + 2)];

                //bottom row:
                uchar3 rgb20; //(i+2)(j-2)
                rgb20.x = d_src[(i + 2) * width + (j - 2)];
                rgb20.y = d_src[(height + (i + 2)) * width + (j - 2)];
                rgb20.z = d_src[(height * 2 + (i + 2)) * width + (j - 2)];

                uchar3 rgb21; //(i+2)(j-1)
                rgb21.x = d_src[(i + 2) * width + (j - 1)];
                rgb21.y = d_src[(height + (i + 2)) * width + (j - 1)];
                rgb21.z = d_src[(height * 2 + (i + 2)) * width + (j - 1)];

                uchar3 rgb22; //(i+2)(j)
                rgb22.x = d_src[(i + 2) * width + (j)];
                rgb22.y = d_src[(height + (i + 2)) * width + (j)];
                rgb22.z = d_src[(height * 2 + (i + 2)) * width + (j)];

                uchar3 rgb23; //(i+2)(j+1)
                rgb23.x = d_src[(i + 2) * width + (j + 1)];
                rgb23.y = d_src[(height + (i + 2)) * width + (j + 1)];
                rgb23.z = d_src[(height * 2 + (i + 2)) * width + (j + 1)];

                uchar3 rgb24; //(i+2)(j+2)
                rgb24.x = d_src[(i + 2) * width + (j + 2)];
                rgb24.y = d_src[(height + (i + 2)) * width + (j + 2)];
                rgb24.z = d_src[(height * 2 + (i + 2)) * width + (j + 2)];



                if (filter == 0) {
                    //Gaussian blurr 5x5

                    unsigned int blurr_rgbx = (unsigned int)(rgb.x * (0.140625f) + (rgb5.x + rgb4.x + rgb7.x + rgb2.x) * (0.09375f) + (rgb1.x + rgb3.x + rgb6.x + rgb8.x) * (0.0625f) + (rgb9.x + rgb13.x + rgb20.x + rgb24.x) * (0.00390625f) + (rgb10.x + rgb12.x + rgb14.x + rgb16.x + rgb17.x + rgb19.x + rgb21.x + rgb23.x) * (0.015625f) + (rgb11.x + rgb15.x + rgb18.x + rgb22.x) * (0.0234375f));
                    unsigned int blurr_rgby = (unsigned int)(rgb.y * (0.140625f) + (rgb5.y + rgb4.y + rgb7.y + rgb2.y) * (0.09375f) + (rgb1.y + rgb3.y + rgb6.y + rgb8.y) * (0.0625f) + (rgb9.y + rgb13.y + rgb20.y + rgb24.y) * (0.00390625f) + (rgb10.y + rgb12.y + rgb14.y + rgb16.y + rgb17.y + rgb19.y + rgb21.y + rgb23.y) * (0.015625f) + (rgb11.y + rgb15.y + rgb18.y + rgb22.y) * (0.0234375f));
                    unsigned int blurr_rgbz = (unsigned int)(rgb.z * (0.140625f) + (rgb5.z + rgb4.z + rgb7.z + rgb2.z) * (0.09375f) + (rgb1.z + rgb3.z + rgb6.z + rgb8.z) * (0.0625f) + (rgb9.z + rgb13.z + rgb20.z + rgb24.z) * (0.00390625f) + (rgb10.z + rgb12.z + rgb14.z + rgb16.z + rgb17.z + rgb19.z + rgb21.z + rgb23.z) * (0.015625f) + (rgb11.z + rgb15.z + rgb18.z + rgb22.z) * (0.0234375f));

                    d_dst[i * width + j] = (unsigned char)(blurr_rgbx > 255 ? 255 : blurr_rgbx);
                    d_dst[(height + i) * width + j] = (unsigned char)(blurr_rgby > 255 ? 255 : blurr_rgby);
                    d_dst[(height * 2 + i) * width + j] = (unsigned char)(blurr_rgbz > 255 ? 255 : blurr_rgbz);

                }
                if (filter == 1) {
                    //Increase brightness filter

                    unsigned int blurr_rgbx = (unsigned int)(rgb.x * (0.04f) + (rgb5.x + rgb4.x + rgb7.x + rgb2.x) * (0.16f) + (rgb1.x + rgb3.x + rgb6.x + rgb8.x) * (0.16f) + (rgb9.x + rgb13.x + rgb20.x + rgb24.x) * (0.16f) + (rgb10.x + rgb12.x + rgb14.x + rgb16.x + rgb17.x + rgb19.x + rgb21.x + rgb23.x) * (0.32f) + (rgb11.x + rgb15.x + rgb18.x + rgb22.x) * (0.16f));
                    unsigned int blurr_rgby = (unsigned int)(rgb.y * (0.04f) + (rgb5.y + rgb4.y + rgb7.y + rgb2.y) * (0.16f) + (rgb1.y + rgb3.y + rgb6.y + rgb8.y) * (0.16f) + (rgb9.y + rgb13.y + rgb20.y + rgb24.y) * (0.16f) + (rgb10.y + rgb12.y + rgb14.y + rgb16.y + rgb17.y + rgb19.y + rgb21.y + rgb23.y) * (0.32f) + (rgb11.y + rgb15.y + rgb18.y + rgb22.y) * (0.16f));
                    unsigned int blurr_rgbz = (unsigned int)(rgb.z * (0.04f) + (rgb5.z + rgb4.z + rgb7.z + rgb2.z) * (0.16f) + (rgb1.z + rgb3.z + rgb6.z + rgb8.z) * (0.16f) + (rgb9.z + rgb13.z + rgb20.z + rgb24.z) * (0.16f) + (rgb10.z + rgb12.z + rgb14.z + rgb16.z + rgb17.z + rgb19.z + rgb21.z + rgb23.z) * (0.32f) + (rgb11.z + rgb15.z + rgb18.z + rgb22.z) * (0.16f));

                    d_dst[i * width + j] = (unsigned char)(blurr_rgbx > 255 ? 255 : blurr_rgbx);
                    d_dst[(height + i) * width + j] = (unsigned char)(blurr_rgby > 255 ? 255 : blurr_rgby);
                    d_dst[(height * 2 + i) * width + j] = (unsigned char)(blurr_rgbz > 255 ? 255 : blurr_rgbz);
                }

                if (filter == 2) {
                    //Averaging filter
                    unsigned int blurr_rgbx = (unsigned int)(rgb.x * (0.04f) + (rgb5.x + rgb4.x + rgb7.x + rgb2.x) * (0.04f) + (rgb1.x + rgb3.x + rgb6.x + rgb8.x) * (0.04f) + (rgb9.x + rgb13.x + rgb20.x + rgb24.x) * (0.04f) + (rgb10.x + rgb12.x + rgb14.x + rgb16.x + rgb17.x + rgb19.x + rgb21.x + rgb23.x) * (0.04f) + (rgb11.x + rgb15.x + rgb18.x + rgb22.x) * (0.04f));
                    unsigned int blurr_rgby = (unsigned int)(rgb.y * (0.04f) + (rgb5.y + rgb4.y + rgb7.y + rgb2.y) * (0.04f) + (rgb1.y + rgb3.y + rgb6.y + rgb8.y) * (0.04f) + (rgb9.y + rgb13.y + rgb20.y + rgb24.y) * (0.04f) + (rgb10.y + rgb12.y + rgb14.y + rgb16.y + rgb17.y + rgb19.y + rgb21.y + rgb23.y) * (0.04f) + (rgb11.y + rgb15.y + rgb18.y + rgb22.y) * (0.04f));
                    unsigned int blurr_rgbz = (unsigned int)(rgb.z * (0.04f) + (rgb5.z + rgb4.z + rgb7.z + rgb2.z) * (0.04f) + (rgb1.z + rgb3.z + rgb6.z + rgb8.z) * (0.04f) + (rgb9.z + rgb13.z + rgb20.z + rgb24.z) * (0.04f) + (rgb10.z + rgb12.z + rgb14.z + rgb16.z + rgb17.z + rgb19.z + rgb21.z + rgb23.z) * (0.04f) + (rgb11.z + rgb15.z + rgb18.z + rgb22.z) * (0.04f));

                    d_dst[i * width + j] = (unsigned char)(blurr_rgbx > 255 ? 255 : blurr_rgbx);
                    d_dst[(height + i) * width + j] = (unsigned char)(blurr_rgby > 255 ? 255 : blurr_rgby);
                    d_dst[(height * 2 + i) * width + j] = (unsigned char)(blurr_rgbz > 255 ? 255 : blurr_rgbz);
                }

            }
            else // set remaining pixels outside of wanted region to the same pixels
            {
                d_dst[i * width + j] = (unsigned char)d_src[i * width + j];
                d_dst[(height + i) * width + j] = (unsigned char)d_src[(height + i) * width + j];
                d_dst[(height * 2 + i) * width + j] = (unsigned char)d_src[(height * 2 + i) * width + j];
            }
        }
    }
}

void linearizeData(unsigned int destination[], unsigned char linear_data[],unsigned char data[], int width, int height, size_t frame_size, int channels, int type) {

    if (type == 0) {

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {

                destination[i * width + j] = (unsigned int)data[channels * (width * i + j) + 0]; // blue (gives warning:Buffer overrun while writing to 'linear_data': the writable size is 'frame_size*1' bytes, but '2' bytes may be written.)

                destination[(height + i) * width + j] = (unsigned int)data[channels * (width * i + j) + 1]; // green

                destination[(height * 2 + i) * width + j] = (unsigned int)data[channels * (width * i + j) + 2]; //red

            }
        }
    }

    if (type == 1) {
        unsigned char* d_src;
        unsigned int* d_gs;

        cudaMalloc((void**)&d_src, width * height * 3 * sizeof(unsigned char));
        cudaMalloc((void**)&d_gs, frame_size);

        cudaMemcpy(d_src, data, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

        dim3 blockDim(16, 16, 1);
        dim3 gridDim((width + 15) / 16, (height + 15) / 16, 1);

        linearizeDataKernel << <gridDim, blockDim, 1 >> > (d_src, d_gs, width, height, frame_size, channels);
        cudaDeviceSynchronize();

        cudaMemcpy(destination, d_gs, frame_size, cudaMemcpyDeviceToHost);

        cudaFree(d_src);
        cudaFree(d_gs);

        cudaDeviceReset;
    }




}


void copyAndDisplay(unsigned int img_data[], unsigned char linear_data[], Mat& img, int width, int height, size_t frame_size, int channels, int type) {

    if (type == 0) {

        //copy pixel data back onto Mat image
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                img.data[img.channels() * (width * i + j) + 0] = img_data[i * width + j];
                img.data[img.channels() * (width * i + j) + 1] = img_data[(height + i) * width + j];
                img.data[img.channels() * (width * i + j) + 2] = img_data[(height * 2 + i) * width + j];
            }
        }

        imshow("Face Detection", img);
    } 

    if (type == 1) {
        unsigned char* d_gs;
        unsigned int* d_src;

        cudaMalloc((void**)&d_gs, width * height * 3 * sizeof(unsigned char));
        cudaMalloc((void**)&d_src, frame_size);

        cudaMemcpy(d_src, img_data, frame_size, cudaMemcpyHostToDevice);

        dim3 blockDim(16, 16, 1);
        dim3 gridDim((width + 15) / 16, (height + 15) / 16, 1);

        copyAndDisplayKernel << <gridDim, blockDim, 1 >> > (d_gs, d_src, width, height, frame_size, channels);
        cudaDeviceSynchronize();

        cudaMemcpy(img.data, d_gs, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        cudaFree(d_src);
        cudaFree(d_gs);

        cudaDeviceReset;

        imshow("Face Detection", img);
    }

}


void FaceDetect(unsigned int frame[], Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale, int frame_width, int frame_height, size_t frame_size, int type, int filter)
{

    vector<Rect> faces;
    Mat gray, smallImg;

    cvtColor(img, gray, COLOR_BGR2GRAY); // Convert to Gray Scale 
    double fx = 1 / scale;

    // Resize the Grayscale Image  
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
    equalizeHist(smallImg, smallImg);

    // Detect faces of different sizes using cascade classifier  
    cascade.detectMultiScale(smallImg, faces, 1.1,
        2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

   

    // Apply filters from faces detected
    for (size_t i = 0; i < faces.size(); i++)
    {
        Rect r = faces[i];

        if (type == 1) {
            unsigned int* d_src;
            unsigned int* d_gs;
            cudaMalloc((void**)&d_src, frame_size);
            cudaMalloc((void**)&d_gs, frame_size);

            dim3 blockDim(16, 16, 1);
            dim3 gridDim((frame_width + 15) / 16, (frame_height + 15) / 16, 1);

            cudaMemcpy(d_src, frame, frame_size, cudaMemcpyHostToDevice);
            CudaGaussianFilter << < gridDim, blockDim, 1 >> > (d_src, d_gs, r.width, r.height, frame_width, frame_height, r.x, r.y, filter);
            cudaDeviceSynchronize();

            cudaMemcpy(frame, d_gs, frame_size, cudaMemcpyDeviceToHost);

            cudaDeviceReset;
            cudaFree(d_src);
            cudaFree(d_gs);
        }

        if (type == 0) {
            unsigned int* temp;
            temp = new unsigned int[frame_size];
            temp = frame;
            SerialGaussianFilter(temp, frame, r.width, r.height, frame_width, frame_height, r.x, r.y, filter);
        }
    }
 

}

void timeRoutine(int type, int maxIter, int filter) {
    // VideoCapture class for playing video for which faces to be detected 
    VideoCapture capture;
    Mat frame;

    // PreDefined trained XML classifiers with facial features 
    CascadeClassifier cascade, nestedCascade;
    double scale = 1;

    // Load classifiers from "opencv/data/haarcascades" directory  
    nestedCascade.load("C:/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml");

    // Change path before execution  
    cascade.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalcatface.xml");

    // Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video 
    capture.open("C:/Users/antho/Downloads/wash.mp4");

    int width = capture.get(CV_CAP_PROP_FRAME_WIDTH); // video width
    int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);// video height

    //  Size of img data
    size_t frame_size = width * height * 3 * sizeof(unsigned int);

    //  Size of old data
    size_t old_data_size = width * height * 3 * sizeof(unsigned char);

    //  Pointers used to store data at different stages
    unsigned char* old_data;
    unsigned int* img_data;

    //  Memory Allocation
    old_data = new unsigned char[old_data_size];
    img_data = new unsigned int[frame_size];

    int iter = 0; // used for testing iterations

    if (type == 1) {
        cudaError_t cudaStatus;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        //begin Cuda face filter...
        while (iter < maxIter)
        {
            capture >> frame; // gets the succeeding frame for every loop iteration

            Mat src = frame.clone(); // copies the frame into a Mat datatype

            linearizeData(img_data, old_data, src.data, width, height, frame_size, src.channels(), type);

            FaceDetect(img_data, src, cascade, nestedCascade, scale, width, height, frame_size, type, filter);

            copyAndDisplay(img_data, old_data, src, width, height, frame_size, src.channels(), type);

            char c = (char)waitKey(1);
            //Press q to exit from window 
            if (c == 27 || c == 'q' || c == 'Q')
                break;
            iter++;
        }
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        cout << endl << "Parallel run took: " << milliseconds << "miliseconds" << endl;
    }

    if (type == 0) {

        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        while (iter < maxIter)
        {
            capture >> frame; // gets the succeeding frame for every loop iteration

            Mat src = frame.clone(); // copies the frame into a Mat datatype

            linearizeData(img_data, old_data, src.data, width, height, frame_size, src.channels(), type);

            FaceDetect(img_data, src, cascade, nestedCascade, scale, width, height, frame_size, type, filter);

            copyAndDisplay(img_data, old_data, src, width, height, frame_size, src.channels(), type);

            char c = (char)waitKey(1);
            //Press q to exit from window 
            if (c == 27 || c == 'q' || c == 'Q')
                break;
            iter++;
        }

        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

        std::cout << "Sequential run took: " << time_span.count() * 1000 << " miliseconds.";
        std::cout << std::endl;
    }

}

int main(int argc, const char** argv)
{
    //  Set type to: 1) 0 for sequential or 2) 1 for parallel
    int type = 1;

    //  Set type to: 1) 0 for Gaussian Blurr filter  2) 1 for bright filter  3) 2 for average smoothing filter
    int filter = 1;

    //  Set the max number of frames to be grabbed
    int maxIter = 50;

    timeRoutine(type,maxIter,filter);

    return 0;

}