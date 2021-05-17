#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <math.h>


using namespace cv;
using namespace std;


//defining structing elements
Mat kernel_0 = (Mat_<int>(3, 3) << 0, 0, 0, 0, 1, 1, 0, 0, 0);
Mat kernel_1 = (Mat_<int>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
Mat kernel_2 = (Mat_<int>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);


// Global variables
Mat img_binary;
Mat output;
int morph_type = 0; // choose morphological operation
int kernel_type = 0;
string trackbar_name_morph_type = "Type: \n 0: Erode \n 1: Dilate \n 2: Open \n 3: Close \n";
string trackbar_name_kernel_type = "Type: \n 0:  (1x2)  \n 1:  (3x3)  \n 2:  (3x3)  \n 3:  (9x9)  \n 4: (15x15) \n";
string window_name = "Original Image";

//for saving the output images
string morph[4] = {"Erode","Dilate","Open","Close"};
string kernel[5] = {"12","33N4","33N8","99","1515"}; 


//function for eroding
Mat erode(Mat src, Mat kernel, int iterations)
{
    Mat dst = src.clone();
    Mat src_copy = src.clone();
    int val = 0;

	//for 9*9 and 15*15
    for(int iter=0; iter<iterations; iter++)
    {
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
            	// if current pixel is zero then it is zero
                if (src_copy.at<uchar>(i, j) == 255)
                {
					//if current pixel is one then it is zero if any one of its structring element has zero
                    if (i > 0 && j > 0 && kernel.at<uchar>(0, 0) == 1 && src_copy.at<uchar>(i - 1, j - 1) != 255)
                        dst.at<uchar>(i, j) = val;
                    if (i > 0 && kernel.at<uchar>(0, 1) == 1 && src_copy.at<uchar>(i - 1, j) != 255)
                        dst.at<uchar>(i, j) = val;
                    if (i > 0 && (j + 1) < src.cols && kernel.at<uchar>(0, 2) == 1 && src_copy.at<uchar>(i - 1, j + 1) != 255)
                        dst.at<uchar>(i, j) = val;
                    if (j > 0 && kernel.at<uchar>(1, 0) == 1 && src_copy.at<uchar>(i, j - 1) != 255)
                        dst.at<uchar>(i, j) = val;
                    if ((j + 1) < src.cols && kernel.at<uchar>(1, 2) == 1 && src_copy.at<uchar>(i, j + 1) != 255)
                        dst.at<uchar>(i, j) = val;
                    if ((i + 1) < src.rows && j > 0 && kernel.at<uchar>(2, 0) == 1 && src_copy.at<uchar>(i + 1, j - 1) != 255)
                        dst.at<uchar>(i, j) = val;
                    if ((i + 1) < src.rows && kernel.at<uchar>(2, 1) == 1 && src_copy.at<uchar>(i + 1, j) != 255)
                        dst.at<uchar>(i, j) = val;
                    if ((i + 1) < src.rows && (j + 1) < src.cols && kernel.at<uchar>(2, 2) == 1 && src_copy.at<uchar>(i + 1, j + 1) != 255)
                        dst.at<uchar>(i, j) = val;
                }
            }
        }
        src_copy = dst.clone();
    }
    return dst;
}

Mat dilate(Mat src, Mat kernel, int iterations)
{
    Mat dst = src.clone();
    Mat src_copy = src.clone();
    int val = 255;

    for (int iter = 0; iter < iterations; iter++)
    {
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
            	//if current pixel is one then it is one
                if (src_copy.at<uchar>(i, j) == 0)
                {
                    // if current pixel is zero then it is one if any of the elements of structing element is one

                    if (i > 0 && j > 0 && kernel.at<uchar>(2, 2) == 1 && src_copy.at<uchar>(i - 1, j - 1) != 0)
                        dst.at<uchar>(i, j) = val;
                    if (i > 0 && kernel.at<uchar>(2, 1) == 1 && src_copy.at<uchar>(i - 1, j) != 0)
                        dst.at<uchar>(i, j) = val;
                    if (i > 0 && (j + 1) < src.cols && kernel.at<uchar>(2, 0) == 1 && src_copy.at<uchar>(i - 1, j + 1) != 0)
                        dst.at<uchar>(i, j) = val;
                    if (j > 0 && kernel.at<uchar>(1, 2) == 1 && src_copy.at<uchar>(i, j - 1) != 0)
                        dst.at<uchar>(i, j) = val;
                    if ((j + 1) < src.cols && kernel.at<uchar>(1, 0) == 1 && src_copy.at<uchar>(i, j + 1) != 0)
                        dst.at<uchar>(i, j) = val;
                    if ((i + 1) < src.rows && j > 0 && kernel.at<uchar>(0, 2) == 1 && src_copy.at<uchar>(i + 1, j - 1) != 0)
                        dst.at<uchar>(i, j) = val;
                    if ((i + 1) < src.rows && kernel.at<uchar>(0, 1) == 1 && src_copy.at<uchar>(i + 1, j) != 0)
                        dst.at<uchar>(i, j) = val;
                    if ((i + 1) < src.rows && (j + 1) < src.cols && kernel.at<uchar>(0, 0) == 1 && src_copy.at<uchar>(i + 1, j + 1) != 0)
                        dst.at<uchar>(i, j) = val;
                }
            }
        }
        src_copy = dst.clone();
    }
    return dst;
}


//called when there is a change
void on_trackbar(int, void *)
{   
    int iterations = 1;
    Mat kernel;
    if(kernel_type == 0) kernel = kernel_0;
    else if(kernel_type == 1) kernel = kernel_1;
    else if(kernel_type == 2) kernel = kernel_2;
    else if(kernel_type == 3)
    {
        kernel = kernel_2;
        //9*9
        iterations = 4;
    }
    else if(kernel_type == 4)
    {
    	//15*15
        kernel = kernel_2;
        iterations = 7;
    }

	//selecting morph type
    if(morph_type == 0) output = erode(img_binary, kernel, iterations);
    else if(morph_type == 1) output = dilate(img_binary, kernel, iterations);
    else if(morph_type == 2)
    {   
        Mat temp;
        // opening is erosion followed by dilation
        temp = erode(img_binary, kernel, iterations);
        output = dilate(temp, kernel, iterations);
    }
    else if (morph_type == 3)
    {   
        Mat temp;
        // closing is dilation followed by erosion
        temp = dilate(img_binary, kernel, iterations);
        output = erode(temp, kernel, iterations);
    }
}



int main(int argc, char **argv)
{

    Mat img_grayscale = imread("ricegrains.bmp", IMREAD_GRAYSCALE);
    
    //thresholding
    threshold(img_grayscale, img_binary, 127, 255, 0); // 0 is for THRESH_BINARY, 127 threshold value, 255 maxVal
    output = img_binary.clone();

	//adding track bar
    namedWindow(window_name, 1);
    createTrackbar(trackbar_name_morph_type, window_name, &morph_type, 3, on_trackbar);
    createTrackbar(trackbar_name_kernel_type, window_name, &kernel_type, 4, on_trackbar);

	//press esc to stop
    while (waitKey(10) != 27)
    {
        imshow("Output", output);
        imwrite("./output/"+morph[morph_type]+kernel[kernel_type]+".jpg", output);
        imshow(window_name, img_grayscale);
    }
    
    return 0;
}
