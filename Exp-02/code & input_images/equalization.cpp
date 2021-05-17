#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <stdlib.h>
#include <cmath>


using namespace std;
using namespace cv;

#define LEVEL 256
#define INTENSITY_MAX 255
#define INTENSITY_MIN 0



// This function takes image object, histogram and size
// Function reads the image and creates a histogram.
void imageTopdf(Mat image, float histogram[], int size) {

    for (int i = 0; i < LEVEL; i++) {
        histogram[i] = 0;
    }

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            histogram[(int)image.at<uchar>(y, x)]++;
        }
    }

	//dividing by size to make value between 0 and 1
    for (int i = 0; i < LEVEL; i++) {
        histogram[i] = histogram[i] / size;
    }

    return;

}

// This fucnction is used to calculate the transfer function of a given
// histogram. The transfer function created is just the cumulative frequency distribution
void get_tf(float histogram[], float tranFunc[]) {

    tranFunc[0] = histogram[0];
    for (int i = 1; i < LEVEL; i++) {
        tranFunc[i] = histogram[i] + tranFunc[i - 1];

    }
    return;
}

// This function is used to map the histogram to the intensity values that will be displayed on the image
void intensityMapping(float tranFunc[], float histogram[]) {
    
    float tranFuncMin = INTENSITY_MAX + 1;
    for (int i = 0; i < LEVEL; i++) {
        if (tranFuncMin > tranFunc[i]) {
            tranFuncMin = tranFunc[i];
        }
    }

    for (int i = 0; i < LEVEL; i++) {
        histogram[i] = (((tranFunc[i] - tranFuncMin) / (1 - tranFuncMin)) * (LEVEL - 1) + tranFuncMin);
    	//histogram[i] = (tranFunc[i] * (LEVEL - 1));
    }

    return;
}


//Function to convert an Red Grenn Blue(RGB) space to Hue Saturation Intensity(HSI) space
void convert_RGB_To_HSI(Mat inputImage, Mat inputImageHSI, float** H, float** S, float** I) {
    double r, g, b, h, s, in;
	//formula based conversion
    for (int i = 0; i < inputImage.rows; i++)
    {
        for (int j = 0; j < inputImage.cols; j++)
        {

            b = inputImage.at<Vec3b>(i, j)[0];
            g = inputImage.at<Vec3b>(i, j)[1];
            r = inputImage.at<Vec3b>(i, j)[2];



            float min_val = 0.0;
            min_val = min(r, min(b, g));
            s = 1 - 3 * (min_val / (b + g + r));

            in = (b + g + r) / 3; // TO SEE

            if (s < 0.00001)
            {
                s = 0.0;
            }
            else if (s > 0.99999) {
                s = 1.0;
            }

            if (s != 0.0)
            {
                h = 0.5 * ((r - g) + (r - b)) / sqrt(((r - g) * (r - g)) + ((r - b) * (g - b)));
                h = acos(h);

                if (b <= g)
                {
                    h = h;
                }
                else {
                    h = ((360 * 3.14159265) / 180.0) - h;
                }
            }
            else {
                h = 0.0;
            }

            inputImageHSI.at<Vec3b>(i, j)[0] = H[i][j] = (h * 180) / 3.14159265;
            inputImageHSI.at<Vec3b>(i, j)[1] = S[i][j] = s * 100;
            inputImageHSI.at<Vec3b>(i, j)[2] = I[i][j] = in;

        }
    }
    return;
}

//Function to convert an  Hue Saturation Intensity(HSI) space to Red Grenn Blue(RGB) space
void convert_HSI_To_RGB(Mat outputImage, Mat inputImageHSI, float** H, float** S, float** I) {
    float r, g, b, h, s, in;
	//formula based conversion
    for (int i = 0; i < inputImageHSI.rows; i++) {
        for (int j = 0; j < inputImageHSI.cols; j++) {

            h = H[i][j];
            s = S[i][j] / 100;
            in = I[i][j];

            if (h >= 0.0 && h < 120.0) {
                b = in * (1 - s);
                r = in * (1 + (s * cos(h * 3.14159265 / 180.0) / cos((60 - h) * 3.14159265 / 180.0)));
                g = 3 * in - (r + b);
            }
            else if (h >= 120.0 && h < 240.0) {
                h = h - 120;
                r = in * (1 - s);
                g = in * (1 + (s * cos(h * 3.14159265 / 180.0) / cos((60 - h) * 3.14159265 / 180.0)));
                b = 3 * in - (r + g);
            }
            else {
                h = h - 240;
                g = in * (1 - s);
                b = in * (1 + (s * cos(h * 3.14159265 / 180.0) / cos((60 - h) * 3.14159265 / 180.0)));
                r = 3 * in - (g + b);
            }

            if (b < INTENSITY_MIN)
                b = INTENSITY_MIN;
            if (b > INTENSITY_MAX)
                b = INTENSITY_MAX;

            if (g < INTENSITY_MIN)
                g = INTENSITY_MIN;
            if (g > INTENSITY_MAX)
                g = INTENSITY_MAX;

            if (r < INTENSITY_MIN)
                r = INTENSITY_MIN;
            if (r > INTENSITY_MAX)
                r = INTENSITY_MAX;

            outputImage.at<Vec3b>(i, j)[0] = round(b);
            outputImage.at<Vec3b>(i, j)[1] = round(g);
            outputImage.at<Vec3b>(i, j)[2] = round(r);

        }
    }
    return;
}


//Functio to create histogram from intensity value calculated from the RGB to HSI function
void intensityToHistogram(float** I, float histogram[], int rows, int cols) {

    for (int i = 0; i < LEVEL; i++) {
        histogram[i] = 0;
    }

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            histogram[(int)I[y][x]]++;
        }
    }

    for (int i = 0; i < LEVEL; i++) {
        histogram[i] = histogram[i] / (rows * cols);
    }

    return;

}

//Function to display histogram of an image and to write the historam in the outout file
void showHistogram(Mat& image, string fileName) {
    int bins = 256;             // number of bins
    int nc = image.channels();    // number of channels
    vector<Mat> histogram(nc);       // array for storing the histograms
    vector<Mat> canvas(nc);     // images for displaying the histogram
    int hmax[3] = { 0,0,0 };      // peak value for each histogram
	string name = "./output/histo_";     //for saving

    // The rest of the code will be placed here
    for (int i = 0; i < histogram.size(); i++)
        histogram[i] = Mat::zeros(1, bins, CV_32SC1);
	
	//histogram calculation
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < nc; k++) {
                uchar val = nc == 1 ? image.at<uchar>(i, j) : image.at<Vec3b>(i, j)[k];
                histogram[k].at<int>(val) += 1;
            }
        }
    }

	//calculation maximum value of each histogram
    for (int i = 0; i < nc; i++) {
        for (int j = 0; j < bins - 1; j++)
            hmax[i] = histogram[i].at<int>(j) > hmax[i] ? histogram[i].at<int>(j) : hmax[i];
    }

	//names
    const char* wname[3] = { "Blue", "Green", "Red" };
    Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };

	//drawing the lines
    for (int i = 0; i < nc; i++) {
        canvas[i] = Mat::ones(125, bins, CV_8UC3);

        for (int j = 0, rows = canvas[i].rows; j < bins - 1; j++) {
            line(
                canvas[i],
                Point(j, rows),
                Point(j, rows - (histogram[i].at<int>(j) * rows / hmax[i])),
                nc == 1 ? Scalar(255, 255, 255) : colors[i],
                1, 8, 0
            );
        }

		imwrite(nc == 1 ? name + fileName : name + wname[i] + fileName, canvas[i]);
        imshow(nc == 1 ? fileName : wname[i] + fileName, canvas[i]);

    }
}


int main() {
 
	cout << " HISTOGRAM EQUALIZATION " << endl;
	string fileName, EqualizedImageFileName;    //inputs

	cout << "Enter the inputImage FileName" << endl;
	cin >> fileName;

	cout << "Enter the fileName for Equalised Image" << endl;
	cin >> EqualizedImageFileName;
	
	// Mat image object
	Mat inputImage = imread(fileName, IMREAD_UNCHANGED);
	// displaying if input is not proper
	if (inputImage.empty()) {
	    cerr << "Error: Loading image" << endl;
	    return -1;
	}

	//if image is grayscale
	if (inputImage.channels() == 1) {
	
	    int size = inputImage.rows * inputImage.cols;
	    
	    //input histogram
	    float histogram[LEVEL];
	    imageTopdf(inputImage, histogram, size);
		
		//cummative histogram
	    float tranFunc[LEVEL];
	    get_tf(histogram, tranFunc);

		//intensity mapping
	    float outHistogram[LEVEL];
	    intensityMapping(tranFunc, outHistogram);

		//new image object
	    Mat outputImage = inputImage.clone();
	    for (int y = 0; y < inputImage.rows; y++) {
		for (int x = 0; x < inputImage.cols; x++)
		    outputImage.at<uchar>(y, x) = saturate_cast<uchar>(outHistogram[inputImage.at<uchar>(y, x)]); //rounds values to nearest integer
	    }

		imwrite("./output/"+fileName, inputImage);
	    imshow("Input Image", inputImage);
	    showHistogram(inputImage, fileName);

	    imwrite("./output/"+EqualizedImageFileName, outputImage);
	    imshow("Histogram Equalized Image", outputImage);
	    showHistogram(outputImage, EqualizedImageFileName);

	    waitKey();
	}
	//similarly for color images
	if (inputImage.channels() == 3) {
	    Mat inputImageHSI(inputImage.rows, inputImage.cols, inputImage.type());
		//convert to HSI format
	    float** H = new float* [inputImage.rows];
	    float** S = new float* [inputImage.rows];
	    float** I = new float* [inputImage.rows];
	    for (int i = 0; i < inputImage.rows; i++) {
		H[i] = new float[inputImage.cols];
		S[i] = new float[inputImage.cols];
		I[i] = new float[inputImage.cols];
	    }

	    convert_RGB_To_HSI(inputImage, inputImageHSI, H, S, I);

	    float histogram[LEVEL];
	    intensityToHistogram(I, histogram, inputImage.rows, inputImage.cols);
	    
	    float tranFunc[LEVEL];
	    get_tf(histogram, tranFunc);

		//outHistogram is the mapping
	    float outHistogram[LEVEL];
	    intensityMapping(tranFunc, outHistogram);


	    float** outI = new float* [inputImage.rows];
	    for (int i = 0; i < inputImage.rows; i++) {
		outI[i] = new float[inputImage.cols];
	    }

	    for (int i = 0; i < inputImage.rows; i++) {
		for (int j = 0; j < inputImage.cols; j++) {
		    outI[i][j] = (int)outHistogram[(int)I[i][j]];
		}
	    }

	    Mat outputImage(inputImage.rows, inputImage.cols, inputImage.type());
	    convert_HSI_To_RGB(outputImage, inputImageHSI, H, S, outI);

	    namedWindow("Original Image", WINDOW_AUTOSIZE);
	    imwrite("./output/"+fileName, inputImage);
	    imshow("Original Image", inputImage);
	    showHistogram(inputImage, fileName);

	    namedWindow("RGB Histogram Equalized Image", WINDOW_AUTOSIZE);
	    imwrite("./output/"+EqualizedImageFileName, outputImage);
	    imshow("RGB Histogram Equalized Image", outputImage);
	    showHistogram(outputImage,EqualizedImageFileName);

	    waitKey();
	}


    waitKey();
    return 0;
}

