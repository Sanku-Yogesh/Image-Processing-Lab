//almost same functions used in equalization so not commented


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

    for (int i = 0; i < LEVEL; i++) {
        histogram[i] = histogram[i] / size;
    }

    return;

}

// This fucnction is used to calculate the transfer function of from a given
// histogram. The transfer function created is just the cumulative frequency distribution
void get_tf(float histogram[], float tranFunc[]) {

    tranFunc[0] = histogram[0];
    for (int i = 1; i < LEVEL; i++) {
        tranFunc[i] = histogram[i] + tranFunc[i - 1];

    }
    return;
}

// THis function is used to map the histogram to the intensity values that will be displayed on the image
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

//Function to match histogram of the input image to the target image
void histogramMatching(float inputTranFunc[], float targetTranFunc[], float histogram[], float targetHistogram[]) {
	/*
    for (int i = 0; i < LEVEL; i++) {
        int j = 0;
        do {
            histogram[i] = j;
            j++;
        } while (inputTranFunc[i] > targetTranFunc[j]);
    }*/
    float mindiff = 10000.00;
    for(int i=0,j=0;i<LEVEL;i++){
    	mindiff = 10000.00;
        while(j<LEVEL){
	    	if(mindiff >= abs(targetTranFunc[j]-inputTranFunc[i])){
	    		mindiff = abs(targetTranFunc[j]-inputTranFunc[i]);
	    		j=j+1;
	    	}
	    	else{
	    		histogram[i] = --j;
	    		break;
	    	}
	    }
		if(j>=LEVEL && i<LEVEL) histogram[i++] = LEVEL-1;
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

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < nc; k++) {
                uchar val = nc == 1 ? image.at<uchar>(i, j) : image.at<Vec3b>(i, j)[k];
                histogram[k].at<int>(val) += 1;
            }
        }
    }

    for (int i = 0; i < nc; i++) {
        for (int j = 0; j < bins - 1; j++)
            hmax[i] = histogram[i].at<int>(j) > hmax[i] ? histogram[i].at<int>(j) : hmax[i];
    }

    const char* wname[3] = { "Blue", "Green", "Red" };
    Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };

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
		imwrite(nc == 1 ? "./output/" + fileName : name + wname[i] + fileName, canvas[i]);
        imshow(nc == 1 ? fileName : wname[i] + fileName, canvas[i]);

    }
}



int main() {

	string inputFileName, targetFileName, MatchedImageFileName;
	cout << "Enter the inputImage FileName" << endl;
	cin >> inputFileName;
	Mat inputImage = imread(inputFileName, IMREAD_UNCHANGED );
	if (inputImage.empty()) {
	    cerr << "Error: Loading image" << endl;
	    //_getch();
	    return -1;
	}

	cout << "Enter the targetImage FileName" << endl;
	cin >> targetFileName;
	Mat targetImage = imread(targetFileName, IMREAD_UNCHANGED );
	if (targetImage.empty()) {
	    cerr << "Error: Loading image" << endl;
	    //_getch();
	    return -1;
	}
	cout << "Enter the Filename to store the matched image" << endl;
	cin >> MatchedImageFileName;

	//for grayscale
	if (inputImage.channels() == 1 && targetImage.channels() == 1) {
	    int inputSize = inputImage.rows * inputImage.cols;

	    float inputHistogram[LEVEL];
	    imageTopdf(inputImage, inputHistogram, inputSize);
	    
	    float inputTranFunc[LEVEL];
	    get_tf(inputHistogram, inputTranFunc);


	    int targetSize = targetImage.rows * targetImage.cols;
	    float targetHistogram[LEVEL];
	    imageTopdf(targetImage, targetHistogram, targetSize);

	    float targetTranFunc[LEVEL];
	    get_tf(targetHistogram, targetTranFunc);

		//for mapping
	    float outHistogram[LEVEL];
	    histogramMatching(inputTranFunc, targetTranFunc, outHistogram, targetHistogram);


	    Mat outputImage = inputImage.clone();

	    for (int y = 0; y < inputImage.rows; y++) {
		for (int x = 0; x < inputImage.cols; x++) {
		    outputImage.at<uchar>(y, x) = (int)(outHistogram[inputImage.at<uchar>(y, x)]);
		}
	    }

		imwrite("./output/"+inputFileName, inputImage);
	    imshow("Original Image", inputImage);
	    showHistogram(inputImage, inputFileName);
	    
		imwrite("./output/"+targetFileName, targetImage);
	    imshow("Target Image", targetImage);
	    showHistogram(targetImage, targetFileName);

		imwrite("./output/"+MatchedImageFileName, outputImage);
	    imshow("Histogram Matched Image", outputImage);
	    showHistogram(outputImage, MatchedImageFileName);

	    waitKey();

	}
	else {
	    cout << "RGB Histogram Matching" << endl;

	    Mat inputImage = imread(inputFileName, IMREAD_COLOR);
	    Mat inputImageHSI(inputImage.rows, inputImage.cols, inputImage.type());

	    float** inputImage_H = new float* [inputImage.rows];
	    float** inputImage_S = new float* [inputImage.rows];
	    float** inputImage_I = new float* [inputImage.rows];
	    for (int i = 0; i < inputImage.rows; i++) {
		inputImage_H[i] = new float[inputImage.cols];
		inputImage_S[i] = new float[inputImage.cols];
		inputImage_I[i] = new float[inputImage.cols];
	    }

	    convert_RGB_To_HSI(inputImage, inputImageHSI, inputImage_H, inputImage_S, inputImage_I);

	    float inputHistogram[LEVEL];
	    intensityToHistogram(inputImage_I, inputHistogram, inputImage.rows, inputImage.cols);

	    float inputTranFunc[LEVEL];
	    get_tf(inputHistogram, inputTranFunc);

	    Mat targetImage = imread(targetFileName, IMREAD_COLOR);
	    Mat targetImageHSI(targetImage.rows, targetImage.cols, targetImage.type());

	    float** targetImage_H = new float* [targetImage.rows];
	    float** targetImage_S = new float* [targetImage.rows];
	    float** targetImage_I = new float* [targetImage.rows];
	    for (int i = 0; i < targetImage.rows; i++) {
		targetImage_H[i] = new float[targetImage.cols];
		targetImage_S[i] = new float[targetImage.cols];
		targetImage_I[i] = new float[targetImage.cols];
	    }

	    convert_RGB_To_HSI(targetImage, targetImageHSI, targetImage_H, targetImage_S, targetImage_I);

	    float targetHistogram[LEVEL];
	    intensityToHistogram(targetImage_I, targetHistogram, targetImage.rows, targetImage.cols);

	    float targetTranFunc[LEVEL];
	    get_tf(targetHistogram, targetTranFunc);


	    float outHistogram[LEVEL];
	    histogramMatching(inputTranFunc, targetTranFunc, outHistogram, targetHistogram);

	    float** outI = new float* [inputImage.rows];
	    for (int i = 0; i < inputImage.rows; i++) {
		outI[i] = new float[inputImage.cols];
	    }

	    for (int i = 0; i < inputImage.rows; i++) {
		for (int j = 0; j < inputImage.cols; j++) {
		    outI[i][j] = (int)outHistogram[(int)inputImage_I[i][j]];
		}
	    }

	    Mat outputImage(inputImage.rows, inputImage.cols, inputImage.type());
	    convert_HSI_To_RGB(outputImage, inputImageHSI, inputImage_H, inputImage_S, outI);

		
		imwrite("./output/"+inputFileName, inputImage);
	    imshow("Original Image", inputImage);
	    showHistogram(inputImage, inputFileName);
	    
		imwrite("./output/"+targetFileName, targetImage);
	    imshow("Target Image", targetImage);
	    showHistogram(targetImage, targetFileName);

		imwrite("./output/"+MatchedImageFileName, outputImage);
	    imshow("Histogram Matched Image", outputImage);
	    showHistogram(outputImage, MatchedImageFileName);
	    

	    waitKey();
	}
	waitKey();
	return 0;
}






