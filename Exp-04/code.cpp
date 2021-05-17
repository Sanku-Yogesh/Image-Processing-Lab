#include "opencv2/core/core.hpp"
#define _CRT_SECURE_NO_WARNINGS
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cstdlib>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
# define PI 3.1416  


using namespace std;
using namespace cv;

const int IDEAL_LPF = 0;
const int IDEAL_HPF = 1;
const int GAUSSIAN_LPF = 2;
const int GAUSSIAN_HPF = 3;
const int BUTTER_LPF = 4;
const int BUTTER_HPF = 5;

int imgnum = 0;
int filternum = 0;
int cutoff_G = 1;
int gaussianSigma = 1; // 1 to 100 @ inc = 10
int butterN = 1; // 1 to 10 @ inc = 1
int butterC = 1; // 0.1 to 1 @ inc = 0.1
float inc = 0.05;
int imgmax = 8;
String input_image[8] = { "livingroom.jpg", "cameraman.jpg", "jetplane.jpg", "lake.jpg","livingroom.jpg", "mandril_gray.jpg", "pirate.jpg", "walkbridge.jpg" };

Mat img;
Mat img_after_filter;
Mat img_before_filter;
Mat filterspectrum;
Mat IFFTImg;

const string filters[6] = { "IDEAL_LPF", "IDEAL_HPF", "GAUSSIAN_LPF", "GAUSSIAN_HPF", "BUTTER_LPF", "BUTTER_HPF" };

class complex_float {
public:
	double real;
	double img;

public:
	complex_float()
	{
		this->real = 0;
		this->img = 0;
	}
	complex_float(double real, double img)
	{
		this->real = real;
		this->img = img;
	}
	complex_float operator+(const complex_float& b)
	{
		double r = real + b.real;
		double i = img + b.img;
		return complex_float(r, i);
	}
	complex_float operator-(const complex_float& b)
	{
		double r = real - b.real;
		double i = img - b.img;
		return complex_float(r, i);
	}
	complex_float operator*(const complex_float& b)
	{
		double k1 = b.real * (real + img);
		double k2 = real * (b.img - b.real);
		double k3 = img * (b.img + b.real);
		return complex_float(k1 - k3, k1 + k2);
	}

	complex_float operator*(const double& b)
	{
		return complex_float(real * b, img * b);
	}

	void operator*=(const double& b)
	{
		real *= b;
		img *= b;
	}

	complex_float operator/(const double& b)
	{
		return complex_float(real / b, img / b);
	}

	void operator=(const double& b)
	{
		real = b;
		img = 0;
	}

	double magnitude()
	{
		return sqrt(real * real + img * img);
	}
	void print() {
		cout << real << " + " << img << "i";
	}

};

template<typename T>
void Transpose(T** input_matrix, int N)
{
	T temp;
	for (int i = 0; i < N; i++) {
		T* start = input_matrix[i] + i;
		for (int j = i + 1; j < N; j++) {
			temp = input_matrix[i][j];
			input_matrix[i][j] = input_matrix[j][i];
			input_matrix[j][i] = temp;
		}
	}
}

template<typename T>
void FFTShift(T** input_matrix, int N)
{
	T temp;
	int offset = N / 2;
	for (int i = 0; i < offset; i++) {
		T* start = input_matrix[i] + i;
		for (int j = 0; j < offset; j++) {
			temp = input_matrix[i][j];
			input_matrix[i][j] = input_matrix[i + offset][j + offset];
			input_matrix[i + offset][j + offset] = temp;
		}
	}

	for (int i = N / 2; i < N; i++) {
		T* start = input_matrix[i] + i;
		for (int j = 0; j < offset; j++) {
			temp = input_matrix[i][j];
			input_matrix[i][j] = input_matrix[i - offset][j + offset];
			input_matrix[i - offset][j + offset] = temp;
		}
	}
}

template<typename T>
void FFTShift(Mat& input_matrix, int N)
{
	T temp;
	int offset = N / 2;
	for (int i = 0; i < offset; i++) {
		for (int j = 0; j < offset; j++) {
			temp = input_matrix.at<T>(i, j);
			input_matrix.at<T>(i, j) = input_matrix.at<T>(i + offset, j + offset);
			input_matrix.at<T>(i + offset, j + offset) = temp;
		}
	}

	for (int i = N / 2; i < N; i++) {
		for (int j = 0; j < offset; j++) {
			temp = input_matrix.at<T>(i, j);
			input_matrix.at<T>(i, j) = input_matrix.at<T>(i - offset, j + offset);
			input_matrix.at<T>(i - offset, j + offset) = temp;
		}
	}
}

complex_float* FFT(uchar* x, int N, int arrSize, int zeropos, int gap)
{
	complex_float* fft;
	fft = new complex_float[N];

	int i;
	if (N == 2)
	{
		fft[0] = complex_float(x[zeropos] + x[zeropos + gap], 0);
		fft[1] = complex_float(x[zeropos] - x[zeropos + gap], 0);
	}
	else
	{
		complex_float wN = complex_float(cos(2 * PI / N), sin(-2 * PI / N));
		complex_float w = complex_float(1, 0);
		gap *= 2;
		complex_float* X_even = FFT(x, N / 2, arrSize, zeropos, gap); //N/2 POINT DFT OF EVEN X's
		complex_float* X_odd = FFT(x, N / 2, arrSize, zeropos + (arrSize / N), gap); //N/2 POINT DFT OF ODD X's
		complex_float todd;
		for (i = 0; i < N / 2; ++i)
		{
			todd = w * X_odd[i];
			fft[i] = X_even[i] + todd;
			fft[i + N / 2] = X_even[i] - todd;
			w = w * wN;
		}

		delete[] X_even;
		delete[] X_odd;
	}

	return fft;
}

complex_float* FFT(complex_float* x, int N, int arrSize, int zeropos, int gap)
{
	complex_float* fft;
	fft = new complex_float[N];

	int i;
	if (N == 2)
	{
		fft[0] = x[zeropos] + x[zeropos + gap];
		fft[1] = x[zeropos] - x[zeropos + gap];
	}
	else
	{
		complex_float wN = complex_float(cos(2 * PI / N), sin(-2 * PI / N));
		complex_float w = complex_float(1, 0);
		gap *= 2;
		complex_float* X_even = FFT(x, N / 2, arrSize, zeropos, gap); //N/2 POINT DFT OF EVEN X's
		complex_float* X_odd = FFT(x, N / 2, arrSize, zeropos + (arrSize / N), gap); //N/2 POINT DFT OF ODD X's
		complex_float todd;
		for (i = 0; i < N / 2; ++i)
		{
			todd = w * X_odd[i];
			fft[i] = X_even[i] + todd;
			fft[i + N / 2] = X_even[i] - todd;
			w = w * wN;
		}

		delete[] X_even;
		delete[] X_odd;
	}

	return fft;
}

complex_float* IFFT(complex_float* fft, int N, int arrSize, int zeropos, int gap)
{
	complex_float* signal;
	signal = new complex_float[N];

	int i;
	if (N == 2)
	{
		signal[0] = fft[zeropos] + fft[zeropos + gap];
		signal[1] = fft[zeropos] - fft[zeropos + gap];
	}
	else
	{
		complex_float wN = complex_float(cos(2 * PI / N), sin(2 * PI / N));
		complex_float w = complex_float(1, 0);
		gap *= 2;
		complex_float* X_even = IFFT(fft, N / 2, arrSize, zeropos, gap); //N/2 POINT DFT OF EVEN X's
		complex_float* X_odd = IFFT(fft, N / 2, arrSize, zeropos + (arrSize / N), gap); //N/2 POINT DFT OF ODD X's
		complex_float todd;
		for (i = 0; i < N / 2; ++i)
		{
			todd = w * X_odd[i];
			signal[i] = (X_even[i] + todd) * 0.5;
			signal[i + N / 2] = (X_even[i] - todd) * 0.5;
			w = w * wN;
		}

		delete[] X_even;
		delete[] X_odd;
	}

	return signal;
}


complex_float** IFFT2D(complex_float** orig_image, int N) {

	complex_float** ifftResult;
	ifftResult = new complex_float * [N];
	// ROW WISE FFT
	for (int i = 0; i < N; i++) {
		ifftResult[i] = IFFT(orig_image[i], N, N, 0, 1);
	}
	Transpose<complex_float>(ifftResult, N);

	int d = N * N;
	// COLUMN WISE FFT
	for (int i = 0; i < N; i++) {
		ifftResult[i] = IFFT(ifftResult[i], N, N, 0, 1);
		for (int j = 0; j < N; j++) {
			ifftResult[i][j] = ifftResult[i][j] / d;
		}
	}
	Transpose<complex_float>(ifftResult, N);

	cout << endl;

	return ifftResult;
}

complex_float** FFT2D(Mat& orig_img) {
	if (orig_img.rows != orig_img.cols) {
		cout << "img is not Valid";
		return nullptr;
	}
	int N = orig_img.rows;
	complex_float** FFT2Result_h;
	FFT2Result_h = new complex_float * [N];

	// ROW WISE FFT
	for (int i = 0; i < N; i++) {
		uchar* row = orig_img.ptr<uchar>(i);
		FFT2Result_h[i] = FFT(row, N, N, 0, 1);
	}
	Transpose<complex_float>(FFT2Result_h, N);

	// COLUMN WISE FFT
	for (int i = 0; i < N; i++) {
		FFT2Result_h[i] = FFT(FFT2Result_h[i], N, N, 0, 1);
	}
	Transpose<complex_float>(FFT2Result_h, N);

	return FFT2Result_h;
}

void complex_to_mat(complex_float** orig_image, Mat& dest, int N, bool shift = false, float maxF = 1.0) {
	if (shift) {
		FFTShift(orig_image, N);
	}
	dest = Mat(N, N, CV_32F, cv::Scalar::all(0));
	float min = 99999;
	float max = 0;

	// Find min and max
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			orig_image[i][j] = orig_image[i][j] / N;
			float m = orig_image[i][j].magnitude();
			if (m < min) {
				min = m;
			}
			if (m > max) {
				max = m;
			}
		}
	}


	// Normalize the image
	float range = (max - min);
	for (int i = 0; i < N; i++) {
		float* p = dest.ptr<float>(i);
		for (int j = 0; j < N; j++) {
			p[j] = (orig_image[i][j].magnitude() - min) * maxF / range;
		}
	}
}

void ApplyFilter(complex_float** orig_image, Mat& filterspectrum, int N, int FilterType) {
	float cutoff = cutoff_G * inc; //Ideal filter cutoff
	float sigma_squared = gaussianSigma * inc + inc; //Compute Gaussing filter sigma from trackbar input
	int butter_n = butterN; // Butterworth parameter n
	// Cutoff lies in [0, 2]
	cutoff *= cutoff; // Square it to avoid further sqrt
	filterspectrum = Mat(N, N, CV_32F); // Image for showing the frequency spectrum of the filter
	float d = N * N;
	complex_float** filterFFT;
	switch (FilterType) {
	case IDEAL_LPF:
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float f = (i * i / d) + (j * j / d);
				if (f > cutoff) {
					orig_image[i][j] = 0;
					orig_image[N - 1 - i][N - 1 - j] = 0;
					orig_image[N - 1 - i][j] = 0;
					orig_image[i][N - 1 - j] = 0;
				}
				else {
					filterspectrum.at<float>(i, j) = filterspectrum.at<float>(N - 1 - i, N - 1 - j) = filterspectrum.at<float>(N - 1 - i, j) = filterspectrum.at<float>(i, N - 1 - j) = 1;
				}
			}
		}
		break;
	case IDEAL_HPF:
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float f = (i * i / d) + (j * j / d);
				if (f <= cutoff) {
					orig_image[i][j] = 0;
					orig_image[N - 1 - i][N - 1 - j] = 0;
					orig_image[N - 1 - i][j] = 0;
					orig_image[i][N - 1 - j] = 0;
				}
				else {
					filterspectrum.at<float>(i, j) = filterspectrum.at<float>(N - 1 - i, N - 1 - j) = filterspectrum.at<float>(N - 1 - i, j) = filterspectrum.at<float>(i, N - 1 - j) = 1;
				}
			}
		}
		break;
	case GAUSSIAN_LPF:
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float wx2 = pow(2 * PI * i / N, 2);
				float wy2 = pow(2 * PI * j / N, 2);
				float coeff = exp(-(wx2 + wy2) / (2 * sigma_squared));
				orig_image[i][j] *= coeff;
				orig_image[N - 1 - i][N - 1 - j] *= coeff;
				orig_image[N - 1 - i][j] *= coeff;
				orig_image[i][N - 1 - j] *= coeff;

				filterspectrum.at<float>(i, j) = filterspectrum.at<float>(N - 1 - i, N - 1 - j) = filterspectrum.at<float>(N - 1 - i, j) = filterspectrum.at<float>(i, N - 1 - j) = coeff;
			}
		}
		break;
	case GAUSSIAN_HPF:
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float wx2 = pow(2 * PI * i / N, 2);
				float wy2 = pow(2 * PI * j / N, 2);
				float coeff = 1 - exp(-(wx2 + wy2) / (2 * sigma_squared));
				orig_image[i][j] *= coeff;
				orig_image[N - 1 - i][N - 1 - j] *= coeff;
				orig_image[N - 1 - i][j] *= coeff;
				orig_image[i][N - 1 - j] *= coeff;

				filterspectrum.at<float>(i, j) = filterspectrum.at<float>(N - 1 - i, N - 1 - j) = filterspectrum.at<float>(N - 1 - i, j) = filterspectrum.at<float>(i, N - 1 - j) = coeff;

			}
		}
		break;
	case BUTTER_LPF:
		cutoff = pow((butterC * inc + inc) * PI, 2);
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float wx2 = pow(2 * PI * i / N, 2);
				float wy2 = pow(2 * PI * j / N, 2);
				float coeff = 1 / (1 + pow((wx2 + wy2) / cutoff,  butter_n));
				orig_image[i][j] *= coeff;
				orig_image[N - 1 - i][N - 1 - j] *= coeff;
				orig_image[N - 1 - i][j] *= coeff;
				orig_image[i][N - 1 - j] *= coeff;

				filterspectrum.at<float>(i, j) = filterspectrum.at<float>(N - 1 - i, N - 1 - j) = filterspectrum.at<float>(N - 1 - i, j) = filterspectrum.at<float>(i, N - 1 - j) = coeff;

			}
		}
		break;
	case BUTTER_HPF:
		cutoff = pow((butterC * inc + inc) * PI, 2);
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float wx2 = pow(2 * PI * i / N, 2);
				float wy2 = pow(2 * PI * j / N, 2);
				float coeff = 1 / (1 + pow(cutoff / (wx2 + wy2),  butter_n));
				orig_image[i][j] *= coeff;
				orig_image[N - 1 - i][N - 1 - j] *= coeff;
				orig_image[N - 1 - i][j] *= coeff;
				orig_image[i][N - 1 - j] *= coeff;

				filterspectrum.at<float>(i, j) = filterspectrum.at<float>(N - 1 - i, N - 1 - j) = filterspectrum.at<float>(N - 1 - i, j) = filterspectrum.at<float>(i, N - 1 - j) = coeff;

			}
		}
		break;
	}
}

void apply_filter(string filename) {

	if (!img.data)
	{
		cout << "eeror in img data access";
		return;
	}
	complex_float** fft2result = FFT2D(img);
	complex_to_mat(fft2result, img_before_filter, img.rows, false, 255);
	ApplyFilter(fft2result, filterspectrum, img.rows, filternum);
	FFTShift<float>(filterspectrum, img.rows);	//ApplyFilter(fft2result, img.rows, IDEAL_LPF, 0.15);

	complex_float** ifft2result = IFFT2D(fft2result, img.rows);
	float maxF = 1;
	switch (filternum) {
	case IDEAL_LPF:
	case GAUSSIAN_LPF:
	case BUTTER_LPF:
		maxF = 255;
	}
	complex_to_mat(fft2result, img_after_filter, img.rows, true, maxF);
	complex_to_mat(ifft2result, IFFTImg, img.rows);


	FFTShift<float>(img_before_filter, img.rows);

	// Show the results
	imshow("FFT Before Filter", img_before_filter);
	imshow("FFT After Filter", img_after_filter);
	imshow("Filter Spectrum", filterspectrum);
	imshow("Final Output image", IFFTImg);
}

void cutoff_change(int, void*) {
	cout << "Cutoff frequency chosen: " << cutoff_G * inc + inc << endl;
	apply_filter(input_image[imgnum]);
}

void GSigma_change(int, void*) {
	cout << "Sigma Value: " << gaussianSigma * inc + inc << endl;
	apply_filter(input_image[imgnum]);
}

void butterN_change(int, void*) {
	cout << "ButterWorth Order: " << butterN << endl;
	apply_filter(input_image[imgnum]);
}

void butterC_change(int, void*) {
	cout << "ButterWorth cutoff: " << (butterC * inc + inc) << endl;
	apply_filter(input_image[imgnum]);
}

void img_change(int, void*) {

	img = imread(input_image[imgnum], 0);

	if (!img.data) // Check for invalid input
	{
		cout << "Could not open or find the img" << endl;
		return;
	}
	imshow("original_image", img); // Show our img inside it.
	apply_filter(input_image[imgnum]);
}

void filter_change(int, void*) {
	cout << "Filter: " << filters[filternum] << endl;
	apply_filter(input_image[imgnum]);
}

int main()
{
	
	namedWindow("frequency filtering", 0);
	createTrackbar("img selection", "frequency filtering", &imgnum, imgmax, img_change);// selecting image
	createTrackbar("Filter selection", "frequency filtering", &filternum, 5, filter_change); //selecting filter
	createTrackbar("Ideal Filter cutoff ", "frequency filtering", &cutoff_G, 0.5 / inc, cutoff_change); // cutoff frequency
	createTrackbar("Gaussian Filter Sigma ", "frequency filtering", &gaussianSigma, 2 / inc, GSigma_change); // gaussian sigma
	createTrackbar("ButterWorth n ", "frequency filtering", &butterN, 10, butterN_change); // butterworth order
	createTrackbar("ButterWorth c", "frequency filtering", &butterC, 1 / inc, butterC_change); // butterworth cutoff
	img_change(imgnum, 0);
	namedWindow("Filter Spectrum", 0);
	waitKey(0);
	return 0;
}
