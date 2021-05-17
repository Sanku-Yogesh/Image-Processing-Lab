#include <fstream>
#include <bits/stdc++.h>
#include <experimental/filesystem>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>
#include <cmath>

namespace fs = std::experimental::filesystem;
using namespace cv;
using namespace std;

int fileName = 0;
int filterType = 0;
int filterSize = 0;

vector<string> fileList;

const string filters[11] = {"No", "Mean", "Median", "Prewitt_H", "Prewitt_V", "Laplacian", "Sobel_H", "Sobel_V", "Sobel_D", "Gaussian" , "LoG"};

//defining some filters
int Prewitt_H_3[3][3] = {  {-1, -1, -1 },
							{ 0,  0,  0 },
					   	    { 1,  1,  1 }};

int Prewitt_H_5[5][5] = { { -1, -1, -1, -1, -1 },
						   { -2, -2, -2, -2, -2 },
						   {  0,  0,  0,  0,  0 },
						   {  2,  2,  2,  2,  2 },
						   {  1,  1,  1,  1,  1 } };

int Prewitt_H_7[7][7] =  { { -1, -1, -1, -1, -1, -1, -1 },
							{ -2, -2, -2, -2, -2, -2, -2 },
							{ -3, -3, -3, -3, -3, -3, -3 },
							{ 0, 0, 0, 0, 0, 0, 0 },
							{ 3, 3, 3, 3, 3, 3, 3 },
							{ 2, 2, 2, 2, 2, 2, 2 },
							{ 1, 1, 1, 1, 1, 1, 1 } };


int Prewitt_H_9[9][9] =  { {-1,-1, -1, -1, -1, -1, -1, -1, -1 },
							{-2,-2, -2, -2, -2, -2, -2, -2, -2 },
							{-3,-3, -3, -3, -3, -3, -3, -3, -3 },
							{-4,-4,-4,-4,-4,-4,-4,-4,-4},
							{0,0, 0, 0, 0, 0, 0, 0, 0 },
							{4,  4,  4,  4,  4,  4,  4,  4,  4},
							{3,3, 3, 3, 3, 3, 3, 3, 3 },
							{2,2, 2, 2, 2, 2, 2, 2, 2 },
							{1,1, 1, 1, 1, 1, 1, 1, 1 } };

int Prewitt_V_3[3][3] = { { 1, 0, -1 },
						   { 1, 0, -1 },
						   { 1, 0, -1 } };

int Prewitt_V_5[5][5] ={ {-1,-2,0,2,1},
							{-1,-2,0,2,1},
							{-1,-2,0,2,1},
							{-1,-2,0,2,1},
							{-1,-2,0,2,1}
							};

int Prewitt_V_7[7][7] = {{-1,-2,-3,0,3,2,1},
							{-1,-2,-3,0,3,2,1},
							{-1,-2,-3,0,3,2,1},
							{-1,-2,-3,0,3,2,1},
							{-1,-2,-3,0,3,2,1},
							{-1,-2,-3,0,3,2,1},
							{-1,-2,-3,0,3,2,1}};

int Prewitt_V_9[9][9] = {{-1,-2,-3,-4,0,4,3,2,1},
							{-1,-2,-3,-4,0,4,3,2,1},
							{-1,-2,-3,-4,0,4,3,2,1},
							{-1,-2,-3,-4,0,4,3,2,1},
							{-1,-2,-3,-4,0,4,3,2,1},
							{-1,-2,-3,-4,0,4,3,2,1},
							{-1,-2,-3,-4,0,4,3,2,1},
							{-1,-2,-3,-4,0,4,3,2,1},
							{-1,-2,-3,-4,0,4,3,2,1}};

int laplacian_3[3][3] = {{-1, -1, -1},
						{-1, 8, -1},
						{-1, -1, -1}};
							
int laplacian_5[5][5] = {{-1, -1,-1,-1,-1},
						{-1, -1,  -1,  -1, -1},
						{-1, -1, 24,  -1, -1},
						{-1, -1,  -1,  -1, -1},
						{-1,-1, -1, -1, -1}};


int laplacian_7[7][7] = {	{-1, -1, -1, -1, -1, -1, -1},
							{-1, -1, -1, -1, -1, -1, -1},
							{-1, -1, -1, -1, -1, -1, -1},
							{-1, -1, -1, 48, -1, -1, -1},
							{-1, -1, -1, -1, -1, -1, -1},
							{-1, -1, -1, -1, -1, -1, -1},
							{-1, -1, -1, -1, -1, -1, -1}};
							
int laplacian_9[9][9] = {	{-1,-1, -1,-1, -1, -1, -1, -1, -1},
							{-1,-1, -1,-1, -1, -1, -1, -1, -1},
							{-1,-1, -1,-1, -1, -1, -1, -1, -1},
							{-1,-1, -1,-1, -1, -1, -1, -1, -1},
							{-1,-1, -1,-1, 80, -1, -1, -1, -1},
							{-1,-1, -1,-1, -1, -1, -1, -1, -1},
							{-1,-1, -1,-1, -1, -1, -1, -1, -1},
							{-1,-1, -1,-1, -1, -1, -1, -1, -1},
							{-1,-1, -1,-1, -1, -1, -1, -1, -1}};

vector<vector<int>> sobel_H_3= { {  1,  2,  1},
						{  0,  0,  0},
						{ -1, -2, -1} };

vector<vector<int>> sobel_H_5= { { 1,   4,   6,   4,  1},
						{ 2,   8,  12,   8,  2},
						{ 0,   0,   0,   0,  0},
						{-2,  -8 , -12, -8, -2},
						{ -1, -4,  -6,  -4, -1 } };

vector<vector<int>> sobel_H_7= { {  1,  6,  15,  20,  15,  6,  1},
						{  2,  12, 30,  40, 30, 12,  2},
						{  3, 18, 45,  60, 45, 18,  3},
						{  0,  0,  0,   0,  0,  0,  0},
						{ -3, -18,-45,-60,-45,-18, -3},
						{ -2, -12, -30,-40,-30,-12, -2},
						{ -1, -6, -15, -20, -15, -6, -1} };

					
vector<vector<int>> sobel_H_9 = { {  1,  8,  28,  56, 70 , 56,  28,  8,  1},
						{  2,  16, 56,  112, 140 , 112 , 56 , 16,  2},
						{  3, 24, 84,  168, 210, 168, 84,  24, 3},
						{  4, 32, 112,  224, 280, 224, 112, 32, 4},
						{  0,  0,  0,   0,  0,  0,  0, 0 , 0},
						{  -4, -32, -112,  -224, -280, -224, -112, -32, -4},
						{ -3, -24, -84,  -168, -210, -168, -84,  -24, -3},
						{ -2,  -16, -56,  -112, -140 , -112 , -56 , -16,  -2},
						{ -1,  -8,  -28,  -56, -70 , -56,  -28, -8,  -1} };


vector<vector<int>> sobel_V_3(3,vector<int>(3,0));

vector<vector<int>> sobel_V_5(5,vector<int>(5,0));

vector<vector<int>> sobel_V_7(7,vector<int>(7,0));

vector<vector<int>> sobel_V_9(9,vector<int>(9,0));

vector<vector<int>> sobel_D_3(3,vector<int>(3,0));

vector<vector<int>> sobel_D_5(5,vector<int>(5,0));

vector<vector<int>> sobel_D_7(7,vector<int>(7,0));

vector<vector<int>> sobel_D_9(9,vector<int>(9,0));

//for constructing sobel vertical filters
void flip(){
	int n=3;
	while(n<=9){
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				//storing the values
				if(n==3) sobel_V_3[i][j] = sobel_H_3[j][i];
				if(n==5) sobel_V_5[i][j] = sobel_H_5[j][i];
				if(n==7) sobel_V_7[i][j] = sobel_H_7[j][i];
				if(n==9) sobel_V_9[i][j] = sobel_H_9[j][i];
			}
		}
		n = n + 2;
	}
}

//for constructing sobel diagonal filters
void rotate(vector<vector<int>> &matrix,int n,vector<vector<int>> &out){
	
	int N = 2*n;
	for(int level = 0;level < n;level++){
		int i=level,j=level;
		do{
			int shift = n-level;
			//conditions
			if(n-i>=0 && n-j>=0){
				if(j-shift>=level) out[i][j-shift] = matrix[i][j];
				else out[i-(j-shift-level)][level] = matrix[i][j];
				if(j-1>=level) j--;
				else i++;
			}
			else 	if(n-i<=0 && n-j>=0){
					if(i+shift<=N-level) out[i+shift][j] = matrix[i][j];
					else out[N-level][j+i+shift-N+level] = matrix[i][j];
					if(i+1<=N-level) i++;
					else j++;
			}
			else 	if(n-i<=0 && n-j<=0){
					if(j+shift<=N-level) out[i][j+shift] = matrix[i][j];
					else out[i-(j+shift-N+level)][N-level] = matrix[i][j];
					if(j+1<=N-level) j++;
					else i--;
			}
			else 	if(n-i>=0 && n-j<=0){
					if(i-shift>=level) out[i-shift][j] = matrix[i][j];
					else out[level][j-(level-i+shift)] = matrix[i][j];
					if(i-1>=level) i--;
					else j--;
			}
			
		}while(i!=level || j!=level);
	}
}
		
//datatype to store filter
typedef struct Frame{
    int side;
    int **data;
}Frame;

//dynamic memory allocation for filters
Frame allocateWindow (int size , int *temp){
	Frame window;
	window.side = size;
	window.data = new int*[window.side];

	for (int i = 0; i < window.side; i++)
		window.data[i] = new int[window.side];

	for (int i = 0; i < window.side; i++){
		for (int j = 0; j < window.side; j++){
			window.data[i][j] = *((temp + i*size) + j);
		}
	}
	return (window);
}

//same as the above function 
Frame S_allocateWindow (int size , vector<vector<int>> &temp){
	Frame window;
	window.side = size;
	window.data = new int*[window.side];

	for (int i = 0; i < window.side; i++)
		window.data[i] = new int[window.side];

	for (int i = 0; i < window.side; i++){
		for (int j = 0; j < window.side; j++){
			window.data[i][j] = temp[i][j];
		}
	}
	return (window);
}

//function to store paths of images in the directory
void GetFiles(vector<string>& out, const string& directory){
	string path = directory;
	for (const auto & entry : fs::directory_iterator(path))
		out.push_back(entry.path());
}

//convolution of filter and image for prewitt, sobel, laplacian
Mat general(Mat input, Frame *window){
	int i, j, count, sum;
	i = j = count = 0;
	int new_size = (window->side)/2;
	Mat output = input.clone();

	for(i = 0; i < input.cols; i++){
		for(j = 0; j < input.rows; j++){
			sum = 0;
			count = 0;
			for(int i1 = -new_size; i1 <= new_size; i1++)
				if(((i + i1) >= 0 ) && ((i + i1) < input.cols))
					for(int j1 = -new_size; j1 <= new_size; j1++)
						if(((j + j1) >= 0) && ((j + j1) < input.rows)){
				  			sum += window->data[i1 + new_size][j1 + new_size] * input.at<uchar>(i + i1, j + j1);
				  			count += abs(window->data[i1 + new_size][j1 + new_size]);
						}
			//checking for clipping
			if(sum >  255) sum = 255;
			else if(sum < 0) sum = 0;
			output.at<uchar>(i, j ) = sum;
		}
	}
  return output;
}

//prewitt
Mat Prewitt(Mat input, string type , int size){
	cout << "Prewitt Operator ";
	Frame mask;
	if (type == "Horizontal"){
		cout << "Type: " << type << " ";
		if(size == 3){
			cout << "Size: " << size << endl;
			mask = allocateWindow(3, (int *)Prewitt_H_3);
		}
		if(size == 5){
			cout << "Size: " << size << endl;
			mask = allocateWindow(5, (int *)Prewitt_H_5);
		}
		if(size == 7){
			cout << "Size: " << size << endl;
			mask = allocateWindow(7, (int *)Prewitt_H_7);
		}
		if(size == 9){
			cout << "Size: " << size << endl;
			mask = allocateWindow(9, (int *)Prewitt_H_9);
		}
	}

	if (type == "Vertical"){
		cout << "Type: " << type << " ";
		if(size == 3){
			cout << "Size: " << size << endl;
			mask = allocateWindow(3, (int *)Prewitt_V_3);
		}
		if(size == 5){
			cout << "Size: " << size << endl;
			mask = allocateWindow(5, (int *)Prewitt_V_5);
		}
		if(size == 7){
			cout << "Size: " << size << endl;
			mask = allocateWindow(7, (int *)Prewitt_V_7);
		}
		if(size == 9){
			cout << "Size: " << size << endl;
			mask = allocateWindow(9, (int *)Prewitt_V_9);
		}
	}
	Mat output = general(input, &mask);

	return output;
}

//laplacian
Mat Laplacian(Mat input, int size){
	Frame mask;
	cout << "Laplacian Operator ";
	if(size == 3){
		cout << "Size: " << size << endl;
		mask = allocateWindow(3, (int *)laplacian_3);
	}

	if(size == 5){
		cout << "Size: " << size << endl;
		mask = allocateWindow(5, (int *)laplacian_5);
	}

	if(size == 7){
		cout << "Size: " << size << endl;
		mask = allocateWindow(7, (int *)laplacian_7);
	}
	if(size == 9){
		cout << "Size: " << size << endl;
		mask = allocateWindow(9, (int *)laplacian_9);
	}
	Mat output = general(input, &mask);
	return output;
}

//sobel
Mat Sobel(Mat input, string type,  int size){
	Frame mask;
	cout << "Sobel Operator ";

	if (type == "Horizontal"){
		cout << "Type: " << type << " ";
		if(size == 3){
			cout << "Size: " << size << endl;
			mask = S_allocateWindow(3, sobel_H_3);
		}
		if(size == 5){
			cout << "Size: " << size << endl;
			mask = S_allocateWindow(5, sobel_H_5);
		}
		if(size == 7){
			cout << "Size: " << size << endl;
			mask = S_allocateWindow(7, sobel_H_7);
		}
		if(size == 9){
			cout << "Size: " << size << endl;
			mask = S_allocateWindow(9, sobel_H_9);
		}
	}

	if (type == "Vertical"){
		cout << "Type: " << type << " ";
		if(size == 3){
			cout << "Size: " << size << endl;
			mask = S_allocateWindow(3, sobel_V_3);
		}
		if(size == 5){
			cout << "Size: " << size << endl;
			mask = S_allocateWindow(5, sobel_V_5);
		}
		if(size == 7){
			cout << "Size: " << size << endl;
			mask = S_allocateWindow(7, sobel_V_7);
		}
		if(size == 9){
			cout << "Size: " << size << endl;
			mask = S_allocateWindow(9, sobel_V_9);
		}
	}

	if (type == "Diagonal"){
		cout << "Type: " << type << " ";
		if (size == 3){
			cout << "Size: " << size << endl;
			mask = S_allocateWindow(3, sobel_D_3);
		}
		if(size == 5){
			cout << "Size: " << size << endl;
			mask = S_allocateWindow(5, sobel_D_5);
		}
		if(size == 7){
			cout << "Size: " << size << endl;
			mask = S_allocateWindow(7, sobel_D_7);
		}
		if(size == 9){
			cout << "Size: " << size << endl;
			mask = S_allocateWindow(9, sobel_D_9);
		}
	}

	Mat output = general(input, &mask);

	return output;
}

//convolution for gaussian filter
Mat convolute(Mat input_image, bool padding, Mat kernel) {
	Mat out = input_image.clone();
	int span = (kernel.rows - 1) / 2;
	float value;
	int i_, j_;

	for (int x = 0; x < out.rows; x++)
		for (int y = 0; y < out.cols; y++) {
			uchar& output_intensity = out.at<uchar>(x, y);
			value = 0;
			for (int i = 0; i < kernel.rows; i++) {
				for (int j = 0; j < kernel.cols; j++) {
					i_ = x - span + i;
					j_ = y - span + j;
					if (i_ < 0) {i_ = 0;continue;}
					if (j_ < 0) {j_ = 0;continue;}
					if (i_ >= out.rows) {i_ = out.rows - 1;continue;}
					if (j_ >= out.cols) {j_ = out.cols - 1;continue;}
					uchar input_intensity = input_image.at<uchar>(i_, j_);
					value += (1.0 * kernel.at<float>(i, j) * input_intensity);
				}
				output_intensity = (int)value;
			}
		}
	return out;
}

Mat gaussian_filter(Mat input_image, int kernel_size)
{
	cout << "gaussian operator" << endl;
	int span = (kernel_size - 1) / 2;
	//taking standard deviation as 1
	double stdv = 1.0;
	double r, s = 2.0 * stdv * stdv;
	// Initialization of sum for normalization
	double sum = 0.0;
	//dynamic memory allocation
	float** kernel = (float**)malloc(kernel_size * sizeof(float*));
	for (int i = 0; i < kernel_size; i++)
		kernel[i] = (float*)malloc(kernel_size * sizeof(float));

	//calculating
	for (int x = -span; x <= span; x++) // Loop to generate kernel
	{
		for (int y = -span; y <= span; y++)
		{
			r = (x * x + y * y);
			kernel[x + span][y + span] = (exp(-(r) / s)) / sqrt((22 / 7) * s);
			sum += kernel[x + span][y + span];
		}
	}

	Mat gaussian_kernel(kernel_size, kernel_size, CV_32F);
	for (int i = 0; i < kernel_size; i++) // Loop to normalize the kernel
		for (int j = 0; j < kernel_size; j++)
		{
			//normalisation
			kernel[i][j] /= sum;
			gaussian_kernel.at<float>(i, j) = kernel[i][j];

		}

	return convolute(input_image, true, gaussian_kernel);
}

//log operator
Mat log_filter(Mat input_image, int kernel_size)
{
	Mat gout = gaussian_filter(input_image, kernel_size);
	Mat out = Laplacian(gout, kernel_size);
	return out;
}

//function for mean filter
Mat Mean(Mat input, int size){
  int i, j, count, sum;
  i = j = count = 0;
  int new_size = size/2;
  Mat output = input.clone();

  //iterating 
  for(i = 0; i < input.cols; i++)
    for(j = 0; j < input.rows; j++){
      sum = 0;
      count = 0;
      //zero padding
      for(int i1 = -new_size; i1 <= new_size; i1++)
        if(((i + i1) >= 0 ) && ((i + i1) < input.cols))
          for(int j1 = -new_size; j1 <= new_size; j1++)
            if(((j + j1) >= 0) && ((j + j1) < input.rows)){
              //filter coefficients are 1 for mean filter
              sum += input.at<uchar>(i + i1, j + j1);
              count++;
            }
      //output.at<uchar>(i, j) = (sum/count);
      output.at<uchar>(i, j) = (sum/(size*size));
    }

  return output;
}

//function for median filter
Mat Median(Mat input, int size){
  int i, j, count, median;
  i = j = count = 0;
  int new_size = size/2;
  Mat output = input.clone();

  vector<int> frame ( size*size, 0);
  for(i = 0; i < input.cols; i++){
    for(j = 0; j < input.rows; j++){
        count = 0;
        frame.clear();
        //pushing the intensity values into vector
		for(int i1 = -new_size; i1 <= new_size; i1++){
			if(((i + i1) >= 0 ) && ((i + i1) < input.cols)){
				for(int j1 = -new_size; j1 <= new_size; j1++)
					if(((j + j1) >= 0) && ((j + j1) < input.rows))
						frame.push_back(input.at<uchar>(i + i1, j + j1));
					else frame.push_back(0);
			}
			else frame.push_back(0);
		}
		sort(frame.begin(), frame.end());
		size = frame.size();
		
		//checking if size is even or odd
		if (size%2) median = frame[size / 2];
		else median = (frame[size / 2 - 1] + frame[size / 2]) / 2;
		output.at<uchar>(i, j ) = median;
    }
  }

  return output;
}

//this function will be called whenever there is a change in filter
//and appropriate filters will be applied based on filterType
void onFilterChange(int, void*){
	cout<< "Applying "<< filters[ filterType ] <<" Filter: " << endl;

	Mat img2 = imread(fileList[fileName], IMREAD_GRAYSCALE);

	//calling appropriate filters
	if (filterType == 0){
		}
	else if (filterType == 1){
		img2 = Mean(img2, 2*filterSize+3);
		}
	else if (filterType == 2){
		img2 = Median(img2, 2*filterSize+3);
		}
	else if (filterType == 3){
		img2 = Prewitt(img2, "Horizontal", 2*filterSize+3);
		}
	else if (filterType == 4){
		img2 = Prewitt(img2, "Vertical", 2*filterSize+3);
		}
	else if (filterType == 5){
		img2 = Laplacian(img2, 2*filterSize+3);
		}
	else if (filterType == 6){
		img2 = Sobel(img2, "Horizontal", 2*filterSize+3);
		}
	else if (filterType == 7){
		img2 = Sobel(img2, "Vertical", 2*filterSize+3);
		}
	else if (filterType == 8){
		img2 = Sobel(img2, "Diagonal", 2*filterSize+3);
		}
	else if (filterType == 9){
		img2 = gaussian_filter(img2, 2*filterSize+3);
		}
	else if (filterType == 10){
		img2 = log_filter(img2, 2*filterSize+3);
		}
	
	cout<< "Operation Complete"<<endl;
	imshow("Output Image", img2);
}

//this function will be called whenever there is a change in filter size
void onSizeChange(int, void*){
	cout<< "Size of "<< filters[ filterType ] <<" Filter changed to "<< 2*filterSize+3 <<endl;
	//applying filter
	onFilterChange(filterType, 0);
}

//this function will be called whenever there is a change in input file
void onInputChange(int, void*){
	cout << "Opening: " << fileList[fileName] << endl;

	Mat img = imread(fileList[fileName], IMREAD_GRAYSCALE);

	if (!img.data){
		cout << "Error:Image not found" <<endl;
		return;
	}
	imshow("Input Image", img);
	//applying filter
	onFilterChange(filterType , 0);
}

int main() {
	//construct sobel vertical
	flip();
	
	//construct sobel diagonal
	rotate(sobel_V_3,1,sobel_D_3);
	rotate(sobel_V_5,2,sobel_D_5);
	rotate(sobel_V_7,3,sobel_D_7);
	rotate(sobel_V_9,4,sobel_D_9);
	
	//store names of images in filelist
	GetFiles(fileList, "./NormalImages");
	GetFiles(fileList, "./NoisyImages");

	//creating the windows
	namedWindow("Input Image");
	namedWindow("Output Image");

	//attaching the tracker to input image window 
	createTrackbar("Source Choose", "Input Image", &fileName, fileList.size() - 1, onInputChange);
	
	//attaching the tracker to output image window
	createTrackbar("Filter Choose", "Output Image", &filterType, 10, onFilterChange);
	createTrackbar("Size Choose", "Output Image", &filterSize, 3, onSizeChange);

	onInputChange(0 , 0);
	onFilterChange(0 , 0);
	onSizeChange(0 , 0);

	waitKey(0);
	return 0;
}
