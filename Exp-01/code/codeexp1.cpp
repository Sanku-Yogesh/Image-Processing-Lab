#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <cmath>
using namespace std;
#define SIN(x) sin(x * 3.141592653589 / 180)
#define COS(x) cos(x * 3.141592653589 / 180)

//BMP file header
struct BITMAP_header{
	char name[2]; // this should be equal to BM
	unsigned int size; // using int because sizeof(int)=4 in my machine.
	int garbage; //this is not required
	unsigned int image_offset; //offset from where image starts in file.
};

//BMP info header
struct DIB_header{
	unsigned int header_size; //size of info header.
	unsigned int width; //Horizontal width of bitmap in pixels
	unsigned int height; //Vertical height of bitmap in pixels
	unsigned short int colorplanes; //Number of Planes (=1)
	unsigned short int bitsperpixel; //Bits per Pixel used to store palette information.
	unsigned int compression; //Type of Compression
	unsigned int image_size; //(compressed) Size of Image
	unsigned int temp[4]; //XpixelsPerM,YpixelsPerM,Colors Used,Important Colors
};


//colortable for grayscale images
struct table{
	unsigned char blue;
	unsigned char green;
	unsigned char red;
	unsigned char unwanted; //justed added to make struct of size 4 bytes
};

//struct for storing gray values
struct Gray{
	unsigned char value;
};

//struct for storing rgb values
struct RGB{
	unsigned char blue;
	unsigned char green;
	unsigned char red;
};

//allocating struct to read images(color or gray)
template <typename T>
struct Image{
	int height;
	int width;
	int bpp;
	T **rgbORgray;
};

//free the memory
template <typename T>
void freeImage(struct Image<T> pic){
	for(int i = pic.height-1;i>=0;i--) free(pic.rgbORgray[i]); //row destruction
	free(pic.rgbORgray);
}

//fuction to read pixel array from images
template <typename T>
struct Image<T> readbmpfile(FILE *fp,int height,int width, int bpp){
	
	struct Image<T> pic;
	//dymanic memory allocation for double pointer
	pic.rgbORgray = (T **) malloc(height*sizeof(void*));
	pic.height = height;
	pic.width = width;
	pic.bpp = bpp;
	
	
	//calculating the pixel array width = image width + padding
	int bytestoread = ((bpp*width+31)/32);
	bytestoread = 4*bytestoread;
	int numOfrgb = bytestoread/sizeof(T) + !((bytestoread)%sizeof(T)==0);
	/*
	//calculating the pixel array width = image width + padding
	int bytestoread = ((bpp*width+31)/32)*4;
	int numOfrgb = bytestoread/sizeof(T) + 1;
	*/
	
	for(int i = pic.height-1;i>=0;i--){
		pic.rgbORgray[i] = (T*) malloc(numOfrgb*sizeof(T));
		//reading pixel array
		fread(pic.rgbORgray[i],1,bytestoread,fp);
	}
	return pic;
}

//conversion of RGB to grayscale
unsigned char grayscale(struct RGB rgb){
	return (0.3*rgb.red + 0.6*rgb.green + 0.1*rgb.blue);
}

//function to convert rgb to grayscale
// struct Image<RGB> to struct Image <Gray>
struct Image<Gray> RGBImageToGrayscale(struct Image<RGB> pic){
	int height = pic.height;
	int width = pic.width;
	int bpp = pic.bpp;
	//new pixel array to store grayscale image
	struct Image<Gray> grayPic;
	grayPic.height = height;
	grayPic.width = width;
	//dynamic allocation
	grayPic.rgbORgray = (struct Gray**) malloc(height*sizeof(void*));
	
	
	//pixel array width
	int bytestoread = ((bpp*width+31)/32);
	bytestoread = 4*bytestoread;
	int numOfrgb = bytestoread/sizeof(struct Gray) + !((bytestoread)%sizeof(struct Gray)==0);
	
	/*
	//pixel array width
	int bytestoread = ((bpp*width+31)/32)*4;
	int numOfrgb = bytestoread/sizeof(struct Gray) + 1;
	*/
	
	for(int i = height-1;i>=0;i--){
		grayPic.rgbORgray[i] = (struct Gray*) malloc(numOfrgb*sizeof(struct Gray));

	}
	//conversion
	for(int i=0;i<pic.height;i++){
		for(int j=0;j<pic.width;j++){
			grayPic.rgbORgray[i][j].value = grayscale(pic.rgbORgray[i][j]);
		}
	}
	return grayPic;
}

//function to flip the image
template <typename T>
struct Image<T> flipImage(struct Image<T> pic){
	// swap the height and width for flipping the image
	int height = pic.width;
	int width = pic.height;
	int bpp = pic.bpp;
	
	//defining the new pixel array to store the flipped image
	struct Image<T> flipPic;
	flipPic.height = height;
	flipPic.width = width;
	
	//dynamic allocating the pixel array
	flipPic.rgbORgray = (T**) malloc(height*sizeof(void*));
	

	//pixel array width = image width + padding
	int bytestoread = ((bpp*width+31)/32);
	bytestoread = 4*bytestoread;
	int numOfrgb = bytestoread/sizeof(T) + !((bytestoread)%sizeof(T)==0);
	
	/*
	//pixel array width = image width + padding
	int bytestoread = ((bpp*width+31)/32)*4;
	int numOfrgb = bytestoread/sizeof(T) + 1;
	*/
	
	for(int i = height-1;i>=0;i--){
		flipPic.rgbORgray[i] = (T*) malloc(numOfrgb*sizeof(T));

	}
	
	//transposing the pixel array
	for(int i = 0; i <height ;i++){
		for(int j=0;j<width;j++){
			flipPic.rgbORgray[i][j] = pic.rgbORgray[j][i];
		}
	}
	return flipPic;
}

//function to rotate the image by 90 degrees
template <typename T>
struct Image<T> rotate90(struct Image<T> pic){
	//swapping the height and width
	int height = pic.width;
	int width = pic.height;
	int bpp = pic.bpp;
	
	//defining the new pixel array to store the rotated image
	struct Image<T> rotatePic;
	rotatePic.height = height;
	rotatePic.width = width;
	
	//dynamic allocation of pixel array
	rotatePic.rgbORgray = (T**) malloc(height*sizeof(void*));
	
	//pixel array width = image width + padding
	int bytestoread = ((bpp*width+31)/32);
	bytestoread = 4*bytestoread;
	int numOfrgb = bytestoread/sizeof(T) + !((bytestoread)%sizeof(T)==0);
	
	/*
	//pixel array width = image width + padding
	int bytestoread = ((bpp*width+31)/32)*4;
	int numOfrgb = bytestoread/sizeof(T) + 1;
	*/
	for(int i = height-1;i>=0;i--){
		rotatePic.rgbORgray[i] = (T*) malloc(numOfrgb*sizeof(T));
	}
	
	//rotating the pixel array
	for(int i = height-1; i >=0 ;i--){
		for(int j=0;j<width;j++){
			rotatePic.rgbORgray[i][j] = pic.rgbORgray[j][height-1-i];
		}
	}
	return rotatePic;
}


// input : coordinates in image axis 
// output : rotated coordinates in image axis
// this function first converts image axis coordinates to normal axis coordinates
// and computes the rotated coordinates in normal axis 
// and remaps to image coordinates
void rotate(float *imagei,float *imagej, int imagei_pivot, int imagej_pivot,int angle,int height,int width)
{
        //mapping to normal axis
        float x = *imagej;
        float y = height-*imagei;
        int x_pivot = imagej_pivot;
        int y_pivot = height-imagei_pivot;
        
        // Shifting the pivot point to the origin
        // and the given points accordingly
        int x_shifted = x - x_pivot;
        int y_shifted = y - y_pivot;
 
        // Calculating the rotated point co-ordinates
        // and shifting it back
        x = x_pivot + (x_shifted * COS(angle) - y_shifted * SIN(angle));
        y = y_pivot + (x_shifted * SIN(angle) + y_shifted * COS(angle));

	//remapping to image axis     
        *imagej = x;
        *imagei = height-y;  
}

//checking whether the coordinate is inside the rotated image or not
int isin(int height, int width, int i,int j){
	float c1 = (i/(float)height) + (j/(float)width);
	float c2 = (i/(float)height) - (j/(float)width);
	if(c1<=0.5 || c1>=1.5) return 0;
	if(c2<=-0.5 || c2>=0.5) return 0;
	return 1;
}

//function for rotating the image by 45 degrees
template<typename T>
struct Image<T> rotate45(struct Image<T> pic){
	
	//code for computing the final height and final width
	int height = pic.height,width=pic.width,bpp=pic.bpp;
	int middlei = pic.height/2;
	int middlej = pic.width/2;
	float I,J;
	
	//min j comes when we take 0,0
	I=0;J=0;
	rotate(&I,&J,middlei,middlej,45,height,width);
	int minj = J;
	
	//min i comes when we take 0,W
	I=0;J=width;
	rotate(&I,&J,middlei,middlej,45,height,width);
	float mini = I;
	
	//max i comes when we take h,0
	I=height;J=0;
	rotate(&I,&J,middlei,middlej,45,height,width);
	float maxi = I;
	
	//max j comes when we take h,w
	I=height;J=width;
	rotate(&I,&J,middlei,middlej,45,height,width);
	float maxj = J;
	
	//final width and height
	int modifiedwidth = round(abs(minj)+maxj);
	int modifiedheight = round(abs(mini)+maxi);
	
	//new pixel array to store the rotated image
	struct Image<T> rotatePic;
	
	//2d array to note the visited pixels
	int **vis;
	//dynamic allocating
	rotatePic.rgbORgray = (T**) calloc(modifiedheight,sizeof(void*));
	vis = (int **) calloc(modifiedheight,sizeof(void*));
	
	rotatePic.height = modifiedheight;
	rotatePic.width = modifiedwidth;
	//pixel array width
	
	
	int bytestoread = ((bpp*modifiedwidth+31)/32);
	bytestoread = 4*bytestoread;
	int numOfrgb = bytestoread/sizeof(T) + !((bytestoread)%sizeof(T)==0);
	
	
	/*
	int bytestoread = ((bpp*modifiedwidth+31)/32)*4;
	int numOfrgb = bytestoread/sizeof(T) + 1;
	*/
	
	for(int i = modifiedheight-1;i>=0;i--){
		rotatePic.rgbORgray[i] = (T*) calloc(numOfrgb,sizeof(T));
		vis[i] = (int *) calloc(numOfrgb,sizeof(int));
	}
	
	//rotating the pixel array
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			I=i;J=j;
			//rotated coordinates
			rotate(&I,&J,middlei,middlej,45,height,width);
			I=round(abs(mini)+I);J=round(abs(minj)+J);
			rotatePic.rgbORgray[(int)I][(int)J] = pic.rgbORgray[i][j];
			vis[(int)I][(int)J] = 1; // noting the visited coordinate
		}
	}
	
	//interpolating
	int k=0,l=0; int directions[8][2] = {{-1,0},{0,-1},{1,0},{0,1},{-1,-1},{-1,1},{1,-1},{1,1}};
	for(int i=0;i<modifiedheight;i++){
		for(int j=0;j<modifiedwidth;j++){
			//if it is black region skip
			if(vis[i][j] == 1 || isin(modifiedheight,modifiedwidth,i,j)!=1 ) continue;
			for(auto d:directions){
				// checking in 8 directions
				k = ((i+d[0])>=0 && (i+d[0])<modifiedheight) ? i+d[0] : i;
				l = ((j+d[1])>=0 && (j+d[1])<modifiedwidth) ? j+d[1] : j;
				//if one of them is visited
				if(vis[k][l] ==1){
					rotatePic.rgbORgray[i][j] = rotatePic.rgbORgray[k][l];
					break;
				}
			}
		}
	}
	return rotatePic;
}

//function to scale the pixel array
template <typename T>
struct Image<T> scale(struct Image<T> pic,float scalei, float scalej){
	// final width and height
	int width = scalej*pic.width;
	int height = scalei*pic.height;
	int bpp = pic.bpp;
	//new pixel array
	struct Image<T> scalePic;
	int **vis;
	scalePic.rgbORgray = (T**) malloc(height*sizeof(void*));
	vis = (int **) calloc(height,sizeof(void*));
	scalePic.height = height;
	scalePic.width = width;
	
	//pixel array width = image width + padding
	int bytestoread = ((bpp*width+31)/32);
	bytestoread = 4*bytestoread;
	int numOfrgb = bytestoread/sizeof(T) + !((bytestoread)%sizeof(T)==0);
	
	/*
	int bytestoread = ((bpp*width+31)/32)*4;
	int numOfrgb = bytestoread/sizeof(T) + 1;
	*/
	
	for(int i = height-1;i>=0;i--){
		scalePic.rgbORgray[i] = (T*) malloc(numOfrgb*sizeof(T));
		vis[i] = (int *) calloc(numOfrgb,sizeof(int));
	}
	//scaling the pixel array
	int k=0,l=0; 
	for(int i = 0; i <pic.height ;i++){
		for(int j=0;j<pic.width;j++){
			k = scalei*i;l=scalej*j;
			scalePic.rgbORgray[k][l] = pic.rgbORgray[i][j];
			vis[k][l] =1;
		}
	}
	//interpolation
	int directions[8][2] = {{-1,0},{0,-1},{1,0},{0,1},{-1,-1},{-1,1},{1,-1},{1,1}};
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			
			if(vis[i][j] == 1) continue;
			for(auto d:directions){
				// checking in 8 directions
				k = ((i+d[0])>=0 && (i+d[0])<height) ? i+d[0] : i;
				l = ((j+d[1])>=0 && (j+d[1])<width) ? j+d[1] : j;
				//if one of them is visited
				if(vis[k][l] ==1){
					scalePic.rgbORgray[i][j] = scalePic.rgbORgray[k][l];
					break;
				}
				// else scalePic.rgbORgray[i][j] = scalePic.rgbORgray[k][l];
			}
		}
	}	
	return scalePic;
}

//function to write bmp file
template <typename T>
void writebmp(struct BITMAP_header header , struct DIB_header dibheader, struct Image<T> pic,char* filename){
	//opening new file in writing mode
	FILE *fpw = fopen(filename,"w");
	dibheader.height = pic.height;
	dibheader.width = pic.width;
	dibheader.image_size = 3*pic.height*pic.width;
	int bpp = dibheader.bitsperpixel;
	//writing headers into file 
	fwrite(header.name,2,1,fpw);
	fwrite(&header.size,3*sizeof(int),1,fpw);
	fwrite(&dibheader,sizeof(struct DIB_header),1,fpw);
	
	//pixel array width = image width + padding
	
		
	int bytestoread = ((bpp*pic.width+31)/32);
	bytestoread = 4*bytestoread;
	int numOfrgb = bytestoread/sizeof(T) + !((bytestoread)%sizeof(T)==0);
	
	/*
	int bytestoread = ((bpp*pic.width+31)/32)*4;
	int numOfrgb = bytestoread/sizeof(T) + 1;
	*/
	//only for grayscale images
	if(dibheader.bitsperpixel ==8){
		dibheader.image_size = pic.height*pic.width; //changing for grayscale images
		struct table colortable[256];
		
		//filling the colortable as per definition of grayscale
		for (int i=0;i<256;i++){
			colortable[i].blue = i;
			colortable[i].green = i;
			colortable[i].red = i;
		}
		
		//writing colortable into file
		fwrite(colortable,256*sizeof(table),1,fpw);
	}
	
	//writing pixel array into file
	//note : we are writing the pixel array from h-1
	for(int i = pic.height-1;i>=0;i--){
		fwrite(pic.rgbORgray[i],bytestoread,1,fpw);
	}
}


//geometrical transformation function
template <typename T> 
void geometrical(struct BITMAP_header header , struct DIB_header dibheader, struct Image<T> pic,char* flipname,char *rot90name,char *rot45name,char *scalename){
	
	//image flipping
	struct Image<T> flipPic = flipImage<T>(pic);    //calling function to flip the image
	writebmp<T>(header,dibheader,flipPic,flipname); //writing pixel array to image
	freeImage<T>(flipPic); //memory freeing
	
	//image rotating 90 degree
	struct Image<T> rotate90Pic = rotate90<T>(pic);     // calling function to rotate the image by 90 degree
	writebmp<T>(header,dibheader,rotate90Pic,rot90name);//writing pixel array to image
	freeImage<T>(rotate90Pic);//memory freeing
	
	//image rotation by 45 degrees
	struct Image<T> rotate45Pic = rotate45<T>(pic);     // calling function to rotate the image by 45 degree
	writebmp<T>(header,dibheader,rotate45Pic,rot45name);//writing pixel array to image
	
	struct Image<T> scalePic = scale<T>(pic,2,2);      //calling function to scale the image
	writebmp(header,dibheader,scalePic,scalename); //writing pixel array to image
}

int readbmp(char *filename,char *grayname,char *flipname,char *rot90name,char *rot45name, char*scalename){
	FILE *fp = fopen(filename,"rb"); //rb means read in binary
	
	struct BITMAP_header header; //bmp header
	struct DIB_header dibheader; //dib header
	
	fread(header.name,2,1,fp); 
	fread(&header.size,3*sizeof(int),1,fp);          //reading remaining 3* 4 bytes from the address of header.size
	fread(&dibheader,sizeof(struct DIB_header),1,fp); // reading dibheader

	//checking whether file is BMP or not
	if(header.name[0]!='B' || header.name[1]!='M'){
		cout << "not a BMP file" << endl;
		return -1;
	}
	
	cout << endl << "HEADER INFORMATION ";
	cout << endl;
	// Separator
	cout << "__________________________________________________" << endl; 
	// Header information
	cout << "File Type: " << "\t\t\t | " << header.name[0] << header.name[1] << "\t\t |" << endl;
	cout << "File Size (in bytes): " << "\t\t | " << header.size << "\t |" << endl;
	cout << "Reserved 1: " << "\t\t\t | " << header.garbage << "\t\t |" << endl;
	//cout << "Reserved 2: " << "\t\t\t | " << bm_head.reserved2 << "\t\t |" << endl;
	cout << "Data Offset: " << "\t\t\t | " << header.image_offset << "\t\t |" << endl;
	cout << "Header Size (in bytes): " << "\t | " << dibheader.header_size << "\t\t |" << endl;
	cout << "Image Width: " << "\t\t\t | " << dibheader.width << "\t\t |" << endl;
	cout << "Image Height: " << "\t\t\t | " << dibheader.height << "\t\t |" << endl;
	cout << "Number of Planes: " << "\t\t | " << dibheader.colorplanes << "\t\t |" << endl;
	cout << "Bits per Pixel: " << "\t\t | " << dibheader.bitsperpixel << "\t\t |" << endl;
	cout << "Compression Type: " << "\t\t | " << dibheader.compression << "\t\t |" << endl;
	cout << "Image Size (in bytes): " << "\t\t | " << dibheader.image_size << "\t |" << endl;
	cout << "Resolution in x-direction: " << "\t | " << dibheader.temp[0] << "\t\t |" << endl;
	cout << "Resolution in y-direction: " << "\t | " << dibheader.temp[1] << "\t\t |" << endl;
	cout << "Colors Used: " << "\t\t\t | " << dibheader.temp[2] << "\t\t |" << endl;
	cout << "Colors Important: " << "\t\t | " << dibheader.temp[3] << "\t\t |" << endl;
	
	fseek(fp,header.image_offset,SEEK_SET); // moving to the pixel array
	
	//checking whether image is gray or colored
	if(dibheader.bitsperpixel ==8){
		//reading pixel array
		struct Image<struct Gray> pic  = readbmpfile<Gray>(fp,dibheader.height,dibheader.width,dibheader.bitsperpixel);
		
		//calling geometrical transformation function
		geometrical<struct Gray>(header,dibheader,pic,flipname,rot90name,rot45name,scalename);
		
		freeImage(pic);
	}
	else{
		struct Image<struct RGB> pic = readbmpfile<RGB>(fp,dibheader.height,dibheader.width,dibheader.bitsperpixel);
		
		//call to change to gray scale
		struct Image<struct Gray> grayPic = RGBImageToGrayscale(pic);
		struct BITMAP_header gray_header = header;
		struct DIB_header gray_dibheader = dibheader;
		//modifing the necessary fields
		gray_dibheader.bitsperpixel = 8;
		gray_header.size = (dibheader.height*dibheader.width)+header.size+256*sizeof(table);
		gray_header.image_offset += 256*sizeof(table);
		
		
		writebmp(gray_header,gray_dibheader,grayPic,grayname);//writing into file
		freeImage(grayPic);
		
		geometrical<struct RGB>(header,dibheader,pic,flipname,rot90name,rot45name,scalename);
		freeImage(pic);
	}
	
	fclose(fp);
	return 1;
}


int main(int argc,char *argv[]){
	//for color
	if(argc==7) readbmp(argv[1],argv[2],argv[3],argv[4],argv[5],argv[6]);
	//for grayscale
	if(argc==6) readbmp(argv[1],argv[1],argv[2],argv[3],argv[4],argv[5]);
	return 0;
}

