#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <memory.h>
// #include <ctype.h>
// #include <omp.h>
// #include <amp.h>

#include <sys/types.h>
#ifdef _WIN32
typedef unsigned __int64 uint64_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int8 uint8_t;

#define rand_r(seed) rand()
#else
typedef u_int64_t uint64_t;
typedef u_int32_t uint32_t;
typedef u_int16_t uint16_t;
typedef u_int8_t  uint8_t;
#endif

typedef uint8_t uchar;
typedef uint16_t ushort;
typedef uint32_t uint;

//#define USE_OMP 0
#ifndef USE_SSE
#  define USE_SSE 0
#endif
#ifndef USE_OMP
#  define USE_OMP 0
#endif

#if USE_SSE == 1
#  include <emmintrin.h>
#endif
#if USE_OMP == 1
#  include <omp.h>
#endif

#define N_MAX_OBJECTS 100
#define DIST(a,b)    (abs(a.x - b.x) + abs(a.y - b.y))
#define PI (3.14159265358979323846)
#define PHI (0.61803398874989484820458683436564)
#define EPSILON (1e-12)
#define FZERO (0.000001)
#define NOISE_BUFF_LEN (100000)
#define SATURATE(level, white_level) ((level)<0?0:((level)>(white_level)?(white_level):(level)))


#define MAX(a,b)    (((a) > (b)) ? (a) : (b))
#define MIN(a,b)    (((a) < (b)) ? (a) : (b))

enum EMethod {
	ObjectSearch_BestIntegral = 0,
	ObjectSearch_BestContrast = 1
};

struct sSize
{
	int Width, Height;
	int Stride;
};

struct img_desc_t
{
	int width, height;
	int stride;
};

struct SAreaInfo
{
	int x, y;
	double S;
};

struct SObjectInfo
{
	double		Certainty;
	double		AvgX, AvgY, MaxX, MaxY;
	int			Left, Top, Right, Bottom;
	double		Diameter, Area, Volume;
	double		AvgSignal, StdevSignal, MaxSignal;
};


#ifdef __cplusplus
extern "C" {
#endif
	/*declaration functions for image_fun.c*/
	int			add(uchar *firstImg, uchar *secondImg, int width, int height, int strideFir, int strideSec, uchar *outImg, int strideOut, int koef, int bpp);
	int			multOnMask(uchar *img, int width, int height, int stride, uchar *mask, int bpp);
	int			smooth(uchar *img, int stride_in, int width, int height, uchar *outImg, int stride_out, int d, int bpp);
	int			contrast(uchar *img, int stride_in, int width, int height, uchar *outImg, int stride_out, int dMin, int dMax, uint *mask, int bpp);
	int			difference(uchar *img, uchar *lastImg, uchar *outImg, int width, int strideIn, int height, int strideOut, int bpp);
	int			applyCalibration(uchar *imgSrc, int strideSrc, int width, int height, uchar *calibration, uchar *out_img, int bpp);
	int			calibrationFrom(uchar *source, int strideIn, int width, int height, uchar *outImg, int strideOut, int smoothed, int bpp);


	/*declaration functions for matrix.cpp*/
	double		fitToBiPoly(const double *Xs, const double *Ys, const double *Zs, int count, double *coefficients, int nCoefs, int *countCoeffs);
	void		square(double *data, int rows, int cols, double *out, int *rows_out, int *cols_out);
	void		transpose(double *data, int rows, int cols, double *out, int *rowsOut, int *colsOut);
	void		multBy(double *data, int rows, int cols, double *other, int rowsOth, int colsOth, double *out, int *rowsOut, int *colsOut);
	double		calcBiPolyValue(double *coefficients, int countCoeffs, double x, double y);

	/*declaration functions for image_info.c*/
	void		fitInt(int min, int max, int *val);
	void		quickSort(struct SAreaInfo *items, int count);
	int			imageInfo(uchar *img_src, int width, int height, int stride, double* avg, double* min, double* max, double* stdev, uint bpp);
	int			integralFrom(uchar *source, int width, int height, int stride, uint *Integral, uint bpp);
	int			integralSum(const uint *integralBuffer, struct sSize size, int left, int top, int right, int bottom, int *count);
	double		localDifference(const uint *integral, struct sSize size, int x, int y, int innerD, int outerD, uint* IntegralMask, int *count);
	double		integralAvg(const uint *imgSrc, struct sSize size, int left, int top, int right, int bottom);
	int			localThreshold(uchar *processed, int bpp, struct sSize size, int *left, int *top, int *right, int *bottom);
	int			localizeFeature(uint *integral, struct sSize size, int *left, int *top, int *right, int *bottom);
	int			energyDistribution(uchar *img, int width, int height, int stride, int bpp, double x0, double y0, const double *R, double *I, int length);
	int			powerDistribution(uchar *img, int width, int height, int stride, int bpp, double x0, double y0, const double *R, double *I, int length);

	/*declaration functions for graphics.c*/
	int			drawObject(uchar *img, int width, int height, int stride, int x, int y, int signal, int radius, int bpp);
	int			addMarker(uchar *img, int width, int height, int stride, double x, double y, int left, int top, int right, int bottom, int bpp);
	int			addNoiseUniform(uchar *img, int width, int height, int stride, int background, int noise_min, int noise_max, int bpp);
	int			addNoiseNorm(uchar *img, int width, int height, int stride, int m, double s, int bpp);

	/*declaration functions for sampling.c*/
	void downsample(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, int bpp, int factor);
	void upsample(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, int bpp, int factor);
	
	/*declaration functions for convolve.c*/
	void convolve(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, int bpp, const double *kernel, int kwidth, int kheight);

	/*declaration functions for search_objects.c*/
	int searchObjects(uchar *processed, int width, int height, int stridePr, uchar *source, int strideSrc, uchar* mask, uint* integralMask, int bpp, uint method, int minSize, int maxSize, double minCertainty,
		              int nMaxObjects, struct SObjectInfo *objects_info, int *count);

	int searchObjectsForMultiROI(int countROI, int *ROI, uchar *Processed, int width, int height, int stridePr, uchar *source, int strideSrc, uchar* mask, uint* integralMask, int bpp, uint method,
		                         int minSize, int maxSize, double minCertainty, struct SObjectInfo *objects);

#ifdef __cplusplus
}
#endif
