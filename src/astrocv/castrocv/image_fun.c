#include "astrocv.h"

static void add_8b(uchar *firstImg, uchar *secondImg, uchar *outImg, int width, int height, int strideA, int strideB, int strideC, int koef)
{
	const int white_level = (1 << 8) - 1;
	int i;
	#if  USE_OMP == 1
	#pragma omp parallel for private(i)
	#endif //  USE_OMP
	for (i = 0; i < height; i++)
	{
		const uchar *pA = (const uchar*)((char*)firstImg + i*strideA);
		const uchar *pB = (const uchar*)((char*)secondImg + i*strideB);
		uchar *pC = (uchar*)((char*)outImg + i*strideC);
		int j;
		for (j = 0; j < width; j++)
		{
			int val = (int)(*pA++) + koef * (int)(*pB++);
			(*pC++) = SATURATE(val, white_level);
		}
	}
}


static void add_16b(uchar *firstImg, uchar *secondImg, uchar *outImg, int width, int height, int strideA, int strideB, int strideC, int koef)
{
	const int white_level = (1 << 16) - 1;
	int i;
	#if  USE_OMP == 1
	#pragma omp parallel for private(i)
	#endif //  USE_OMP
	for (i = 0; i < height; i++)
	{
		const ushort *pA = (const ushort*)((char*)firstImg + i*strideA);
		const ushort *pB = (const ushort*)((char*)secondImg + i*strideB);
		ushort *pC = (ushort*)((char*)outImg + i*strideC);
		int j;
		for (j = 0; j < width; j++)
		{
			int val = (int)(*pA++) + koef * (int)(*pB++);
			(*pC++) = SATURATE(val, white_level);
		}
	}
}


static void add_32b(uchar *firstImg, uchar *secondImg, uchar *outImg, int width, int height, int strideA, int strideB, int strideC, int koef)
{
	const uint white_level = (uint)((1LL << 32) - 1);
	int i;
	#if  USE_OMP == 1
	#pragma omp parallel for private(i)
	#endif //  USE_OMP
	for (i = 0; i < height; i++)
	{
		const uint *pA = (const uint*)((char*)firstImg + i*strideA);
		const uint *pB = (const uint*)((char*)secondImg + i*strideB);
		uint *pC = (uint*)((char*)outImg + i*strideC);
		int j;
		for (j = 0; j < width; j++)
		{
			uint val = (int)(*pA++) + koef * (int)(*pB++);
			(*pC++) = SATURATE(val, white_level);
		}
	}
}


/*add: out_img(C)=first_img(A) + second_img(B)*/
int add(uchar *firstImg, uchar *secondImg, int width, int height, int strideFir, int strideSec, uchar *outImg, int strideOut, int koef, int bpp)
{
	switch (bpp)
	{
	case 8:
		add_8b(firstImg, secondImg, outImg, width, height, strideFir, strideSec, strideOut, koef);
		return 0;
	case 16:
		add_16b(firstImg, secondImg, outImg, width, height, strideFir, strideSec, strideOut, koef);
		return 0;
	case 32:
		add_32b(firstImg, secondImg, outImg, width, height, strideFir, strideSec, strideOut, koef);
		return 0;
	default:
		return -1;
	}
}

static void mult_8b(uchar *img, int width, int height, int stride, uchar *mask)
{
	const int white_level = (1 << 8) - 1;
	int i;
#if  USE_OMP == 1
#pragma omp parallel for private(i)
#endif //  USE_OMP
	for (i = 0; i < height; i++)
	{
		uchar *pA = img + i*stride;
		uchar *pB = mask + i*width;
		//uchar *pC = (uchar*)((char*)img + i*stride);
		int j;
		for (j = 0; j < width; j++)
		{
			int val = (int)(*pA) * (int)(*pB++);
			(*pA++) = SATURATE(val, white_level);
		}
	}
}

static void mult_16b(uchar *img, int width, int height, int stride, uchar *mask)
{
	const int white_level = (1 << 16) - 1;
	int i;
#if  USE_OMP == 1
#pragma omp parallel for private(i)
#endif //  USE_OMP
	for (i = 0; i < height; i++)
	{
		const ushort *pA = (const ushort*)((char*)img + i*stride);
		const uchar *pB = (const uchar*)((char*)mask + i*width);
		ushort *pC = (ushort*)((char*)img + i*stride);
		int j;
		for (j = 0; j < width; j++)
		{
			int val = (int)(*pA++) * (int)(*pB++);
			(*pC++) = SATURATE(val, white_level);
		}
	}
}


static void mult_32b(uchar *img, int width, int height, int stride, uchar *mask)
{
	const uint white_level = (uint)((1LL << 32) - 1);
	int i;
#if  USE_OMP == 1
#pragma omp parallel for private(i)
#endif //  USE_OMP
	for (i = 0; i < height; i++)
	{
		const uint *pA = (const uint*)((char*)img + i*stride);
		const uchar *pB = (const uchar*)((char*)mask + i*width);
		uint *pC = (uint*)((char*)img + i*stride);
		int j;
		for (j = 0; j < width; j++)
		{
			uint val = (int)(*pA++) * (int)(*pB++);
			(*pC++) = SATURATE(val, white_level);
		}
	}
}

/*mult: first_img(C)=img(A) * mask(B)*/
int multOnMask(uchar *img, int width, int height, int stride, uchar *mask, int bpp)
{
	switch (bpp)
	{
	case 8:
		mult_8b(img, width, height, stride, mask);
		return 0;
	case 16:
		mult_16b(img, width, height, stride, mask);
		return 0;
	case 32:
		mult_32b(img, width, height, stride, mask);
		return 0;
	default:
		return -1;
	}
}



static void smooth_8b(uint *integral, uchar *outImg, struct sSize size, int d)
{
	int y;
	if (d < 2)
		d = 2;
#if USE_OMP == 1
#pragma omp parallel for private(y)
#endif
	for (y = 0; y < size.Height; y++)
	{
		uchar *dst = (uchar*)(outImg + y*size.Stride);
		int x;
		for (x = 0; x < size.Width; x++)
		{
			double val = integralAvg(integral, size, x - d, y - d, x + d, y + d);
			*(dst++) = (uchar)SATURATE(val, (double)0xff);
		}
	}
}

static void smooth_16b(uint *integral, uchar *outImg, struct sSize size, int d)
{
	int y;
	if (d < 2)
		d = 2;
#if USE_OMP == 1
#pragma omp parallel for private(y)
#endif
	for (y = 0; y < size.Height; y++)
	{
		ushort *dst = (ushort*)(outImg + y*size.Stride);
		int x;
		for (x = 0; x < size.Width; x++)
		{
			double val = integralAvg(integral, size, x - d, y - d, x + d, y + d);
			*(dst++) = (ushort)SATURATE(val, (double)0xffff);
		}
	}
}

static void smooth_32b(uint *integral, uchar *outImg, struct sSize size, int d)
{
	int y;
	if (d < 2)
		d = 2;
#if USE_OMP == 1
#pragma omp parallel for private(y)
#endif
	for (y = 0; y < size.Height; y++)
	{
		uint *dst = (uint*)(outImg + y*size.Stride);
		int x;
		for (x = 0; x < size.Width; x++)
		{
			double val = integralAvg(integral, size, x - d, y - d, x + d, y + d);
			*(dst++) = (uint)SATURATE(val, (double)0xffffffff);
		}
	}
}

int smooth(uchar *img, int stride_in, int width, int height, uchar *outImg, int stride_out, int d, int bpp)
{
	uint *integral;
	struct sSize size;
	size.Width = width;
	size.Stride = stride_out;
	size.Height = height;

	integral = (uint*)malloc(size.Width * size.Height * sizeof(int));
	integralFrom(img, width, height, stride_in, integral, bpp);

	switch (bpp)
	{
	case 8:
	{
		smooth_8b(integral, outImg, size, d);
		free(integral);
		return 0;
	}
	case 16:
	{
		smooth_16b(integral, outImg, size, d);
		free(integral);
		return 0;
	}
	case 32:
	{
		smooth_32b(integral, outImg, size, d);
		free(integral);
		return 0;
	}
	default:
		free(integral);
		return -1;
	}
}


static void spatial_8b(uint *integral, uint *integralMask, uchar *outImg, struct sSize size, int dMin, int dMax)
{
	int y;
	#if USE_OMP == 1
	#pragma omp parallel for private(y)
	#endif
	for (y = 0; y < size.Height; y++)
	{
		uchar *dst = (uchar*)(outImg + y*size.Stride);
		int x, cnt;
		for (x = 0; x < size.Width; x++)
		{
			double val = localDifference(integral, size, x, y, dMin, dMax, integralMask, &cnt);
			val *= (double)cnt;
			*(dst++) = (uchar)SATURATE(val, (double)0xff);
		}
	}
}

static void spatial_16b(uint *integral, uint *integralMask, uchar *outImg, struct sSize size, int dMin, int dMax)
{
	int y;
	#if USE_OMP == 1
	#pragma omp parallel for private(y)
	#endif
	for (y = 0; y < size.Height; y++)
	{
		ushort *dst = (ushort*)(outImg + y*size.Stride);
		int x, cnt;
		for (x = 0; x < size.Width; x++)
		{
			double val = localDifference(integral, size, x, y, dMin, dMax, integralMask, &cnt);
			val *= (double)cnt;
			*(dst++) = (ushort)SATURATE(val, (double)0xffff);
		}
	}
}

static void spatial_32b(uint *integral, uint *integralMask, uchar *outImg, struct sSize size, int dMin, int dMax)
{
	int y;
	#if USE_OMP == 1
	#pragma omp parallel for private(y)
	#endif
	for (y = 0; y < size.Height; y++)
	{
		uint *dst = (uint*)(outImg + y*size.Stride);
		int x, cnt;
		for (x = 0; x < size.Width; x++)
		{
			double val = localDifference(integral, size, x, y, dMin, dMax, integralMask, &cnt);
			val *= (double)cnt;
			*(dst++) = (uint)SATURATE(val, (double)0xffffffff);
		}

	}
}

int contrast(uchar *img, int stride_in, int width, int height, uchar *outImg, int stride_out, int dMin, int dMax, uint *integralMask, int bpp)
{
	struct sSize size;
	uint *Integral;

	if (dMin < 1)
		dMin = 1;
	if (dMax < dMin + 2)
		dMax = dMin + 2;
	size.Width = width;
	size.Stride = stride_out;
	size.Height = height;

	Integral = (uint *)malloc(size.Width * size.Height * sizeof(int));
	integralFrom(img, width, height, stride_in, Integral, bpp);
	switch (bpp)
	{
	case 8:
	{
		spatial_8b(Integral, integralMask, outImg, size, dMin, dMax);
		free(Integral);
		return 0;
	}
	case 16:
	{
		spatial_16b(Integral, integralMask, outImg, size, dMin, dMax);
		free(Integral);
		return 0;
	}
	case 32:
	{
		spatial_32b(Integral, integralMask, outImg, size, dMin, dMax);
		free(Integral);
		return 0;
	}
	default:
		free(Integral);
		return -1;
	}
}


static void difference_8b(uchar *imgSrc, uchar *lastImg, uchar *outImg, struct sSize size, int strideOut)
{
	int i;
	#if USE_OMP == 1
	#pragma omp parallel for private(i)
	#endif
	for (i = 0; i < size.Height; i++)
	{
		const uchar *src  = (const uchar*)(imgSrc + i*size.Stride);
		const uchar *last = (const uchar*)(lastImg + i*size.Stride);
		uchar *dst = (uchar*)(outImg + i*size.Stride);
		int j;
		for (j = 0; j < size.Width; ++j)
		{
			if (*src < *last)
				*(dst++) = *(last++) - *(src++);
			else
				*(dst++) = *(src++) - *(last++);
		}
	}
}

static void difference_16b(uchar *imgSrc, uchar *lastImg, uchar *outImg, struct sSize size, int strideOut)
{
	int i;
	#if USE_OMP == 1
	#pragma omp parallel for private(i)
	#endif
	for (i = 0; i < size.Height; i++)
	{
		const ushort *src  = (const ushort*)(imgSrc + i*size.Stride);
		const ushort *last = (const ushort*)(lastImg + i*size.Stride);
		ushort *dst = (ushort*)(outImg + i*size.Stride);
		int j;
		for (j = 0; j < size.Width; ++j)
		{
			if (*src < *last)
				*(dst++) = *(last++) - *(src++);
			else
				*(dst++) = *(src++) - *(last++);
		}
	}
}

static void difference_32b(uchar *imgSrc, uchar *lastImg, uchar *outImg, struct sSize size, int strideOut)
{
	int i;
	#if USE_OMP == 1
	#pragma omp parallel for private(i)
	#endif
	for (i = 0; i < size.Height; i++)
	{
		const uint *src  = (const uint*)(imgSrc + i*size.Stride);
		const uint *last = (const uint*)(lastImg + i*size.Stride);
		uint *dst = (uint*)(outImg + i*size.Stride);
		int j;
		for (j = 0; j < size.Width; ++j)
		{
			if (*src < *last)
				*(dst++) = *(last++) - *(src++);
			else
				*(dst++) = *(src++) - *(last++);
		}
	}
}

int difference(uchar *img, uchar *lastImg, uchar *outImg, int width, int strideIn, int height, int strideOut, int bpp)
{
	struct sSize sizeSrc;
	sizeSrc.Width = width;
	sizeSrc.Stride = strideIn;
	sizeSrc.Height = height;
	switch (bpp)
	{
	case 8:
	{
		difference_8b(img, lastImg, outImg, sizeSrc, strideOut);
		return 0;
	}
	case 16:
	{
		difference_16b(img, lastImg, outImg, sizeSrc, strideOut);
		return 0;
	}
	case 32:
	{
		difference_32b(img, lastImg, outImg, sizeSrc, strideOut);
		return 0;
	}
	default:
		return -1;
	}
}


static void applyCalibration_8b(uchar *imgSrc, struct sSize size, uchar *calibration, uchar *outImg)
{
	int i;
	#if USE_OMP == 1
	#pragma omp parallel for private(i)
	#endif
	for (i = 0; i < size.Height; i++)
	{
		const uchar *src = (const uchar*)(imgSrc + i*size.Stride);
		const uchar *cal = (const uchar*)(calibration + i*size.Stride);
		uchar *dst = (uchar*)(outImg + i*size.Stride);
		int j;
		for (j = 0; j < size.Width; j++)
		{
			int val = (int)(*src++) * (int)(*cal++);
			*(dst++) = (uchar)SATURATE(val, 0xff);
		}
	}
}

static void applyCalibration_16b(uchar *imgSrc, struct sSize size, uchar *calibration, uchar *outImg)
{
	int i;
	#if USE_OMP == 1
	#pragma omp parallel for private(i)
	#endif
	for (i = 0; i < size.Height; i++)
	{
		const ushort *src = (const ushort*)(imgSrc + i*size.Stride);
		const ushort *cal = (const ushort*)(calibration + i*size.Stride);
		ushort *dst = (ushort*)(outImg + i*size.Stride);
		int j;
		for (j = 0; j < size.Width; j++)
		{
			int val = (int)(*src++) * (int)(*cal++);
			*(dst++) = (ushort)SATURATE(val, 0xffff);
		}
	}
}

static void applyCalibration_32b(uchar *imgSrc, struct sSize size, uchar *calibration, uchar *outImg)
{
	int i;
	#if USE_OMP == 1
	#pragma omp parallel for private(i)
	#endif
	for (i = 0; i < size.Height; i++)
	{
		const uint *src = (const uint*)(imgSrc + i*size.Stride);
		const uint *cal = (const uint*)(calibration + i*size.Stride);
		uint *dst = (uint*)(outImg + i*size.Stride);
		int j;
		for (j = 0; j < size.Width; j++)
		{
			int val = (int)(*src++) * (int)(*cal++);
			*(dst++) = (uint)SATURATE(val, 0xffffffff);
		}
	}
}


int applyCalibration(uchar *imgSrc, int strideSrc, int width, int height, uchar *calibration, uchar *outImg, int bpp)
{
	struct sSize sizeA;
	sizeA.Width = width;
	sizeA.Stride = strideSrc;
	sizeA.Height = height;
	switch (bpp)
	{
	case 8:
	{
		applyCalibration_8b(imgSrc, sizeA, calibration, outImg);
		return 0;
	}
	case 16:
	{
		applyCalibration_16b(imgSrc, sizeA, calibration, outImg);
		return 0;
	}
	case 32:
	{
		applyCalibration_32b(imgSrc, sizeA, calibration, outImg);
		return 0;
	}
	default:
		return -1;
	}
}


// Smoothed = 0 => no smoothing
//            1 => simple average level
//            3 => linear approximation (a0 + a1*x + a2*y)
//            6 => quadratic approximation (a0 + a1*x + a2*y + a3*x^2 + a4*x*y + a5*y^2)
//            10 => cubic approximation
//            15 => ...
//            21 => ...
static void calibrationFrom_8b(uchar *source, int strideSrc, uchar *outImg, int strideOut, struct sSize size, int smoothed)
{
	double Avg, Min, Max, Stdev;
	double add, mul;
	int count = 0, i, j;
	uchar *src = source;
	uchar *dst = outImg;

	imageInfo(source, size.Width, size.Height, strideSrc, &Avg, &Min, &Max, &Stdev, 8);
	add = (Min < 1.0) ? 1.0 : 0.0;
	mul = 1.0 / (Avg + Stdev + add);

	if (smoothed)
	{
		double *Coeffs = (double*)malloc(smoothed * smoothed * sizeof(double));
		int N = 15;
		double dy = (double)size.Height / (double)(N - 1);
		double dx = (double)size.Width / (double)(N - 1);
		int count_Coeffs;
		double err;
		double  *Xs, *Ys, *Zs;
		Xs = (double*)malloc(N * N * sizeof(double));
		Ys = (double*)malloc(N * N * sizeof(double));
		Zs = (double*)malloc(N * N * sizeof(double));
		smoothed = smoothed > N*N ? N*N : smoothed;
		for (i = 0; i < N; ++i)
		{
			int yind = MIN((int)((double)i * dy), size.Height - 1);
			for (j = 0; j < N; ++j)
			{
				int xind = MIN((int)((double)j * dx), size.Width - 1);
				Xs[count] = xind;
				Ys[count] = yind;
				Zs[count++] = src[yind * strideSrc + xind];
			}
		}
		count_Coeffs = 0;
		err = fitToBiPoly(Xs, Ys, Zs, N * N, Coeffs, smoothed, &count_Coeffs);
		if (err > 1e9)
		{
			smoothed = 0;
		}
		else
		{
#if USE_OMP == 1
#pragma omp parallel for private(i)
#endif
			for (i = 0; i < size.Height; ++i)
			{
				int j;
				for (j = 0; j < size.Width; ++j)
					dst[i*strideOut + j] = (uchar)calcBiPolyValue(Coeffs, count_Coeffs, j, i);
			}
		}
		free(Coeffs);
		free(Xs);
		free(Ys);
		free(Zs);
	}

	if (!smoothed)
	{
#if USE_OMP == 1
#pragma omp parallel for private(i)
#endif
		for (i = 0; i < size.Height; ++i)
		{
			int j;
			for (j = 0; j < size.Width; ++j)
				dst[i * strideOut + j] = (uchar)(1.0 / ((src[i * strideSrc + j] + add) * mul));
		}
	}
}

static void calibrationFrom_16b(ushort * source, int strideSrc, ushort * outImg, int strideOut, struct sSize size, int smoothed)
{
	double Avg, Min, Max, Stdev;
	double add, mul;
	int count = 0, i, j;
	ushort *src = source;
	ushort *dst = outImg;

	imageInfo((uchar*)source, size.Width, size.Height, strideSrc, &Avg, &Min, &Max, &Stdev, sizeof(*source) * 8);
	add = (Min < 1.0) ? 1.0 : 0.0;
	mul = 1.0 / (Avg + Stdev + add);

	if (smoothed)
	{
		double *Coeffs = (double*)malloc(smoothed * smoothed * sizeof(double));
		int N = 15;
		double dy = (double)size.Height / (double)(N - 1);
		double dx = (double)size.Width / (double)(N - 1);
		int count_Coeffs;
		double err;
		double  *Xs, *Ys, *Zs;
		Xs = (double*)malloc(N * N * sizeof(double));
		Ys = (double*)malloc(N * N * sizeof(double));
		Zs = (double*)malloc(N * N * sizeof(double));
		smoothed = smoothed > N*N ? N*N : smoothed;
		for (i = 0; i < N; ++i)
		{
			int yind = MIN((int)((double)i * dy), size.Height - 1);
			for (j = 0; j < N; ++j)
			{
				int xind = MIN((int)((double)j * dx), size.Width - 1);
				Xs[count] = xind;
				Ys[count] = yind;
				Zs[count++] = src[yind * strideSrc + xind];
			}
		}
		count_Coeffs = 0;
		err = fitToBiPoly(Xs, Ys, Zs, N * N, Coeffs, smoothed, &count_Coeffs);
		if (err > 1e9)
		{
			smoothed = 0;
		}
		else
		{
#if USE_OMP == 1
#pragma omp parallel for private(i)
#endif
			for (i = 0; i < size.Height; ++i)
			{
				int j;
				for (j = 0; j < size.Width; ++j)
					dst[i*strideOut + j] = (ushort)calcBiPolyValue(Coeffs, count_Coeffs, j, i);
			}
		}
		free(Coeffs);
		free(Xs);
		free(Ys);
		free(Zs);
	}

	if (!smoothed)
	{
#if USE_OMP == 1
#pragma omp parallel for private(i)
#endif
		for (i = 0; i < size.Height; ++i)
		{
			int j;
			for (j = 0; j < size.Width; ++j)
				dst[i * strideOut + j] = (ushort)(1.0 / ((src[i * strideSrc + j] + add) * mul));
		}
	}
}

static void calibrationFrom_32b(uint *source, int strideSrc, uint *outImg, int strideOut, struct sSize size, int smoothed)
{
	double Avg, Min, Max, Stdev;
	double add, mul;
	int count = 0, i, j;
	uint *src = source;
	uint *dst = outImg;

	imageInfo((uchar*)source, size.Width, size.Height, strideSrc, &Avg, &Min, &Max, &Stdev, sizeof(*source) * 8);
	add = (Min < 1.0) ? 1.0 : 0.0;
	mul = 1.0 / (Avg + Stdev + add);

	if (smoothed)
	{
		double *Coeffs = (double*)malloc(smoothed * smoothed * sizeof(double));
		int N = 15;
		double dy = (double)size.Height / (double)(N - 1);
		double dx = (double)size.Width / (double)(N - 1);
		int count_Coeffs;
		double err;
		double  *Xs, *Ys, *Zs;
		Xs = (double*)malloc(N * N * sizeof(double));
		Ys = (double*)malloc(N * N * sizeof(double));
		Zs = (double*)malloc(N * N * sizeof(double));
		smoothed = smoothed > N*N ? N*N : smoothed;
		for (i = 0; i < N; ++i)
		{
			int yind = MIN((int)((double)i * dy), size.Height - 1);
			for (j = 0; j < N; ++j)
			{
				int xind = MIN((int)((double)j * dx), size.Width - 1);
				Xs[count] = xind;
				Ys[count] = yind;
				Zs[count++] = src[yind * strideSrc + xind];
			}
		}
		count_Coeffs = 0;
		err = fitToBiPoly(Xs, Ys, Zs, N * N, Coeffs, smoothed, &count_Coeffs);
		if (err > 1e9)
		{
			smoothed = 0;
		}
		else
		{
#if USE_OMP == 1
#pragma omp parallel for private(i)
#endif
			for (i = 0; i < size.Height; ++i)
			{
				int j;
				for (j = 0; j < size.Width; ++j)
					dst[i*strideOut + j] = (uint)calcBiPolyValue(Coeffs, count_Coeffs, j, i);
			}
		}
		free(Coeffs);
		free(Xs);
		free(Ys);
		free(Zs);
	}

	if (!smoothed)
	{
#if USE_OMP == 1
#pragma omp parallel for private(i)
#endif
		for (i = 0; i < size.Height; ++i)
		{
			int j;
			for (j = 0; j < size.Width; ++j)
				dst[i * strideOut + j] = (uint)(1.0 / ((src[i * strideSrc + j] + add) * mul));
		}
	}
}

int calibrationFrom(uchar *source, int strideIn, int width, int height, uchar *outImg, int strideOut, int smoothed, int bpp)
{
	struct sSize size;
	size.Width = width;
	size.Stride = strideIn;
	size.Height = height;
	switch (bpp)
	{
	case 8:
	{
		calibrationFrom_8b(source, strideIn, outImg, strideOut, size, smoothed);
		return 0;
	}
	case 16:
	{
		calibrationFrom_16b((ushort *)source, strideIn, (ushort *)outImg, strideOut, size, smoothed);
		return 0;
	}
	case 32:
	{
		calibrationFrom_32b((uint *)source, strideIn, (uint *)outImg, strideOut, size, smoothed);
		return 0;
	}
	default:
		return -1;
	}
}

