#include "astrocv.h"
#include <math.h>

#ifndef USE_OMP
#define USE_OMP 0
#endif


static void drawLine(uchar *img, int width, int height, int stride, int x1, int y1, int x2, int y2, int bpp)
{
	int i;
	const int white_level = 0xFF;

	struct sSize size;
	
	const int deltaX = abs(x2 - x1);
	const int deltaY = abs(y2 - y1);
	const int signX = x1 < x2 ? 1 : -1;
	const int signY = y1 < y2 ? 1 : -1;
	int error = deltaX - deltaY;
	const int nbytes = bpp / 8;
	
	size.Width = width;
	size.Stride = stride;
	size.Height = height;

	if (x1 < 0 || x1 >= width || y1 < 0 || y1 >= height || x2 < 0 || x2 >= width || y2 < 0 || y2 >= height)
		return;
	
	for (i = 0; i < nbytes; i++)
		img[(y2*size.Stride + x2 * (bpp/8)) + i] = white_level;
	while (x1 != x2 || y1 != y2)
	{
	    int error2;
		for (i = 0; i < nbytes; i++)
			img[(y1*size.Stride + x1 * (bpp/8)) + i] = white_level;
		error2 = error * 2;
		if (error2 > -deltaY)
		{
			error -= deltaY;
			x1 += signX;
		}
		if (error2 < deltaX)
		{
			error += deltaX;
			y1 += signY;
		}
	}
}


static int drawObject_8b(uchar *img, struct sSize size, int x, int y, int signal, int radius)
{
	int sumIntens = 0;
	const int x_max = x + radius >= size.Width ? size.Width - 1 : (x + radius);
	const int y_max = y + radius >= size.Height ? size.Height - 1 : (y + radius);
	radius = (radius < 0) ? abs(radius) : radius;
	signal = (signal < 0) ? abs(signal) : signal;
	
	#if USE_OMP == 1
	#pragma omp parallel
	#endif
	{
		int i, j, localSum = 0;
		#if USE_OMP == 1
		#pragma omp for private(j)
		#endif
		for (j = (y - radius < 0) ? 0 : (y - radius); j <= y_max; j++)
		{
			int x_min = (x - radius < 0) ? 0 : (x - radius);
			uchar *dst = (uchar*)(img + j*size.Stride) + x_min;
			for (i = x_min; i <= x_max; i++)
			{
				double d = sqrt((double)((i - x) * (i - x) + (j - y) * (j - y)));
				if (d < radius) {
					double koef = d / radius;
					//double intens = (1.0 - koef * koef);
					double tmp = 10*koef*koef;
					double intens = 5.0 / (5.0 + koef + tmp + tmp*tmp);
					double val = (intens * signal) + *(dst);
					val = SATURATE(val, (double)0xff);
					*dst = (uchar)val;
					localSum += (uchar)val;
				}
				dst++;
			}
		}
		#if USE_OMP == 1
		#pragma omp critical
		#endif
		{
			sumIntens += localSum;
		}
	}
	return sumIntens;
}

static int drawObject_16b(uchar *img, struct sSize size, int x, int y, int signal, int radius)
{
	int sumIntens = 0;
	const int x_max = x + radius >= size.Width ? size.Width - 1 : (x + radius);
	const int y_max = y + radius >= size.Height ? size.Height - 1 : (y + radius);
	radius = (radius < 0) ? abs(radius) : radius;
	signal = (signal < 0) ? abs(signal) : signal;
	
	#if USE_OMP == 1
	#pragma omp parallel
	#endif
	{
		int i, j, localSum = 0;
		#if USE_OMP == 1
		#pragma omp for private(j)
		#endif
		for (j = (y - radius < 0) ? 0 : (y - radius); j <= y_max; j++)
		{
			int x_min = (x - radius < 0) ? 0 : (x - radius);
			ushort *dst = (ushort*)(img + j*size.Stride) + x_min;
			for (i = x_min; i <= x_max; i++)
			{
				double d = sqrt((double)((i - x) * (i - x) + (j - y) * (j - y)));
				if (d < radius) {
					double koef = d / radius;
					//double intens = (1.0 - koef * koef);
					double tmp = 10*koef*koef;
					double intens = 5.0 / (5.0 + koef + tmp + tmp*tmp);
					double val = (intens * signal) + *(dst);
					val = SATURATE(val, (double)0xffff);
					*dst = (ushort)val;
					localSum += (ushort)val;
				}
				dst++;
			}
		}
		#if USE_OMP == 1
		#pragma omp critical
		#endif
		{
			sumIntens += localSum;
		}
	}
	return sumIntens;
}

static int drawObject_32b(uchar *img, struct sSize size, int x, int y, int signal, int radius)
{
	int sumIntens = 0;
	const int x_max = x + radius >= size.Width ? size.Width - 1 : (x + radius);
	const int y_max = y + radius >= size.Height ? size.Height - 1 : (y + radius);
	radius = (radius < 0) ? abs(radius) : radius;
	signal = (signal < 0) ? abs(signal) : signal;
	
	#if USE_OMP == 1
	#pragma omp parallel
	#endif
	{
		int i, j, localSum = 0;
		#if USE_OMP == 1
		#pragma omp for private(j)
		#endif
		for (j = (y - radius < 0) ? 0 : (y - radius); j <= y_max; j++)
		{
			int x_min = (x - radius < 0) ? 0 : (x - radius);
			uint *dst = (uint*)(img + j*size.Stride) + x_min;
			for (i = x_min; i <= x_max; i++)
			{
				double d = sqrt((double)((i - x) * (i - x) + (j - y) * (j - y)));
				if (d < radius) {
					double koef = d / radius;
					//double intens = (1.0 - koef * koef);
					double tmp = 10*koef*koef;
					double intens = 5.0 / (5.0 + koef + tmp + tmp*tmp);
					double val = (intens * signal) + *(dst);
					val = SATURATE(val, (double)0xffffffff);
					*dst = (uint)val;
					localSum += (uint)val;
				}
				dst++;
			}
		}
		#if USE_OMP == 1
		#pragma omp critical
		#endif
		{
			sumIntens += localSum;
		}
	}
	return sumIntens;
}

int drawObject(uchar *img, int width, int height, int stride, int x, int y, int signal, int radius, int bpp)
{
	struct sSize size;
	size.Width = width;
	size.Stride = stride;
	size.Height = height;
	switch (bpp)
	{
	case 8:
		return drawObject_8b(img, size, x, y, signal, radius);
	case 16:
		return drawObject_16b(img, size, x, y, signal, radius);
	case 32:
		return  drawObject_32b(img, size, x, y, signal, radius);
	default:
		return 0;
	}
}



static void addMarker_8b(uchar *img, struct sSize size, int left, int top, int right, int bottom, uint signal)
{
	int rad = MAX(right - left, bottom - top) / 2 + 5;
	int i = 0;
	int x_local = right - (right - left) / 2;
	int y_local = bottom - (bottom - top) / 2;
	int x_max = x_local + rad >= size.Width ? size.Width - 1 : (x_local + rad);
	int y_max = y_local + rad >= size.Height ? size.Height - 1 : (y_local + rad);
	for (i = (y_local - rad < 0) ? 0 : (y_local - rad); i <= y_max; i++)
	{
		int x_min = (x_local - rad < 0) ? 0 : (x_local - rad);
		uchar *dst = (uchar*)(img + i*size.Stride) + x_min;
		int j;
		for (j = x_min; j <= x_max; j++)
		{
			if ((int)hypot(i - y_local, j - x_local) == rad)
				*dst = (uchar)signal;
			dst++;
		}
	}
}

static void addMarker_16b(uchar *img, struct sSize size, int left, int top, int right, int bottom, uint signal)
{
	int rad = MAX(right - left, bottom - top) / 2 + 5;
	int i = 0;
	int x_local = right - (right - left) / 2;
	int y_local = bottom - (bottom - top) / 2;
	int x_max = x_local + rad >= size.Width ? size.Width - 1 : (x_local + rad);
	int y_max = y_local + rad >= size.Height ? size.Height - 1 : (y_local + rad);
	for (i = (y_local - rad < 0) ? 0 : (y_local - rad); i <= y_max; i++)
	{
		int x_min = (x_local - rad < 0) ? 0 : (x_local - rad);
		ushort *dst = (ushort*)(img + i*size.Stride) + x_min;
		int j;
		for (j = x_min; j <= x_max; j++)
		{
			if ((int)hypot(i - y_local, j - x_local) == rad)
				*dst = (ushort)signal;
			dst++;
		}
	}
}

static void addMarker_32b(uchar *img, struct sSize size, int left, int top, int right, int bottom, uint signal)
{
	int rad = MAX(right - left, bottom - top) / 2 + 5;
	int i = 0;
	int x_local = right - (right - left) / 2;
	int y_local = bottom - (bottom - top) / 2;
	int x_max = x_local + rad >= size.Width ? size.Width - 1 : (x_local + rad);
	int y_max = y_local + rad >= size.Height ? size.Height - 1 : (y_local + rad);
	for (i = (y_local - rad < 0) ? 0 : (y_local - rad); i <= y_max; i++)
	{
		int x_min = (x_local - rad < 0) ? 0 : (x_local - rad);
		uint *dst = (uint*)(img + i*size.Stride) + x_min;
		int j;
		for (j = x_min; j <= x_max; j++)
		{
			if ((int)hypot(i - y_local, j - x_local) == rad)
				*dst = (uint)signal;
			dst++;
		}
	}
}

int addMarker(uchar *img, int width, int height, int stride, double x, double y, int left, int top, int right, int bottom, int bpp)
{
	int rad = right - left > bottom - top ? (right - left) / 2 + 5 : (bottom - top) / 2 + 5;
	int signal = (uint)((1LL << bpp) - 1LL);
	struct sSize size;
	size.Width = width;
	size.Stride = stride;
	size.Height = height;
	
	drawLine(img, width, height, stride, (int)(x - rad), (int)(y - rad), (int)(x + rad), (int)(y + rad), bpp);
	drawLine(img, width, height, stride, (int)(x + rad), (int)(y - rad), (int)(x - rad), (int)(y + rad), bpp);
	switch (bpp)
	{
	case 8:
		addMarker_8b(img, size, left, top, right, bottom, signal);
		return 0;
	case 16:
		addMarker_16b(img, size, left, top, right, bottom, signal);
		return 0;
	case 32:
		addMarker_32b(img, size, left, top, right, bottom, signal);
		return 0;
	default:
		return -1;
	}
}


static void addNoiseUniform_8b(uchar *img, struct sSize size, int background, int noiseMin, int noiseMax)
{
	int i = 0;
	#if  USE_OMP == 1
	#pragma omp parallel
	#endif //  USE_OMP
	{
		int j;
		unsigned int seed = (unsigned int)rand();
		#if  USE_OMP == 1
		#pragma omp for private(i)
		#endif //  USE_OMP
		for (i = 0; i < size.Height; i++)
		{
			uchar *dst = (uchar*)(img + i*size.Stride);
			for (j = 0; j < size.Width; j++)
			{
				int rand_val = rand_r(&seed);
				int val = (int)(*dst) + background + (int)((double)rand_val / RAND_MAX * (noiseMax - noiseMin) + noiseMin);
				*(dst++) = (uchar)SATURATE(val, 0xff);
			}
		}
	}
}

static void addNoiseUniform_16b(uchar *img, struct sSize size, int background, int noiseMin, int noiseMax)
{
	int i = 0;
	#if  USE_OMP == 1
	#pragma omp parallel
	#endif //  USE_OMP
	{
		int j;
		unsigned int seed = (unsigned int)rand();
		#if  USE_OMP == 1
		#pragma omp for private(i)
		#endif //  USE_OMP
		for (i = 0; i < size.Height; i++)
		{
			ushort *dst = (ushort*)(img + i*size.Stride);
			for (j = 0; j < size.Width; j++)
			{
				int rand_val = rand_r(&seed);
				int val = (int)(*dst) + background + (int)((double)rand_val / RAND_MAX * (noiseMax - noiseMin) + noiseMin);
				*(dst++) = (ushort)SATURATE(val, 0xff);
			}
		}
	}
}

static void addNoiseUniform_32b(uchar *img, struct sSize size, int background, int noiseMin, int noiseMax)
{
	int i = 0;
	#if  USE_OMP == 1
	#pragma omp parallel
	#endif //  USE_OMP
	{
		int j;
		unsigned int seed = (unsigned int)rand();
		#if  USE_OMP == 1
		#pragma omp for private(i)
		#endif //  USE_OMP
		for (i = 0; i < size.Height; i++)
		{
			uint *dst = (uint*)(img + i*size.Stride);
			for (j = 0; j < size.Width; j++)
			{
				int rand_val = rand_r(&seed);
				int val = (int)(*dst) + background + (int)((double)rand_val / RAND_MAX * (noiseMax - noiseMin) + noiseMin);
				*(dst++) = (uint)SATURATE(val, 0xff);
			}
		}
	}
}

int addNoiseUniform(uchar *img, int width, int height, int stride, int background, int noiseMin, int noiseMax, int bpp)
{
	struct sSize size;
	size.Width = width;
	size.Stride = stride;
	size.Height = height;
	switch (bpp)
	{
	case 8:
		addNoiseUniform_8b(img, size, background, noiseMin, noiseMax);
		return 0;
	case 16:
		addNoiseUniform_16b(img, size, background, noiseMin, noiseMax);
		return 0;
	case 32:
		addNoiseUniform_32b(img, size, background, noiseMin, noiseMax);
		return 0;
	default:
		return -1;
	}
}


static double normRandom(double mX, double sigma)
{
	double  a, b, r, Sq;
	do
	{
		a = 2.0 * rand() / RAND_MAX - 1.0;
		b = 2.0 * rand() / RAND_MAX - 1.0;
		r = a*a + b*b;
	} while (r >= 1);
	Sq = sqrt(-2.0 * log(r) / r);
	return mX + sigma * a * Sq;
}

static double noiseBuffNormal[NOISE_BUFF_LEN];
static int noiseBuffInitialized = 0;
static void checkNoiseBuff(void)
{
	int i = 0, mo = 0, sko = 1;
	if (noiseBuffInitialized)
		return;
	srand((uint)(clock()));
#if  USE_OMP == 1
#pragma omp parallel for private(i)
#endif //  USE_OMP
	for (i = 0; i < NOISE_BUFF_LEN; i++)
	{
		noiseBuffNormal[i] = normRandom(mo, sko);
	}
	noiseBuffInitialized = 1;
}

static void addNoiseNorm_8b(uchar *img, struct sSize size, int m, double s)
{
	#if  USE_OMP == 1
	#pragma omp parallel
	#endif //  USE_OMP
	{
		int i, j;
		int index = (int)(((double)rand() / RAND_MAX) * NOISE_BUFF_LEN);
		#if  USE_OMP == 1
		#pragma omp for
		#endif //  USE_OMP
		for (i = 0; i < size.Height; i++)
		{
			uchar *dst = (uchar*)(img + i*size.Stride);
			for (j = 0; j < size.Width; j++, index++)
			{
				int val = (int)(*dst) + (int)(m + s * noiseBuffNormal[index % NOISE_BUFF_LEN]);
				*(dst++) = (uchar)SATURATE(val, 0xff);
			}
		}
	}
}

static void addNoiseNorm_16b(uchar *img, struct sSize size, int m, double s)
{
	//printf("size = %dx%d, stride = %d, %d +- %lg\n", size.Width, size.Height, size.Stride, m, s);
	//return;
	#if  USE_OMP == 1
	#pragma omp parallel
	#endif //  USE_OMP
	{
		int i, j;
		int index = (int)(((double)rand() / RAND_MAX) * NOISE_BUFF_LEN);
		#if  USE_OMP == 1
		#pragma omp for
		#endif //  USE_OMP
		for (i = 0; i < size.Height; i++)
		{
			ushort *dst = (ushort*)(img + i*size.Stride);
			for (j = 0; j < size.Width; j++, index++)
			{
				int val = (int)(*dst) + (int)(m + s * noiseBuffNormal[index % NOISE_BUFF_LEN]);
				*(dst++) = (ushort)SATURATE(val, 0xffff);
			}
		}
	}
}

static void addNoiseNorm_32b(uchar *img, struct sSize size, int m, double s)
{
	#if  USE_OMP == 1
	#pragma omp parallel
	#endif //  USE_OMP
	{
		int i, j;
		int index = (int)(((double)rand() / RAND_MAX) * NOISE_BUFF_LEN);
		#if  USE_OMP == 1
		#pragma omp for
		#endif //  USE_OMP
		for (i = 0; i < size.Height; i++)
		{
			uint *dst = (uint*)(img + i*size.Stride);
			for (j = 0; j < size.Width; j++, index++)
			{
				int val = (int)(*dst) + (int)(m + s * noiseBuffNormal[index % NOISE_BUFF_LEN]);
				*(dst++) = (uint)SATURATE(val, 0xffffffff);
			}
		}
	}
}

int addNoiseNorm(uchar *img, int width, int height, int stride, int m, double s, int bpp)
{
	struct sSize size;
	size.Width = width;
	size.Stride = stride;
	size.Height = height;
	checkNoiseBuff();
	switch (bpp)
	{
	case 8:
		addNoiseNorm_8b(img, size, m, s);
		return 0;
	case 16:
		addNoiseNorm_16b(img, size, m, s);
		return 0;
	case 32:
		addNoiseNorm_32b(img, size, m, s);
		return 0;
	default:
		return -1;
	}
}

