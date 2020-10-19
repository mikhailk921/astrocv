#include "astrocv.h"

void fitInt(int min, int max, int *val)
{
	if (*val < min)
		*val = min;
	else if (*val > max)
		*val = max;
}

static void qs(struct SAreaInfo *items, int left, int right)
{
	register int i, j;
	register struct SAreaInfo x, y;

	i = left; j = right;
	x = items[(left + right) / 2]; // выбор компаранда

	do {
		while ((items[i].S > x.S) && (i < right)) i++;
		while ((x.S > items[j].S) && (j > left)) j--;

		if (i <= j) {
			y = items[i];
			items[i] = items[j];
			items[j] = y;
			i++; j--;
		}
	} while (i <= j);

	if (left < j) qs(items, left, j);
	if (i < right) qs(items, i, right);
}

// Quick Sort
void quickSort(struct SAreaInfo *items, int count)
{
	qs(items, 0, count - 1);
}

static void info_8b(uchar *imgSrc, struct sSize size, double* avg, double* min, double* max, double* stdev)
{
	int y;
	double n = size.Height * size.Width;
	*avg = 0.0;
	*min = HUGE_VAL;
	*max = -HUGE_VAL;
	*stdev = 0.0;

#if  USE_OMP == 1
#pragma omp parallel
#endif //  USE_OMP
	{
		double sum = 0.0, sum2 = 0.0, mmin = HUGE_VAL, mmax = -HUGE_VAL;
#if  USE_OMP == 1
#pragma omp for private(y)
#endif //  USE_OMP
		for (y = 0; y < size.Height; y++)
		{
			uchar *pix = imgSrc + y*size.Stride;
			int x;
			for (x = 0; x < size.Width; x++)
			{
				sum += *pix;
				sum2 += (double)(*pix)*(double)(*pix);
				if (mmin > *pix)
					mmin = *pix;
				if (mmax < *pix)
					mmax = *pix;
				pix++;
			}
		}

#if  USE_OMP == 1
#pragma omp critical
#endif //USE_OMP
		{
			*avg += sum;
			*stdev += sum2;
			if (*min > mmin)
				*min = mmin;
			if (*max < mmax)
				*max = mmax;
		}
	}

	*stdev = sqrt((*stdev - (*avg)*(*avg) / n) / n);
	*avg /= n;
}

static void info_16b(uchar *imgSrc, struct sSize size, double* avg, double* min, double* max, double* stdev)
{
	int y;
	double n = size.Height * size.Width;
	*avg = 0.0;
	*min = HUGE_VAL;
	*max = -HUGE_VAL;
	*stdev = 0.0;

#if  USE_OMP == 1
#pragma omp parallel
#endif //  USE_OMP
	{
		double sum = 0.0, sum2 = 0.0, mmin = HUGE_VAL, mmax = -HUGE_VAL;
#if  USE_OMP == 1
#pragma omp for private(y)
#endif //  USE_OMP
		for (y = 0; y < size.Height; y++)
		{
			ushort *pix = (ushort*)(imgSrc + y*size.Stride);
			int x;
			for (x = 0; x < size.Width; x++)
			{
				sum += *pix;
				sum2 += (double)(*pix)*(double)(*pix);
				if (mmin > *pix)
					mmin = *pix;
				if (mmax < *pix)
					mmax = *pix;
				pix++;
			}
		}
#if  USE_OMP == 1
#pragma omp critical
#endif //  USE_OMP
		{
			*avg += sum;
			*stdev += sum2;
			if (*min > mmin)
				*min = mmin;
			if (*max < mmax)
				*max = mmax;
		}
	}

	*stdev = sqrt((*stdev - (*avg)*(*avg) / n) / n);
	*avg /= n;
}

static void info_32b(uchar *imgSrc, struct sSize size, double* avg, double* min, double* max, double* stdev)
{
	int y;
	double n = size.Height * size.Width;
	*avg = 0.0;
	*min = HUGE_VAL;
	*max = -HUGE_VAL;
	*stdev = 0.0;

#if  USE_OMP == 1
#pragma omp parallel
#endif //  USE_OMP
	{
		double sum = 0.0, sum2 = 0.0, mmin = HUGE_VAL, mmax = -HUGE_VAL;
#if  USE_OMP == 1
#pragma omp for private(y)
#endif //  USE_OMP
		for (y = 0; y < size.Height; y++)
		{
			uint *pix = (uint*)(imgSrc + y*size.Stride);
			int x;
			for (x = 0; x < size.Width; x++)
			{
				sum += *pix;
				sum2 += (double)(*pix)*(double)(*pix);
				if (mmin > *pix)
					mmin = *pix;
				if (mmax < *pix)
					mmax = *pix;
				pix++;
			}
		}
#if  USE_OMP == 1
#pragma omp critical
#endif //  USE_OMP
		{
			*avg += sum;
			*stdev += sum2;
			if (*min > mmin)
				*min = mmin;
			if (*max < mmax)
				*max = mmax;
		}
	}

	*stdev = sqrt((*stdev - (*avg)*(*avg) / n) / n);
	*avg /= n;
}

int imageInfo(uchar *imgSrc, int width, int height, int stride, double* avg, double* min, double* max, double* stdev, uint bpp)
{
	struct sSize size;
	size.Width = width;
	size.Stride = stride;
	size.Height = height;
	if (width == 0 || stride == 0 || height == 0)
		return -1;
	switch (bpp)
	{
	case 8:
		info_8b(imgSrc, size, avg, min, max, stdev);
		return 0;
	case 16:
		info_16b(imgSrc, size, avg, min, max, stdev);
		return 0;
	case 32:
		info_32b(imgSrc, size, avg, min, max, stdev);
		return 0;
	default:
		return -1;
	}
}


static void integralFrom_8b(uchar *source, struct sSize size, uint *integral)
{
	int x, y;
	uchar *src = source;
	memset(integral, 0, size.Width * size.Height * sizeof(*integral));

	integral[0] = src[0];
	for (x = 1; x < size.Width; x++)
		integral[x] = integral[x - 1] + src[x];

	for (y = 1; y < size.Height; y++)
	{
		src = (uchar*)((char*)source + y*size.Stride);
		integral[y*size.Width] = integral[(y - 1)*size.Width] + *src;
		for (x = 1; x < size.Width; x++)
		{
			integral[y*size.Width + x] = integral[(y - 1)*size.Width + x] +
				integral[y*size.Width + x - 1] -
				integral[(y - 1)*size.Width + x - 1] +
				src[x];
		}
	}
}

static void integralFrom_16b(uchar *source, struct sSize size, uint *integral)
{
	int x, y;
	ushort *src = (ushort*)source;
	memset(integral, 0, size.Width * size.Height * sizeof(*integral));

	integral[0] = src[0];
	for (x = 1; x < size.Width; x++)
		integral[x] = integral[x - 1] + src[x];

	for (y = 1; y < size.Height; y++)
	{
		src = (ushort*)((char*)source + y*size.Stride);
		integral[y*size.Width] = integral[(y - 1)*size.Width] + *src;
		for (x = 1; x < size.Width; x++)
		{
			integral[y*size.Width + x] = (integral[(y - 1)*size.Width + x] +
				integral[y*size.Width + x - 1] -
				integral[(y - 1)*size.Width + x - 1] +
				src[x]);
		}
	}
}

static void integralFrom_32b(uchar *source, struct sSize size, uint *integral)
{
	int x, y;
	uint *src = (uint*)source;
	memset(integral, 0, size.Width * size.Height * sizeof(*integral));

	integral[0] = src[0];
	for (x = 1; x < size.Width; x++)
		integral[x] = integral[x - 1] + src[x];

	for (y = 1; y < size.Height; y++)
	{
		src = (uint*)((char*)source + y*size.Stride);
		integral[y*size.Width] = integral[(y - 1)*size.Width] + *src;
		for (x = 1; x < size.Width; x++)
		{
			integral[y*size.Width + x] = (integral[(y - 1)*size.Width + x] +
				integral[y*size.Width + x - 1] -
				integral[(y - 1)*size.Width + x - 1] +
				src[x]);
		}
	}
}

int integralFrom(uchar *source, int width, int height, int stride, uint *integral, uint bpp)
{
	int result = 0;
	struct sSize size;
	size.Width = width;
	size.Stride = stride;
	size.Height = height;
	if (width == 0 || stride == 0 || height == 0)
		return -1;
	//uint* maskIntegral = (uint*)malloc(size.Width * size.Height * sizeof(uint));
	//integralFromMask(mask, size, maskIntegral);
	switch (bpp)
	{
		case 8:
			integralFrom_8b(source, size, integral);
			break;
		case 16:
			integralFrom_16b(source, size, integral);
			break;
		case 32:
			integralFrom_32b(source, size, integral);
			break;
		default:
			result = -1;
	}

	//free(maskIntegral);
	return result;
}

int integralSum(const uint *integralBuffer, struct sSize size, int left, int top, int right, int bottom, int *count)
{
	int lt, rt, rb, lb;

	fitInt(0, size.Width - 1, &left);
	fitInt(left, size.Width - 1, &right);
	fitInt(0, size.Height - 1, &top);
	fitInt(top, size.Height - 1, &bottom);
	
	rb = integralBuffer[bottom * size.Width + right];
	if (left == 0 && top == 0) {
		lt = rt = lb = 0;
	}
	else if (top == 0) {
		lt = rt = 0;
		lb = integralBuffer[bottom*size.Width + (left - 1)];
	}
	else if (left == 0) {
		lt = lb = 0;
		rt = integralBuffer[(top - 1)*size.Width + right];
	}
	else {
		lt = integralBuffer[(top - 1)*size.Width + (left - 1)];
		lb = integralBuffer[bottom *size.Width + (left - 1)];
		rt = integralBuffer[(top - 1)*size.Width + right];
	}

	*count = (right - left + 1) * (bottom - top + 1);
	return lt + rb - rt - lb;
}

double localDifference(const uint *integral, struct sSize size, int x, int y, int innerD, int outerD, uint* IntegralMask, int *count)
{
	int mind;
	double inner, outer;
	int innercnt = 0, outercnt = 0;

	int inner_left = x - innerD;
	int inner_top = y - innerD;
	int inner_right = x + innerD;
	int inner_bottom = y + innerD;

	int outer_left = x - outerD;
	int outer_top = y - outerD;
	int outer_right = x + outerD;
	int outer_bottom = y + outerD;

	fitInt(0, size.Width - 1, &outer_left);
	fitInt(outer_left, size.Width - 1, &outer_right);
	fitInt(0, size.Height - 1, &outer_top);
	fitInt(outer_top, size.Height - 1, &outer_bottom);

	mind = 2;

	if (outer_right - outer_left <= mind * 2)
		return 0.0;
	if (outer_bottom - outer_top <= mind * 2)
		return 0.0;

	fitInt(outer_left + mind, outer_right - mind, &inner_left);
	fitInt(outer_top + mind, outer_bottom - mind, &inner_top);
	fitInt(outer_left + mind, outer_right - mind, &inner_right);
	fitInt(outer_top + mind, outer_bottom - mind, &inner_bottom);

	inner = integralSum(integral, size, inner_left, inner_top, inner_right, inner_bottom, &innercnt);
	outer = integralSum(integral, size, outer_left, outer_top, outer_right, outer_bottom, &outercnt);

	if (IntegralMask != NULL)
	{
		innercnt = integralSum(IntegralMask, size, inner_left, inner_top, inner_right, inner_bottom, &innercnt);
		outercnt = integralSum(IntegralMask, size, outer_left, outer_top, outer_right, outer_bottom, &outercnt);
	}
		
	*count = outercnt;
	outer -= inner;
	outercnt -= innercnt;
	if (innercnt > 0)
		inner /= (double)innercnt;
	else
		inner = 0.0;
	if (outercnt > 0)
		outer /= (double)outercnt;
	else
		outer = 0.0;

	return (inner - outer);
}

double integralAvg(const uint *imgSrc, struct sSize size, int left, int top, int right, int bottom)
{
	double ret;
	int count;
	ret = integralSum(imgSrc, size, left, top, right, bottom, &count);
	if (count == 0)
		return 0.0;
	ret /= (double)count;
	return ret;
}

static void selectThreshold_8b(uchar *img, int stride, double threshold, int *left, int *top, int *right, int *bottom)
{
	int x, y;
	int new_left = *right, new_top = *bottom, new_right = *left, new_bottom = *top;
	for (y = *top; y <= *bottom; y++)
	{
		uchar *pix = img + y*stride + *left;
		for (x = *left; x <= *right; x++)
		{
			if (*pix >= threshold)
			{
				if (x < new_left)
					new_left = x;
				if (x > new_right)
					new_right = x;
				if (y < new_top)
					new_top = y;
				if (y > new_bottom)
					new_bottom = y;
			}
			pix++;
		}
	}
	*left = new_left;
	*top = new_top;
	*right = new_right;
	*bottom = new_bottom;
}
static void selectThreshold_16b(uchar *img, int stride, double threshold, int *left, int *top, int *right, int *bottom)
{
	int x, y;
	int new_left = *right, new_top = *bottom, new_right = *left, new_bottom = *top;
	for (y = *top; y <= *bottom; y++)
	{
		ushort *pix = (ushort*)(img + y*stride) + *left;
		for (x = *left; x <= *right; x++)
		{
			if (*pix >= threshold)
			{
				if (x < new_left)
					new_left = x;
				if (x > new_right)
					new_right = x;
				if (y < new_top)
					new_top = y;
				if (y > new_bottom)
					new_bottom = y;
			}
			pix++;
		}
	}
	*left = new_left;
	*top = new_top;
	*right = new_right;
	*bottom = new_bottom;
}
static void selectThreshold_32b(uchar *img, int stride, double threshold, int *left, int *top, int *right, int *bottom)
{
	int x, y;
	int new_left = *right, new_top = *bottom, new_right = *left, new_bottom = *top;
	for (y = *top; y <= *bottom; y++)
	{
		uint *pix = (uint*)(img + y*stride) + *left;
		for (x = *left; x <= *right; x++)
		{
			if (*pix >= threshold)
			{
				if (x < new_left)
					new_left = x;
				if (x > new_right)
					new_right = x;
				if (y < new_top)
					new_top = y;
				if (y > new_bottom)
					new_bottom = y;
			}
			pix++;
		}
	}
	*left = new_left;
	*top = new_top;
	*right = new_right;
	*bottom = new_bottom;
}

int localThreshold(uchar *processed, int bpp, struct sSize size, int *left, int *top, int *right, int *bottom)
{
	double threshold;
	double avg, stdev, min, max;
	struct sSize areaSize;

	uchar *area = processed + (*top)*size.Stride + (*left) * (bpp / 8);
	areaSize.Width = (*right - *left + 1);
	areaSize.Height = (*bottom - *top + 1);
	areaSize.Stride = size.Stride;

	switch (bpp)
	{
	case 8:
	{
		info_8b(area, areaSize, &avg, &min, &max, &stdev);
		break;
	}
	case 16:
	{
		info_16b(area, areaSize, &avg, &min, &max, &stdev);
		break;
	}
	case 32:
	{
		info_32b(area, areaSize, &avg, &min, &max, &stdev);
		break;
	}
	default:
		return -1;
	}

	threshold = avg + stdev;
	threshold = avg + 2.0 * stdev;
	if (avg - min < max - avg)
		threshold = avg + (avg - min) + stdev*0.5;
	else
		threshold = avg - (max - avg) - stdev*0.5;
	threshold = MIN(MAX(threshold, min + stdev), max - stdev);

	switch (bpp)
	{
	case 8:
	{
		selectThreshold_8b(processed, size.Stride, threshold, left, top, right, bottom);
		break;
	}
	case 16:
	{
		selectThreshold_16b(processed, size.Stride, threshold, left, top, right, bottom);
		break;
	}
	case 32:
	{
		selectThreshold_32b(processed, size.Stride, threshold, left, top, right, bottom);
		break;
	}
	default:
		return -1;
	}

	return 0;
}

int localizeFeature(uint *integral, struct sSize size, int *left, int *top, int *right, int *bottom)
{
	int new_left, new_top, new_right, new_bottom;
	double l, r, ml, mr, el, er;

	// Left boundary
	l = *left; r = *right - 1;
	ml = l + (r - l) * (1.0 - PHI); mr = l + (r - l) * PHI;
	el = integralAvg(integral, size, (int)ml, *top, *right, *bottom) - integralAvg(integral, size, *left, *top, (int)ml, *bottom);
	er = integralAvg(integral, size, (int)mr, *top, *right, *bottom) - integralAvg(integral, size, *left, *top, (int)mr, *bottom);
	while (r - l > 1.5)
	{
		if (el > er)
		{
			r = mr; mr = ml; ml = l + (r - l) * (1.0 - PHI); er = el;
			el = integralAvg(integral, size, (int)ml, *top, *right, *bottom) - integralAvg(integral, size, *left, *top, (int)ml, *bottom);
		}
		else
		{
			l = ml; ml = mr; mr = l + (r - l) * PHI; el = er;
			er = integralAvg(integral, size, (int)mr, *top, *right, *bottom) - integralAvg(integral, size, *left, *top, (int)mr, *bottom);
		}
	}
	new_left = (int)l;

	// Top boundary
	l = *top; r = *bottom - 1;
	ml = l + (r - l) * (1.0 - PHI); mr = l + (r - l) * PHI;
	el = integralAvg(integral, size, *left, (int)ml, *right, *bottom) - integralAvg(integral, size, *left, *top, *right, (int)ml);
	er = integralAvg(integral, size, *left, (int)mr, *right, *bottom) - integralAvg(integral, size, *left, *top, *right, (int)mr);
	while (r - l > 1.5)
	{
		if (el > er)
		{
			r = mr; mr = ml; ml = l + (r - l) * (1.0 - PHI); er = el;
			el = integralAvg(integral, size, *left, (int)ml, *right, *bottom) - integralAvg(integral, size, *left, *top, *right, (int)ml);
		}
		else
		{
			l = ml; ml = mr; mr = l + (r - l) * PHI; el = er;
			er = integralAvg(integral, size, *left, (int)mr, *right, *bottom) - integralAvg(integral, size, *left, *top, *right, (int)mr);
		}
	}
	new_top = (int)l;

	// Right boundary
	l = new_left + 1; r = *right;
	ml = l + (r - l) * (1.0 - PHI); mr = l + (r - l) * PHI;
	el = integralAvg(integral, size, *left, *top, (int)ml, *bottom) - integralAvg(integral, size, (int)ml, *top, *right, *bottom);
	er = integralAvg(integral, size, *left, *top, (int)mr, *bottom) - integralAvg(integral, size, (int)mr, *top, *right, *bottom);
	while (r - l > 1.5)
	{
		if (el > er)
		{
			r = mr; mr = ml; ml = l + (r - l) * (1.0 - PHI); er = el;
			el = integralAvg(integral, size, *left, *top, (int)ml, *bottom) - integralAvg(integral, size, (int)ml, *top, *right, *bottom);
		}
		else
		{
			l = ml; ml = mr; mr = l + (r - l) * PHI; el = er;
			er = integralAvg(integral, size, *left, *top, (int)mr, *bottom) - integralAvg(integral, size, (int)mr, *top, *right, *bottom);
		}
	}
	new_right = (int)(r)-1;

	// Bottom boundary
	l = new_top + 1; r = *bottom;
	ml = l + (r - l) * (1.0 - PHI); mr = l + (r - l) * PHI;
	el = integralAvg(integral, size, *left, *top, *right, (int)ml) - integralAvg(integral, size, *left, (int)ml, *right, *bottom);
	er = integralAvg(integral, size, *left, *top, *right, (int)mr) - integralAvg(integral, size, *left, (int)mr, *right, *bottom);
	while (r - l > 1.5)
	{
		if (el > er)
		{
			r = mr; mr = ml; ml = l + (r - l) * (1.0 - PHI); er = el;
			el = integralAvg(integral, size, *left, *top, *right, (int)ml) - integralAvg(integral, size, *left, (int)ml, *right, *bottom);
		}
		else
		{
			l = ml; ml = mr; mr = l + (r - l) * PHI; el = er;
			er = integralAvg(integral, size, *left, *top, *right, (int)mr) - integralAvg(integral, size, *left, (int)mr, *right, *bottom);
		}
	}
	new_bottom = (int)(r)-1;

	*left = new_left;
	*top = new_top;
	*right = new_right;
	*bottom = new_bottom;

	return 0;
}

static int getBinIndex(const double *R, double *I, int length, double d2)
{
	int l, r;
	l = 0;
	r = length;
	while (r - l > 1)
	{
		int m = (l + r) / 2;
		if (R[m]*R[m] < d2)
			l = m;
		else
			r = m;
	}
	return l;
}

int energyDistribution(uchar *img, int width, int height, int stride, int bpp, double x0, double y0, const double *R, double *I, int length)
{
	int i, x, y;
	double total;
	
	memset(I, 0, length * sizeof(*I));
	
	switch (bpp)
	{
	case 8:
	{
		for (y = 0; y < height; y++)
		{
			double dy = (double)y - y0;
			uchar *src = (uchar*)(img + y*stride);
			for (x = 0; x < width; x++)
			{
				double dx = (double)x - x0;
				int bin = getBinIndex(R, I, length, dx*dx + dy*dy);
				I[bin] += (double)(*src);
				src++;
			}
		}
		break;
	}
	case 16:
	{
		for (y = 0; y < height; y++)
		{
			double dy = (double)y - y0;
			ushort *src = (ushort*)(img + y*stride);
			for (x = 0; x < width; x++)
			{
				double dx = (double)x - x0;
				int bin = getBinIndex(R, I, length, dx*dx + dy*dy);
				I[bin] += (double)(*src);
				src++;
			}
		}
		break;
	}
	case 32:
	{
		for (y = 0; y < height; y++)
		{
			double dy = (double)y - y0;
			uint *src = (uint*)(img + y*stride);
			for (x = 0; x < width; x++)
			{
				double dx = (double)x - x0;
				int bin = getBinIndex(R, I, length, dx*dx + dy*dy);
				I[bin] += (double)(*src);
				src++;
			}
		}
		break;
	}
	default:
		return -1;
	}
	
	for (i = 1; i < length; i++)
		I[i] += I[i - 1];
	
	total = I[length - 1];
	total = MAX(total, 1.0);
	for (i = 0; i < length; i++)
		I[i] /= total;
	
	return 0;
}

int powerDistribution(uchar *img, int width, int height, int stride, int bpp, double x0, double y0, const double *R, double *I, int length)
{
	int i, x, y;
	double max, min;
	
	memset(I, 0, length * sizeof(*I));
	
	switch (bpp)
	{
	case 8:
	{
		for (y = 0; y < height; y++)
		{
			double dy = (double)y - y0;
			uchar *src = (uchar*)(img + y*stride);
			for (x = 0; x < width; x++)
			{
				double dx = (double)x - x0;
				int bin = getBinIndex(R, I, length, dx*dx + dy*dy);
				I[bin] = MAX(I[bin], (double)(*src));
				src++;
			}
		}
		break;
	}
	case 16:
	{
		for (y = 0; y < height; y++)
		{
			double dy = (double)y - y0;
			ushort *src = (ushort*)(img + y*stride);
			for (x = 0; x < width; x++)
			{
				double dx = (double)x - x0;
				int bin = getBinIndex(R, I, length, dx*dx + dy*dy);
				I[bin] = MAX(I[bin], (double)(*src));
				src++;
			}
		}
		break;
	}
	case 32:
	{
		for (y = 0; y < height; y++)
		{
			double dy = (double)y - y0;
			uint *src = (uint*)(img + y*stride);
			for (x = 0; x < width; x++)
			{
				double dx = (double)x - x0;
				int bin = getBinIndex(R, I, length, dx*dx + dy*dy);
				I[bin] = MAX(I[bin], (double)(*src));
				src++;
			}
		}
		break;
	}
	default:
		return -1;
	}
	
	for (i = length - 2; i >= 0; i--)
		I[i] = MAX(I[i], I[i + 1]);
	
	max = I[0];
	max = MAX(max, 1.0);
	min = 0.0; //I[length - 1];
	for (i = 0; i < length; i++)
		I[i] = (I[i] - min) / (max - min);
	return 0;
}
