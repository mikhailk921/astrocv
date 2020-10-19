#include "astrocv.h"

#ifndef USE_OMP
#define USE_OMP 0
#endif

#if USE_OMP == 1
#include <omp.h>
#endif

static void initAreaInfo(struct SAreaInfo *array, int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		array[i].x = -1;
		array[i].y = -1;
		array[i].S = -HUGE_VAL;
	}
}

static void pushToHeap(struct SAreaInfo *heap, int size, struct SAreaInfo *info)
{
	struct SAreaInfo tmp;
	int child, root = 0;
	heap[0] = *info;
	while (root * 2 + 1 < size)
	{
		child = root * 2 + 1;
		if (child + 1 < size && heap[child].S > heap[child + 1].S)
			child++;
		if (heap[root].S <= heap[child].S)
			break;
		tmp = heap[root];
		heap[root] = heap[child];
		heap[child] = tmp;
		root = child;
	}
}


/*
* Расчёт коэффициентов билинейной интерполяции:
* 		I = a*x^2 + b*x*y + c*y^2 + d
* Начало координат в верхнем левом углу заданного квадрата.
* Для расчёта коэффициентов используются средние значения яркостей по
* границе заданного прямоугольника для четырёх углов.
*/
static void fitBackground_8b(uchar *source, int stride, int left, int top, int right, int bottom, double *a, double *b, double *c, double *d)
{
	double lt = 0.0, rt = 0.0, lb = 0.0, rb = 0.0;
	double kx, ky;
	int cnt1 = 0, cnt2 = 0, cnt3 = 0, cnt4 = 0;
	int x, y;
	uchar *pxTop, *pxBottom;

	pxTop = source + top*stride + left;
	pxBottom = source + top*stride + left;
	for (x = 0; x <= (right - left) / 2; x++, cnt1++)
	{
		lt += *pxTop;
		lb += *pxBottom;
		pxTop++;
		pxBottom++;
	}
	for (; x <= right - left; x++, cnt2++)
	{
		rt += *pxTop;
		rb += *pxBottom;
		pxTop++;
		pxBottom++;
	}
	for (y = 0; y <= (bottom - top) / 2; y++, cnt3++)
	{
		lt += *(source + y*stride + left);
		rt += *(source + y*stride + right);
	}
	for (; y <= (bottom - top); y++, cnt4++)
	{
		lb += *(source + y*stride + left);
		rb += *(source + y*stride + right);
	}
	lt /= (double)(cnt1 + cnt3);
	rt /= (double)(cnt2 + cnt3);
	lb /= (double)(cnt1 + cnt4);
	rb /= (double)(cnt2 + cnt4);

	kx = 1.0 / (right - left + 1.0);
	ky = 1.0 / (bottom - top + 1.0);

	*a = (-lt + rb) * kx*kx;
	*b = (lb + lt - rb - rt) * kx*ky;
	*c = (-lt + rt) * ky*ky;
	*d = lt;
}
static void fitBackground_16b(uchar *source, int stride, int left, int top, int right, int bottom, double *a, double *b, double *c, double *d)
{
	double lt = 0.0, rt = 0.0, lb = 0.0, rb = 0.0;
	double kx, ky;
	int cnt1 = 0, cnt2 = 0, cnt3 = 0, cnt4 = 0;
	int x, y;
	ushort *pxTop, *pxBottom;

	pxTop = (ushort*)(source + top*stride) + left;
	pxBottom = (ushort*)(source + top*stride) + left;
	for (x = 0; x <= (right - left) / 2; x++, cnt1++)
	{
		lt += *pxTop;
		lb += *pxBottom;
		pxTop++;
		pxBottom++;
	}
	for (; x <= right - left; x++, cnt2++)
	{
		rt += *pxTop;
		rb += *pxBottom;
		pxTop++;
		pxBottom++;
	}
	for (y = 0; y <= (bottom - top) / 2; y++, cnt3++)
	{
		lt += *((ushort*)(source + y*stride) + left);
		rt += *((ushort*)(source + y*stride) + right);
	}
	for (; y <= (bottom - top); y++, cnt4++)
	{
		lb += *((ushort*)(source + y*stride) + left);
		rb += *((ushort*)(source + y*stride) + right);
	}
	lt /= (double)(cnt1 + cnt3);
	rt /= (double)(cnt2 + cnt3);
	lb /= (double)(cnt1 + cnt4);
	rb /= (double)(cnt2 + cnt4);

	kx = 1.0 / (right - left + 1.0);
	ky = 1.0 / (bottom - top + 1.0);

	*a = (-lt + rb) * kx*kx;
	*b = (lb + lt - rb - rt) * kx*ky;
	*c = (-lt + rt) * ky*ky;
	*d = lt;
}
static void fitBackground_32b(uchar *source, int stride, int left, int top, int right, int bottom, double *a, double *b, double *c, double *d)
{
	double lt = 0.0, rt = 0.0, lb = 0.0, rb = 0.0;
	double kx, ky;
	int cnt1 = 0, cnt2 = 0, cnt3 = 0, cnt4 = 0;
	int x, y;
	uint *pxTop, *pxBottom;

	pxTop = (uint*)(source + top*stride) + left;
	pxBottom = (uint*)(source + top*stride) + left;
	for (x = 0; x <= (right - left) / 2; x++, cnt1++)
	{
		lt += *pxTop;
		lb += *pxBottom;
		pxTop++;
		pxBottom++;
	}
	for (; x <= right - left; x++, cnt2++)
	{
		rt += *pxTop;
		rb += *pxBottom;
		pxTop++;
		pxBottom++;
	}
	for (y = 0; y <= (bottom - top) / 2; y++, cnt3++)
	{
		lt += *((uint*)(source + y*stride) + left);
		rt += *((uint*)(source + y*stride) + right);
	}
	for (; y <= (bottom - top); y++, cnt4++)
	{
		lb += *((uint*)(source + y*stride) + left);
		rb += *((uint*)(source + y*stride) + right);
	}
	lt /= (double)(cnt1 + cnt3);
	rt /= (double)(cnt2 + cnt3);
	lb /= (double)(cnt1 + cnt4);
	rb /= (double)(cnt2 + cnt4);

	kx = 1.0 / (right - left + 1.0);
	ky = 1.0 / (bottom - top + 1.0);

	*a = (-lt + rb) * kx*kx;
	*b = (lb + lt - rb - rt) * kx*ky;
	*c = (-lt + rt) * ky*ky;
	*d = lt;
}

static void measureSignal_8b(uchar *source, int stride, int left, int top, int right, int bottom, int bgd_x0, int bgd_y0, double a, double b, double c, double d,
	double *s1, double *s2, struct SObjectInfo *result)
{
	int x, y, bgd_x, bgd_y;
	*s1 = *s2 = 0.0;
	for (y = top, bgd_y = top - bgd_y0; y <= bottom; y++, bgd_y++)
	{
		uchar *src = source + y*stride + left;
		for (x = left, bgd_x = left - bgd_x0; x <= right; x++, bgd_x++)
		{
			double bgd = a*bgd_x*bgd_x + b*bgd_x*bgd_y + c*bgd_y*bgd_y + d;
			double signal = (double)(*src) - bgd;
			if (signal < 0.5)
				signal = 0.5;
			*s1 += signal;
			*s2 += signal*signal;
			if (signal > result->MaxSignal)
			{
				result->MaxSignal = signal;
				result->MaxX = x;
				result->MaxY = y;
			}
			result->AvgX += signal * x;
			result->AvgY += signal * y;
			src++;
		}
	}
}
static void measureSignal_16b(uchar *source, int stride, int left, int top, int right, int bottom, int bgd_x0, int bgd_y0, double a, double b, double c, double d,
	double *s1, double *s2, struct SObjectInfo *result)
{
	int x, y, bgd_x, bgd_y;
	*s1 = *s2 = 0.0;
	for (y = top, bgd_y = top - bgd_y0; y <= bottom; y++, bgd_y++)
	{
		ushort *src = (ushort*)(source + y*stride) + left;
		for (x = left, bgd_x = left - bgd_x0; x <= right; x++, bgd_x++)
		{
			double bgd = a*bgd_x*bgd_x + b*bgd_x*bgd_y + c*bgd_y*bgd_y + d;
			double signal = (double)(*src) - bgd;
			if (signal < 0.5)
				signal = 0.5;
			*s1 += signal;
			*s2 += signal*signal;
			if (signal > result->MaxSignal)
			{
				result->MaxSignal = signal;
				result->MaxX = x;
				result->MaxY = y;
			}
			result->AvgX += signal * x;
			result->AvgY += signal * y;
			src++;
		}
	}
}
static void measureSignal_32b(uchar *source, int stride, int left, int top, int right, int bottom, int bgd_x0, int bgd_y0, double a, double b, double c, double d,
	double *s1, double *s2, struct SObjectInfo *result)
{
	int x, y, bgd_x, bgd_y;
	*s1 = *s2 = 0.0;
	for (y = top, bgd_y = top - bgd_y0; y <= bottom; y++, bgd_y++)
	{
		uint *src = (uint*)(source + y*stride) + left;
		for (x = left, bgd_x = left - bgd_x0; x <= right; x++, bgd_x++)
		{
			double bgd = a*bgd_x*bgd_x + b*bgd_x*bgd_y + c*bgd_y*bgd_y + d;
			double signal = (double)(*src) - bgd;
			if (signal < 0.5)
				signal = 0.5;
			*s1 += signal;
			*s2 += signal*signal;
			if (signal > result->MaxSignal)
			{
				result->MaxSignal = signal;
				result->MaxX = x;
				result->MaxY = y;
			}
			result->AvgX += signal * x;
			result->AvgY += signal * y;
			src++;
		}
	}
}

static int measureObject(uchar *source, int bpp, struct sSize size, int left, int top, int right, int bottom, int border, struct SObjectInfo *result)
{
	double s1 = 0, s2 = 0, n = 0;
	int bgd_l = MAX(0, left - border);
	int bgd_t = MAX(0, top - border);
	int bgd_r = MIN(size.Width - 1, right + border);
	int bgd_b = MIN(size.Height - 1, bottom + border);
	double a, b, c, d;

	if (border >= 0)
	{
		switch (bpp)
		{
		case 8:
		{
			fitBackground_8b(source, size.Stride, bgd_l, bgd_t, bgd_r, bgd_b, &a, &b, &c, &d);
			break;
		}
		case 16:
		{
			fitBackground_16b(source, size.Stride, bgd_l, bgd_t, bgd_r, bgd_b, &a, &b, &c, &d);
			break;
		}
		case 32:
		{
			fitBackground_32b(source, size.Stride, bgd_l, bgd_t, bgd_r, bgd_b, &a, &b, &c, &d);
			break;
		}
		default:
			return -1;
		}
	}
	else // do not subtract background
	{
		a = b = c = d = 0.0;
		bgd_l = bgd_t = 0;
	}

	memset(result, 0, sizeof(struct SObjectInfo));
	result->MaxSignal = -HUGE_VAL;

	result->Left = left;
	result->Top = top;
	result->Right = right;
	result->Bottom = bottom;

	result->Diameter = hypot(right - left, bottom - top);
	result->Area = (right - left) * (bottom - top);

	switch (bpp)
	{
	case 8:
	{
		measureSignal_8b(source, size.Stride, left, top, right, bottom, bgd_l, bgd_t, a, b, c, d, &s1, &s2, result);
		break;
	}
	case 16:
	{
		measureSignal_16b(source, size.Stride, left, top, right, bottom, bgd_l, bgd_t, a, b, c, d, &s1, &s2, result);
		break;
	}
	case 32:
	{
		measureSignal_32b(source, size.Stride, left, top, right, bottom, bgd_l, bgd_t, a, b, c, d, &s1, &s2, result);
		break;
	}
	default:
		return -1;
	}

	n = result->Area;
	result->Volume = s1;
	if (n > 0)
	{
		result->AvgSignal = s1 / n;
		result->StdevSignal = sqrt((s2 - s1*s1 / n) / n);
	}
	if (s1 > 0)
	{
		result->AvgX /= s1;
		result->AvgY /= s1;
	}
	return 0;
}

static void bestIntegral(const uint *integral, struct sSize sizeInt, int minSize, int maxSize, int objects_size, uint* integralMask, struct SAreaInfo *objects)
{
	double avgS = 0.0, minS = HUGE_VAL;
	int countS = 0;

	initAreaInfo(objects, objects_size);

	#if USE_OMP == 1
	#pragma omp parallel
	#endif
	{
		#if USE_OMP == 1
		int n_threads = omp_get_num_threads();
		#else
		int n_threads = 1;
		#endif
		int i, x, y;
		int d = maxSize, d2, mincnt;
		double loc_sum = 0.0, loc_min = HUGE_VAL;
		int loc_count = 0;
		int loc_size = objects_size;
		struct SAreaInfo *loc_objects = objects;
		if (n_threads > 1)
		{
			loc_objects = (struct SAreaInfo*)malloc(loc_size * sizeof(struct SAreaInfo));
			initAreaInfo(loc_objects, loc_size);
		}
		if (d < 4) d = 4;
			d2 = d / 2;
		mincnt = d * d / 2;
		#if USE_OMP == 1
		#pragma omp for
		#endif
		for (y = 0; y < sizeInt.Height - d; y += d2)
		{
			for (x = 0; x < sizeInt.Width - d; x += d2)
			{
				int cnt = 0;
				struct SAreaInfo p;
				p.x = x + d2;
				p.y = y + d2;
				p.S = integralSum(integral, sizeInt, x, y, x + d - 1, y + d - 1, &cnt);
				if (integralMask != NULL)
					cnt = integralSum(integralMask, sizeInt, x, y, x + d - 1, y + d - 1, &cnt);
				if (cnt < mincnt)
					continue;
				p.S /= sqrt((double)cnt);
				pushToHeap(loc_objects, loc_size, &p);
				loc_sum += p.S;
				loc_count++;
				if (p.S < loc_min)
					loc_min = p.S;
			}
		}
		#if USE_OMP == 1
		#pragma omp critical
		#endif
		{
			if (n_threads > 1)
			{
				for (i = 0; i < loc_size; i++)
					pushToHeap(objects, objects_size, &loc_objects[i]);
			}
			avgS += loc_sum;
			countS += loc_count;
			if (loc_min < minS)
				minS = loc_min;
		}
		if (n_threads > 1)
			free(loc_objects);
	}

	if (countS == 0)
		avgS = 1.0;
	else if (avgS == minS)
		avgS = minS + 1.0;
	else
		avgS /= (double)countS;
	
	{
		int i;
		for (i = 0; i < objects_size; i++)
			objects[i].S = (objects[i].S - minS) / (avgS - minS);
	}
}


static void bestContrast(const uint *integral, struct sSize sizeInt, int minSize, int maxSize, int objects_size, uint* integralMask, struct SAreaInfo *objects)
{
	double avgS = 0.0, minS = HUGE_VAL;
	int countS = 0;
	
	initAreaInfo(objects, objects_size);
	
	#if USE_OMP == 1
	#pragma omp parallel
	#endif
	{
		#if USE_OMP == 1
			int n_threads = omp_get_num_threads();
		#else
			int n_threads = 1;
		#endif
		int i, x, y;
		int wnd_size = maxSize, wnd_step, mincnt;
		double loc_sum = 0.0, loc_min = HUGE_VAL;
		int loc_count = 0;
		int loc_size = objects_size;
		struct SAreaInfo *loc_objects = objects;
		if (n_threads > 1)
		{
			loc_objects = (struct SAreaInfo*)malloc(loc_size * sizeof(struct SAreaInfo));
			initAreaInfo(loc_objects, loc_size);
		}
		if (wnd_size < 4)
			wnd_size = 4;
		wnd_step = wnd_size / 2;
		mincnt = ((wnd_size + 1) * (wnd_size + 1)) / 2;
		#if USE_OMP == 1
		#pragma omp for
		#endif
		for (y = 0; y < sizeInt.Height; y += wnd_step)
		{
			for (x = 0; x < sizeInt.Width; x += wnd_step)
			{
				int cnt;
				struct SAreaInfo p;
				p.x = x;
				p.y = y;
				p.S = localDifference(integral, sizeInt, x, y, wnd_size / 2, wnd_size / 2 + wnd_step, integralMask, &cnt);
				if (cnt < mincnt)
					continue;
				if (p.S < 0)
					p.S = 0.0;
				p.S *= sqrt((double)cnt);
				pushToHeap(loc_objects, loc_size, &p);
				loc_sum += p.S;
				loc_count++;
				if (p.S < loc_min)
					loc_min = p.S;
			}
		}
		#if USE_OMP == 1
		#pragma omp critical
		#endif
		{
			if (n_threads > 1)
			{
				for (i = 0; i < loc_size; i++)
					pushToHeap(objects, objects_size, &loc_objects[i]);
			}
			avgS += loc_sum;
			countS += loc_count;
			if (loc_min < minS)
				minS = loc_min;
		}
		if (n_threads > 1)
			free(loc_objects);
	}

	if (countS == 0)
		avgS = 1.0;
	else if (avgS == minS)
		avgS = minS + 1.0;
	else
		avgS /= (double)countS;
	
	int i;
#if USE_OMP == 1
#pragma omp parallel for private(i)
#endif
	for (i = 0; i < objects_size; i++)
		objects[i].S = (objects[i].S - minS) / (avgS - minS);
}

int searchObjects(uchar *processed, int width, int height, int stridePr, uchar *source, int strideSrc, uchar* mask, uint* integralMask, int bpp, uint method, int minSize, int maxSize, double minCertainty,
	int nMaxObjects, struct SObjectInfo *objects_info, int *count)
{
	int i, j;
	struct sSize sizePr, sizeSrc, sizeInt;
	uint *integral;
	int objects_size;
	struct SAreaInfo *objects;
	struct SAreaInfo tmp;

	*count = 0;

	if (width == 0 || stridePr == 0 || height == 0)
		return -1;

	nMaxObjects = nMaxObjects > N_MAX_OBJECTS ? N_MAX_OBJECTS : nMaxObjects;
	objects_size = nMaxObjects * 5 + 1;
	objects = (struct SAreaInfo*)malloc(objects_size * sizeof(struct SAreaInfo));

	sizePr.Width = sizeSrc.Width = sizeInt.Width = width;
	sizePr.Height = sizeSrc.Height = sizeInt.Height = height;
	sizePr.Stride = stridePr;
	sizeSrc.Stride = strideSrc;
	sizeInt.Stride = width;


	integral = (uint *)malloc(sizeInt.Width * sizeInt.Height * sizeof(uint));
	integralFrom(processed, sizePr.Width, sizePr.Height, sizePr.Stride, integral, bpp);

	//if (mask != NULL)
	//	multOnMask(processed, width, height, stridePr, mask, bpp);

	switch (method)
	{
	case ObjectSearch_BestIntegral:
	{
		bestIntegral(integral, sizeInt, minSize, maxSize, objects_size, integralMask, &objects[0]);
		break;
	}
	case ObjectSearch_BestContrast:
	{
		bestContrast(integral, sizeInt, minSize, maxSize, objects_size, integralMask, &objects[0]);
		break;
	}
	default:
	{
		free(objects);
		free(integral);
		return -1;
	}
	}

	for (i = 0; i < objects_size;)
	{
		if (objects[i].S < minCertainty)
		{
			objects[i] = objects[objects_size - 1];
			objects_size--;
		}
		else
			i++;
	}

	for (i = 0; i < objects_size; i++)
	{
		for (j = i + 1; j < objects_size;)
		{
			int dx = abs(objects[i].x - objects[j].x);
			int dy = abs(objects[i].y - objects[j].y);
			int dist = dx > dy ? dx : dy;
			if (dist <= maxSize * 2)
			{
				if (objects[j].S > objects[i].S)
				{
					tmp = objects[i];
					objects[i] = objects[j];
					objects[j] = tmp;
				}
				objects[j] = objects[objects_size - 1];
				objects_size--;
			}
			else
				j++;
		}
	}
	
	quickSort(objects, objects_size);
	*count = objects_size < nMaxObjects ? objects_size : nMaxObjects;

	memset(objects_info, 0, sizeof(objects_info[0]) * (*count));
	for (i = 0; i < *count; i++)
	{
		int left = MAX(0, objects[i].x - maxSize),
			top = MAX(0, objects[i].y - maxSize),
			right = MIN(sizeSrc.Width, objects[i].x + maxSize),
			bottom = MIN(sizeSrc.Height, objects[i].y + maxSize);
		//localThreshold(processed, bpp, sizePr, &left, &top, &right, &bottom);
		localizeFeature(integral, sizeInt, &left, &top, &right, &bottom);
		measureObject(source, bpp, sizeSrc, left, top, right, bottom, maxSize / 2, &objects_info[i]);
		objects_info[i].Certainty = objects[i].S;
	}
	
	free(objects);
	free(integral);
	
	return 0;
}


int searchObjectsForMultiROI(int countROI, int *ROI, uchar *processed, int width, int height, int stridePr, uchar *source, int strideSrc, uchar* mask, uint* integralMask, int bpp, uint Method,
	int minSize, int maxSize, double minCertainty, struct SObjectInfo *objects)
{
	int nMaxObjects = 1;
	int i = 0;
#if USE_OMP == 1
#pragma omp parallel for private(i)
#endif
	for (i = 0; i < countROI; i++)
	{
		int cnt = 0;
		int ox = ROI[0 + i * 4],
		    oy = ROI[1 + i * 4],
		    w  = ROI[2 + i * 4],
		    h  = ROI[3 + i * 4];
		uchar *proc = processed + oy*stridePr + ox * (bpp / 8);
		uchar *src = source + oy*strideSrc + ox * (bpp / 8);
		searchObjects(proc, w, h, stridePr, src, strideSrc, mask, integralMask, bpp, Method, minSize, maxSize, minCertainty, nMaxObjects, &objects[i], &cnt);
	}

	return 0;
}
