//#include <C:\Python37\include\Python.h>
#include "Python.h"
#if PY_MAJOR_VERSION >= 3
#  define PyIny_FromLong    PyLong_FromLong
#  define MODULE_ERROR      NULL
#  define MODULE_RETURN(v)  return (v)
#  define MODULE_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
#  define MODULE_DEF(name,doc,methods) \
    static struct PyModuleDef moduledef = { \
        PyModuleDef_HEAD_INIT, (name), (doc), -1, (methods), };
#  define MODULE_CREATE(obj,name,doc,methods) \
    obj = PyModule_Create(&moduledef);
#else /* Python 2.x */
#  define MODULE_ERROR
#  define MODULE_RETURN(v)
#  define MODULE_INIT(name) void init##name(void)
#  define MODULE_DEF(name,doc,methods)
#  define MODULE_CREATE(obj,name,doc,methods) \
    obj = Py_InitModule3((name), (methods), (doc));
#endif


#ifdef WITH_DEBUG
#define DEBUG_OUTPUT(cmd) \
    if (fdebug) \
	    { \
        cmd; \
        fflush(fdebug); \
	    }
#else
#define DEBUG_OUTPUT(cmd) (void)0;
#endif

#define LOOP_COUNT 64


#include "astrocv.h"

static PyObject *castrocv_max_threads_count(PyObject *self, PyObject *args)
{
	#if USE_OMP == 1
	return Py_BuildValue("i", omp_get_num_procs());
	#else
	return Py_BuildValue("i", 1);
	#endif
}

static PyObject *castrocv_set_threads_count(PyObject *self, PyObject *args)
{
	int threads_count;
	PyObject *ret = NULL;
	
	if (!PyArg_ParseTuple(args, "i", &threads_count))
		return ret;
	
	#if USE_OMP == 1
	if (threads_count < 1)
		threads_count = 1;
	else if (threads_count > omp_get_num_procs())
		threads_count = omp_get_num_procs();
	omp_set_num_threads(threads_count);
	#else
	threads_count = 1;
	#endif
	
	return Py_BuildValue("i", threads_count);
}

/* add(Image first_img, Image second_img, Image out_img, uint width, uint stride_A, uint height, uint stride_B, uint stride_C, int bpp) */
static PyObject *castrocv_add(PyObject *self, PyObject *args)
{
	Py_buffer buffirst, bufsec, bufout;
	PyObject *ret = NULL;
	int width, height, strideFir, strideSec, strideOut, ox, oy;
	int koef, bpp;

	if (!PyArg_ParseTuple(args, "s*s*iiiiiis*iii", &buffirst, &bufsec, &ox, &oy, &width, &height, &strideFir, &strideSec,
		&bufout, &strideOut, &koef, &bpp))
		return ret;
	switch (bpp)
	{
	case 8:
	case 12:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	Py_BEGIN_ALLOW_THREADS
	{
		uchar *first_img = (uchar*)buffirst.buf + oy*strideFir + ox * (bpp / 8);
		uchar *second_img = (uchar*)bufsec.buf + oy*strideSec + ox * (bpp / 8);
		uchar *out_img = (uchar*)bufout.buf;

		add(first_img, second_img, width, height, strideFir, strideSec, out_img, strideOut, koef, bpp);
	}
	Py_END_ALLOW_THREADS

finish:
	PyBuffer_Release(&buffirst);
	PyBuffer_Release(&bufsec);
	PyBuffer_Release(&bufout);
	return Py_BuildValue("ii", width, height);
}

/* add(Image first_img, Image second_img, Image out_img, uint width, uint stride_A, uint height, uint stride_B, uint stride_C, int bpp) */
static PyObject *castrocv_image_info(PyObject *self, PyObject *args)
{
	Py_buffer buff;
	PyObject *ret = NULL;
	uint width, height, stride;
	int ox, oy, bpp;
	double avg = 0, min = 0, max = 0, stdev = 0;

	if (!PyArg_ParseTuple(args, "s*iiiiii", &buff, &stride, &ox, &oy, &width, &height, &bpp))
		return ret;
	switch (bpp)
	{
	case 8:
	case 12:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	Py_BEGIN_ALLOW_THREADS
	{
		uchar *src = (uchar*)buff.buf + oy*stride + ox * (bpp / 8);
		imageInfo(src, width, height, stride, &avg, &min, &max, &stdev, bpp);
	}
	Py_END_ALLOW_THREADS

finish :
	PyBuffer_Release(&buff);
	return Py_BuildValue("dddd", avg, min, max, stdev);
}


/* difference(Image img, Image last_img, Image out_img, const int width, const int stride, const int height, const int bpp) */
static PyObject *castrocv_difference(PyObject *self, PyObject *args)
{

	Py_buffer bufimg, buflastimg, bufout;
	PyObject *ret = NULL;
	//int width, height, stride;
	int width, height, strideIn, stride_out, ox, oy;
	int bpp;

	if (!PyArg_ParseTuple(args, "s*s*iiiiis*ii", &bufimg, &buflastimg, &ox, &oy, &width, &strideIn, &height, &bufout, &stride_out, &bpp))
		return ret;
	switch (bpp)
	{
	case 8:
	case 12:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	Py_BEGIN_ALLOW_THREADS
	{
		uchar *img = (uchar*)bufimg.buf + oy*strideIn + ox * (bpp / 8);
		uchar *last_img = (uchar*)buflastimg.buf + oy*strideIn + ox * (bpp / 8);
		uchar *out_img = (uchar*)bufout.buf;

		difference(img, last_img, out_img, width, strideIn, height, stride_out, bpp);
	}
	Py_END_ALLOW_THREADS

finish :
	PyBuffer_Release(&bufimg);
	PyBuffer_Release(&buflastimg);
	PyBuffer_Release(&bufout);
	return Py_BuildValue("ii", width, height);
}

/* contrast(Image img, Image out_img, const int width, const int stride, const int height, int Dmin, int Dmax, const int bpp) */
static PyObject *castrocv_contrast(PyObject *self, PyObject *args)
{
	Py_buffer bufimg, bufmask, bufout;
	PyObject *ret = NULL;
	int width, height, stride_in, stride_out, ox, oy;
	int dMin, dMax;
	int bpp;

	if (!PyArg_ParseTuple(args, "s*iiiiis*s*iiii",
		&bufimg, &stride_in, &ox, &oy, &width, &height,
		&bufmask,
		&bufout, &stride_out,
		&dMin, &dMax, &bpp))
		return ret;
	switch (bpp)
	{
	case 8:
	case 12:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	Py_BEGIN_ALLOW_THREADS
	{
		uchar *img = (uchar*)bufimg.buf + oy*stride_in + ox*(bpp / 8);
	    uint *mask;
		if (bufmask.len == 0)
			mask = NULL;
		else
			mask = (uint*)bufmask.buf;
		uchar *out_img = (uchar*)bufout.buf;

		contrast(img, stride_in, width, height, out_img, stride_out, dMin, dMax, mask, bpp);
	}
	Py_END_ALLOW_THREADS

finish :
	PyBuffer_Release(&bufimg);
	PyBuffer_Release(&bufout);
	return Py_BuildValue("ii", width, height);
}

/* smooth(Image img, Image out_img, const int width, const int stride, const int height, int D, const int bpp) */
// params = [img.data, StrideIn, Ox, Oy, Width, Height, out.data, StrideOut, int(D), bpp]
static PyObject *castrocv_smooth(PyObject *self, PyObject *args)
{

	Py_buffer bufimg, bufout;
	PyObject *ret = NULL;
	int width, height, stride_in, ox, oy;
	int stride_out;
	int D;
	int bpp;

	if (!PyArg_ParseTuple(args, "s*iiiiis*iii",
	                      &bufimg, &stride_in, &ox, &oy, &width, &height,
	                      &bufout, &stride_out,
	                      &D, &bpp))
		return ret;
	switch (bpp)
	{
	case 8:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	Py_BEGIN_ALLOW_THREADS
	{
		uchar *img = (uchar*)bufimg.buf + oy*stride_in + ox*(bpp / 8);
		uchar *out_img = (uchar*)bufout.buf;

		smooth(img, stride_in, width, height, out_img, stride_out, D, bpp);
	}
	Py_END_ALLOW_THREADS

finish :
	PyBuffer_Release(&bufimg);
	PyBuffer_Release(&bufout);
	return Py_BuildValue("ii", width, height);
}

/* calibrationFrom(Image Source, int width, int stride_src, int height, Image out_img, int stride_out, int Smoothed, const int bpp) */
static PyObject *castrocv_calibrationFrom(PyObject *self, PyObject *args)
{

	Py_buffer bufimg, bufout;
	PyObject *ret = NULL;
	int width, height, strideIn, strideOut, ox, oy;
	int Smoothed;
	int bpp;

	if (!PyArg_ParseTuple(args, "s*iiiiis*iii", &bufimg, &strideIn, &ox, &oy, &width, &height, &bufout, &strideOut, &Smoothed, &bpp))
		return ret;
	switch (bpp)
	{
	case 8:
	case 12:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	Py_BEGIN_ALLOW_THREADS
	{
		uchar *src = (uchar*)bufimg.buf + oy*strideIn + ox*(bpp / 8);
		uchar *out = (uchar*)bufout.buf;
		calibrationFrom(src, strideIn, width, height, out, strideOut, Smoothed, bpp);
	}
Py_END_ALLOW_THREADS

	finish :
	PyBuffer_Release(&bufimg);
	PyBuffer_Release(&bufout);
	return Py_BuildValue("ii", width, height);
}

/* applyCalibration(Image img_src, int width, int stride_src, int height, Image Calibration, int stride_calib, Image out_img, int stride_out, const int bpp) */
static PyObject *castrocv_applyCalibration(PyObject *self, PyObject *args)
{

	Py_buffer bufsrc, bufcalib, bufout;
	PyObject *ret = NULL;
	int width, height, strideSrc, ox, oy;
	int bpp;

	if (!PyArg_ParseTuple(args, "s*iiiiis*s*i", &bufsrc, &strideSrc, &ox, &oy, &width, &height, &bufcalib, &bufout, &bpp))
		return ret;
	switch (bpp)
	{
	case 8:
	case 12:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	Py_BEGIN_ALLOW_THREADS
	{
		uchar *src = (uchar*)bufsrc.buf + oy*strideSrc + ox*(bpp / 8);
		uchar *calib = (uchar*)bufcalib.buf;
		uchar *out = (uchar*)bufout.buf;
		applyCalibration(src, strideSrc, width, height, calib, out, bpp);
	}
	Py_END_ALLOW_THREADS

		finish :
	PyBuffer_Release(&bufsrc);
	PyBuffer_Release(&bufcalib);
	PyBuffer_Release(&bufout);
	return Py_BuildValue("ii", width, height);
}

/* searchObjects(Image Processed, int width, int stridePr, int height, Image Source, int strideSrc, uint Method, int MinSize, int MaxSize, double MinCertainty, int NMaxObjects, int *Count, double *Certainty, double *X, double *Y,
	__int32 *Left, __int32 *Top, __int32 *Right, __int32 *Bottom, double *Diameter, double *Area, double *Volume, double *RelativeVolume, double *AvgSignal, double *MaxSignal, int bpp) */
static PyObject *castrocv_searchObjects(PyObject *self, PyObject *args)
{
	int i;
	Py_buffer bufproc, bufsrc, bufmask, bufintmask;
	PyObject *ret = NULL, *result = NULL;
	int width = 0, height = 0, stride_pr = 0, stride_src = 0, MinSize = 0, MaxSize = 0, NMaxObjects = 0, Count = 0;
	struct SObjectInfo *objects = NULL;
	double MinCertainty=0;
	uint Method=0;
	int ox, oy, bpp=0;

	if (!PyArg_ParseTuple(args, "s*iiiiis*is*s*Iiidii", &bufproc, &ox, &oy, &width, &stride_pr, &height, &bufsrc, &stride_src,
													  &bufmask, &bufintmask,
		                                              &Method, &MinSize, &MaxSize, &MinCertainty, &NMaxObjects, &bpp))
		return ret;
	switch (bpp)
	{
	case 8:
	case 12:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	objects = (struct SObjectInfo *)malloc(NMaxObjects * sizeof(struct SObjectInfo));

	Py_BEGIN_ALLOW_THREADS
	{
		uchar *proc = (uchar*)bufproc.buf + oy*stride_pr + ox*(bpp / 8);
		uchar *src = (uchar*)bufsrc.buf + oy*stride_src + ox*(bpp / 8);
		uchar *mask;
		uint *integralMask;
		if (bufmask.len == 0)
		{
			mask = NULL;
			integralMask = NULL;
		}
		else
		{
			mask = (uchar*)bufmask.buf;
			if (bufintmask.len == 0)
			{
				integralMask = NULL;
				mask = NULL;
			}
			else
				integralMask = (uint*)bufintmask.buf;
		}



		searchObjects(proc, width, height, stride_pr, src, stride_src, mask, integralMask, bpp,
					  Method, MinSize, MaxSize, MinCertainty, NMaxObjects, objects, &Count);
	}
	Py_END_ALLOW_THREADS

	result = PyTuple_New(Count);

	for (i = 0; i < Count; i++)
	{
		PyTuple_SetItem(result, i, Py_BuildValue(
			"iiii"
			"d"
			"dddd"
			"ddd"
			"ddd",
			objects[i].Left, objects[i].Top, objects[i].Right, objects[i].Bottom,
			objects[i].Certainty,
			objects[i].AvgX, objects[i].AvgY, objects[i].MaxX, objects[i].MaxY,
			objects[i].Diameter, objects[i].Area, objects[i].Volume,
			objects[i].AvgSignal, objects[i].StdevSignal, objects[i].MaxSignal));
	}
	
	ret = Py_BuildValue("iOii", Count, result, width, height);
	Py_XDECREF(result);

	free(objects);

finish :

	PyBuffer_Release(&bufsrc);
	PyBuffer_Release(&bufproc);
	return ret;
}


/* int searchObjectsForMultiROI(int countROI, int *ROI, uchar *Processed, int width, int stridePr, int height, uchar *source, int strideSrc, uint method,
		int minSize, int maxSize, double minCertainty, struct SObjectInfo *objects, int bpp);*/
static PyObject *castrocv_searchObjectsForMultiROI(PyObject *self, PyObject *args)
{

	Py_buffer bufproc, bufsrc, bufROI, bufmask, bufintmask;
	PyObject *ret = NULL, *result = NULL;
	int width = 0, height = 0, stride_pr = 0, stride_src = 0, MinSize = 0, MaxSize = 0;
	double MinCertainty = 0;
	uint Method = 0;
	int i, countROI, bpp = 0;
	struct SObjectInfo *objects;
	
	if (!PyArg_ParseTuple(args, "is*s*iiis*is*s*Iiidi", &countROI, &bufROI, &bufproc, &width, &stride_pr, &height, &bufsrc, &stride_src, &bufmask, &bufintmask, &Method, &MinSize, &MaxSize, &MinCertainty, &bpp))
		return ret;
	switch (bpp)
	{
	case 8:
	case 12:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	objects = (struct SObjectInfo *)malloc(countROI * sizeof(struct SObjectInfo));
	memset(objects, 0, countROI * sizeof(struct SObjectInfo));

	Py_BEGIN_ALLOW_THREADS
	{
		
		uchar *proc = (uchar*)bufproc.buf;
		uchar *src = (uchar*)bufsrc.buf;
		uchar *mask;
		uint *integralMask;
		if (bufmask.len == 0)
		{
			mask = NULL;
			integralMask = NULL;
		}
		else
		{
			mask = (uchar*)bufmask.buf;
			if (bufintmask.len == 0)
			{
				integralMask = NULL;
				mask = NULL;
			}
			else
				integralMask = (uint*)bufintmask.buf;
		}
		searchObjectsForMultiROI(countROI, (int*)bufROI.buf, proc, width, height, stride_pr, src, stride_src, mask, integralMask, bpp, Method, MinSize, MaxSize, MinCertainty, objects);
	}
	Py_END_ALLOW_THREADS

	result = PyTuple_New(countROI);

	for (i = 0; i < countROI; i++)
	{
		PyTuple_SetItem(result, i, Py_BuildValue(
			"iiii"
			"d"
			"dddd"
			"ddd"
			"ddd",
			objects[i].Left, objects[i].Top, objects[i].Right, objects[i].Bottom,
			objects[i].Certainty,
			objects[i].AvgX, objects[i].AvgY, objects[i].MaxX, objects[i].MaxY,
			objects[i].Diameter, objects[i].Area, objects[i].Volume,
			objects[i].AvgSignal, objects[i].StdevSignal, objects[i].MaxSignal));
	}


	ret = Py_BuildValue("Oii", result, width, height);
	Py_XDECREF(result);

	free(objects);

finish:

	PyBuffer_Release(&bufsrc);
	PyBuffer_Release(&bufproc);
	PyBuffer_Release(&bufROI);
	return ret;
}

static PyObject *castrocv_addNoiseUniform(PyObject *self, PyObject *args)
{
	Py_buffer bufimg;
	PyObject *ret = NULL;
	int width, height, stride;
	int background, noise_min, noise_max, bpp;

	if (!PyArg_ParseTuple(args, "s*iiiiiii", &bufimg, &width, &stride, &height, &background, &noise_min, &noise_max, &bpp))
		return ret;
	switch (bpp)
	{
	case 8:
	case 12:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	Py_BEGIN_ALLOW_THREADS
	{
		uchar *img = (uchar*)bufimg.buf;

		addNoiseUniform(img, width, height, stride, background, noise_min, noise_max, bpp);
	}
	Py_END_ALLOW_THREADS

finish :
	PyBuffer_Release(&bufimg);
	return Py_BuildValue("ii", width, height);
}

static PyObject *castrocv_addNoiseNorm(PyObject *self, PyObject *args)
{
	Py_buffer bufimg;
	PyObject *ret = NULL;
	int width, height, stride;
	int M, bpp;
	double S;

	if (!PyArg_ParseTuple(args, "s*iiiidi", &bufimg, &width, &stride, &height, &M, &S, &bpp))
		return ret;
	switch (bpp)
	{
	case 8:
	case 12:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	Py_BEGIN_ALLOW_THREADS
	{
		uchar *img = (uchar*)bufimg.buf;

		addNoiseNorm(img, width, height, stride, M, S, bpp);
	}
	Py_END_ALLOW_THREADS

finish :
	PyBuffer_Release(&bufimg);
	return Py_BuildValue("ii", width, height);
}

static PyObject *castrocv_drawObject(PyObject *self, PyObject *args)
{
	Py_buffer bufimg;
	PyObject *ret = NULL;
	int width, height, stride;
	int bpp;
	int x, y, signal, radius;
	int intens = 0;

	if (!PyArg_ParseTuple(args, "s*iiiiiiii", &bufimg, &width, &stride, &height, &x, &y, &signal, &radius, &bpp))
		return ret;
	switch (bpp)
	{
	case 8:
	case 12:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	Py_BEGIN_ALLOW_THREADS
	{
		uchar *img = (uchar*)bufimg.buf;

	drawObject(img, width, height, stride, x, y, signal, radius, bpp);
	}
	Py_END_ALLOW_THREADS

finish :
	PyBuffer_Release(&bufimg);
	return Py_BuildValue("i", intens);
}


// addMarker(Image img, int width, int stride, int height, double X, double Y, double Radius, int byteOnPixel)
static PyObject *castrocv_addMarker(PyObject *self, PyObject *args)
{
	Py_buffer bufimg;
	PyObject *ret = NULL;
	int width, height, stride;
	int bpp;
	int x, y, left, top, right, bottom;

	if (!PyArg_ParseTuple(args, "s*iiiiiiiiii", &bufimg, &width, &stride, &height, &x, &y, &left, &top, &right, &bottom, &bpp))
		return ret;
	switch (bpp)
	{
	case 8:
	case 12:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	Py_BEGIN_ALLOW_THREADS
	{
		uchar *img = (uchar*)bufimg.buf;

		addMarker(img, width, height, stride, x, y, left, top, right, bottom, bpp);
	}
	Py_END_ALLOW_THREADS

finish :
	PyBuffer_Release(&bufimg);
	return Py_BuildValue("ii", width, height);
}

//void upsample(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, int bpp, int factor);
static PyObject *castrocv_upsample(PyObject *self, PyObject *args)
{
	Py_buffer bufsrc, bufdst;
	PyObject *ret = NULL;
	int ox, oy, width, height, stride_src, stride_dst;
	int bpp;
	int factor;

	if (!PyArg_ParseTuple(args, "s*is*iiiiiii", &bufdst, &stride_dst, &bufsrc, &stride_src, &ox, &oy, &width, &height, &bpp, &factor))
		return ret;
	switch (bpp)
	{
	case 8:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	Py_BEGIN_ALLOW_THREADS
	{
		void *src = (void*)((char*)bufsrc.buf + oy*stride_src + ox*(bpp/8));
		void *dst = (void*)bufdst.buf;
		
		upsample(dst, stride_dst, src, stride_src, width, height, bpp, factor);
	}
	Py_END_ALLOW_THREADS

finish:
	PyBuffer_Release(&bufsrc);
	PyBuffer_Release(&bufdst);
	return Py_BuildValue("ii", width, height);
}

//void downsample(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, int bpp, int factor);
static PyObject *castrocv_downsample(PyObject *self, PyObject *args)
{
	Py_buffer bufsrc, bufdst;
	PyObject *ret = NULL;
	int ox, oy, width, height, stride_src, stride_dst;
	int bpp;
	int factor;

	if (!PyArg_ParseTuple(args, "s*is*iiiiiii", &bufdst, &stride_dst, &bufsrc, &stride_src, &ox, &oy, &width, &height, &bpp, &factor))
		return ret;
	switch (bpp)
	{
	case 8:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	Py_BEGIN_ALLOW_THREADS
	{
		void *src = (void*)((char*)bufsrc.buf + oy*stride_src + ox*(bpp/8));
		void *dst = (void*)bufdst.buf;
		
		downsample(dst, stride_dst, src, stride_src, width, height, bpp, factor);
	}
	Py_END_ALLOW_THREADS

finish:
	PyBuffer_Release(&bufsrc);
	PyBuffer_Release(&bufdst);
	return Py_BuildValue("ii", width, height);
}

//void convolve(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, int bpp, const double *kernel, int kwidth, int kheight);
static PyObject *castrocv_convolve(PyObject *self, PyObject *args)
{
	Py_buffer bufsrc, bufdst, bufker;
	PyObject *ret = NULL;
	int ox, oy, width, height, stride_src, stride_dst, kwidth, kheight;
	int bpp;

	if (!PyArg_ParseTuple(args, "s*is*iiiiiis*ii", &bufdst, &stride_dst, &bufsrc, &stride_src, &ox, &oy, &width, &height, &bpp, &bufker, &kwidth, &kheight))
		return ret;
	switch (bpp)
	{
	case 8:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	Py_BEGIN_ALLOW_THREADS
	{
		void *src = (void*)((char*)bufsrc.buf + oy*stride_src + ox*(bpp/8));
		void *dst = (void*)bufdst.buf;
		double *kernel = (double*)bufker.buf;
		
		convolve(dst, stride_dst, src, stride_src, width, height, bpp, kernel, kwidth, kheight);
	}
	Py_END_ALLOW_THREADS

finish:
	PyBuffer_Release(&bufsrc);
	PyBuffer_Release(&bufdst);
	PyBuffer_Release(&bufker);
	return Py_BuildValue("ii", width, height);
}

//int			energyDistribution(uchar *img, int width, int height, int stride, int bpp, int x0, int y0, const double *R, double *I, int length);
static PyObject *castrocv_energyDistribution(PyObject *self, PyObject *args)
{
	Py_buffer bufsrc, bufR, bufI;
	PyObject *ret = NULL;
	int ox, oy, width, height, stride;
	double x0, y0;
	int bpp;

	if (!PyArg_ParseTuple(args, "s*iiiiiidds*s*", &bufsrc, &stride, &ox, &oy, &width, &height, &bpp, &x0, &y0, &bufR, &bufI))
		return ret;
	switch (bpp)
	{
	case 8:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	Py_BEGIN_ALLOW_THREADS
	{
		void *src = (void*)((char*)bufsrc.buf + oy*stride + ox*(bpp/8));
		const double *R = (const double*)bufR.buf;
		double *I = (double*)bufI.buf;
		int length = MIN(bufR.len, bufI.len) / sizeof(double);
		
		energyDistribution(src, width, height, stride, bpp, x0, y0, R, I, length);
	}
	Py_END_ALLOW_THREADS

finish:
	PyBuffer_Release(&bufsrc);
	PyBuffer_Release(&bufR);
	PyBuffer_Release(&bufI);
	return Py_BuildValue("ii", width, height);
}

//int			powerDistribution(uchar *img, int width, int height, int stride, int bpp, int x0, int y0, const double *R, double *I, int length);
static PyObject *castrocv_powerDistribution(PyObject *self, PyObject *args)
{
	Py_buffer bufsrc, bufR, bufI;
	PyObject *ret = NULL;
	int ox, oy, width, height, stride;
	double x0, y0;
	int bpp;

	if (!PyArg_ParseTuple(args, "s*iiiiiidds*s*", &bufsrc, &stride, &ox, &oy, &width, &height, &bpp, &x0, &y0, &bufR, &bufI))
		return ret;
	switch (bpp)
	{
	case 8:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}

	Py_BEGIN_ALLOW_THREADS
	{
		void *src = (void*)((char*)bufsrc.buf + oy*stride + ox*(bpp/8));
		const double *R = (const double*)bufR.buf;
		double *I = (double*)bufI.buf;
		int length = MIN(bufR.len, bufI.len) / sizeof(double);
		
		powerDistribution(src, width, height, stride, bpp, x0, y0, R, I, length);
	}
	Py_END_ALLOW_THREADS

finish:
	PyBuffer_Release(&bufsrc);
	PyBuffer_Release(&bufR);
	PyBuffer_Release(&bufI);
	return Py_BuildValue("ii", width, height);
}

//void		integralFrom(char *source, struct sSize size, uint *integral);
static PyObject *castrocv_integralFrom(PyObject *self, PyObject *args)
{
	Py_buffer bufsrc, bufIntegral;
	PyObject *ret = NULL;
	int width, height, stride, bpp;

	if (!PyArg_ParseTuple(args, "s*s*iiii", &bufsrc, &bufIntegral, &width, &height, &stride, &bpp))
		return ret;
	switch (bpp)
	{
	case 8:
	case 16:
	case 32:
		break;
	default:
		goto finish;
	}
	Py_BEGIN_ALLOW_THREADS
	{
	    uchar* src = (uchar*)bufsrc.buf;
	    uint* integral = (uint*)bufIntegral.buf;

		integralFrom(src, width, height, stride, integral, bpp);
	}
	Py_END_ALLOW_THREADS

finish:
	PyBuffer_Release(&bufsrc);
	PyBuffer_Release(&bufIntegral);
	return Py_BuildValue("ii", width, height);
}

// TEST fun
static PyObject *castrocv_test(PyObject *self, PyObject *args)
{
	int len = 0;
	Py_BEGIN_ALLOW_THREADS
	{
		len = 1234567890;
	}
		Py_END_ALLOW_THREADS

		return Py_BuildValue("i", len);
}

static PyMethodDef methods[] = {
	{ "max_threads_count", (PyCFunction)castrocv_max_threads_count, METH_VARARGS,
	"Returns maximum number of OMP threads" },
	{ "set_threads_count", (PyCFunction)castrocv_set_threads_count, METH_VARARGS,
	"Sets number of OMP threads to use" },
	{ "add", (PyCFunction)castrocv_add, METH_VARARGS,
	"Returns sum image" },
	{ "difference", (PyCFunction)castrocv_difference, METH_VARARGS,
	"Returns difference images" },
	{ "contrast", (PyCFunction)castrocv_contrast, METH_VARARGS,
	"Returns contrast image" },
	{ "smooth", (PyCFunction)castrocv_smooth, METH_VARARGS,
	"Returns smooth image" },
	{ "calibrationFrom", (PyCFunction)castrocv_calibrationFrom, METH_VARARGS,
	"Create new calibration image" },
	{ "applyCalibration", (PyCFunction)castrocv_applyCalibration, METH_VARARGS,
	"Apply calibration image" },
	{ "searchObjects", (PyCFunction)castrocv_searchObjects, METH_VARARGS,
	"Search Objects" },
	{ "searchObjectsForMultiROI", (PyCFunction)castrocv_searchObjectsForMultiROI, METH_VARARGS,
	"Search Objects Multi ROI" },
	{ "addNoiseUniform", (PyCFunction)castrocv_addNoiseUniform, METH_VARARGS,
	"add noise" },
	{ "addNoiseNorm", (PyCFunction)castrocv_addNoiseNorm, METH_VARARGS,
	"add noise normal distribution" },
	{ "drawObject", (PyCFunction)castrocv_drawObject, METH_VARARGS,
	"draw fill object" },
	{ "addMarker", (PyCFunction)castrocv_addMarker, METH_VARARGS,
	"add Marker" },
	{ "image_info", (PyCFunction)castrocv_image_info, METH_VARARGS,
	"image info" },
	{"upsample", (PyCFunction)castrocv_upsample, METH_VARARGS,
	"updample"},
	{"downsample", (PyCFunction)castrocv_downsample, METH_VARARGS,
	"downdample"},
	{"convolve", (PyCFunction)castrocv_convolve, METH_VARARGS,
	"Convolution"},
	{"energyDistribution", (PyCFunction)castrocv_energyDistribution, METH_VARARGS,
	"Distribution of energy in the given area"},
	{"powerDistribution", (PyCFunction)castrocv_powerDistribution, METH_VARARGS,
	"Distribution of power (density) in the given area"},
	{"integralFrom", (PyCFunction)castrocv_integralFrom, METH_VARARGS,
	"integral From"},
	{ "TEST", (PyCFunction)castrocv_test, METH_VARARGS,
	"test function" },
	{ NULL, NULL, 0, NULL }
};



MODULE_DEF("castrocv", NULL, methods);

MODULE_INIT(castrocv)
{
	PyObject *m;
	MODULE_CREATE(m, "castrocv", NULL, methods)
		if (!m)
			return MODULE_ERROR;
	MODULE_RETURN(m);
}
