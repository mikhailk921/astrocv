#include "astrocv.h"

//#define NO_OPTIMIZATION

static void convolve_generic(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, int bpp, const double *kernel, int kwidth, int kheight);
static void convolve_3x3_8b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel);
static void convolve_3x3_16b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel);
static void convolve_3x3_32b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel);
static void convolve_5x5_8b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel);
static void convolve_5x5_16b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel);
static void convolve_5x5_32b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel);
static void convolve_7x7_8b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel);
static void convolve_7x7_16b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel);
static void convolve_7x7_32b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel);
static void convolve_mxn_8b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel, int kwidth, int kheight);
static void convolve_mxn_16b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel, int kwidth, int kheight);
static void convolve_mxn_32b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel, int kwidth, int kheight);

void convolve(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, int bpp, const double *kernel, int kwidth, int kheight)
{
#ifndef NO_OPTIMIZATION
    if (kwidth == 3 && kheight == 3)
    {
        switch (bpp)
        {
        case 8:
            convolve_3x3_8b(dst, stride_dst, src, stride_src, width, height, kernel);
            return;
        case 16:
            convolve_3x3_16b(dst, stride_dst, src, stride_src, width, height, kernel);
            return;
        case 32:
            convolve_3x3_32b(dst, stride_dst, src, stride_src, width, height, kernel);
            return;
        }
    }
    if (kwidth == 5 && kheight == 5)
    {
        switch (bpp)
        {
        case 8:
            convolve_5x5_8b(dst, stride_dst, src, stride_src, width, height, kernel);
            return;
        case 16:
            convolve_5x5_16b(dst, stride_dst, src, stride_src, width, height, kernel);
            return;
        case 32:
            convolve_5x5_32b(dst, stride_dst, src, stride_src, width, height, kernel);
            return;
        }
    }
    if (kwidth == 7 && kheight == 7)
    {
        switch (bpp)
        {
        case 8:
            convolve_7x7_8b(dst, stride_dst, src, stride_src, width, height, kernel);
            return;
        case 16:
            convolve_7x7_16b(dst, stride_dst, src, stride_src, width, height, kernel);
            return;
        case 32:
            convolve_7x7_32b(dst, stride_dst, src, stride_src, width, height, kernel);
            return;
        }
    }
    switch (bpp)
    {
    case 8:
        convolve_mxn_8b(dst, stride_dst, src, stride_src, width, height, kernel, kwidth, kheight);
        return;
    case 16:
        convolve_mxn_16b(dst, stride_dst, src, stride_src, width, height, kernel, kwidth, kheight);
        return;
    case 32:
        convolve_mxn_32b(dst, stride_dst, src, stride_src, width, height, kernel, kwidth, kheight);
        return;
    }
#endif

    convolve_generic(dst, stride_dst, src, stride_src, width, height, bpp, kernel, kwidth, kheight);
}

static void convolve_generic(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, int bpp, const double *kernel, int kwidth, int kheight)
{
    int x, y, dx, dy;
    int x0 = kwidth / 2;
    int y0 = kheight / 2;
    
    switch (bpp)
    {
    case 8:
    {
        for (y = y0; y < height - y0; y++)
        {
            for (x = x0; x < width - x0; x++)
            {
                uchar *dst_ptr = (uchar*)((char*)dst + y*stride_dst) + x;
                double value = 0.0;
                for (dy = -y0; dy <= +y0; dy++)
                {
                    const uchar *src_ptr = (const uchar*)((char*)src + (y + dy)*stride_src) + x - x0;
                    for (dx = -x0; dx <= +x0; dx++)
                        value += kernel[(y0 + dy) * kwidth + (x0 + dx)] *
                                 (double)(*src_ptr++);
                }
                *dst_ptr = (uchar)SATURATE(value, (double)0xff);
            }
        }
        break;
    }
    case 16:
    {
        for (y = y0; y < height - y0; y++)
        {
            for (x = x0; x < width - x0; x++)
            {
                ushort *dst_ptr = (ushort*)((char*)dst + y*stride_dst) + x;
                double value = 0.0;
                for (dy = -y0; dy <= +y0; dy++)
                {
                    const ushort *src_ptr = (const ushort*)((char*)src + (y + dy)*stride_src) + x - x0;
                    for (dx = -x0; dx <= +x0; dx++)
                        value += kernel[(y0 + dy) * kwidth + (x0 + dx)] *
                                 (double)(*src_ptr++);
                }
                *dst_ptr = (ushort)SATURATE(value, (double)0xffff);
            }
        }
        break;
    }
    case 32:
    {
        for (y = y0; y < height - y0; y++)
        {
            for (x = x0; x < width - x0; x++)
            {
                uint *dst_ptr = (uint*)((char*)dst + y*stride_dst) + x;
                double value = 0.0;
                for (dy = -y0; dy <= +y0; dy++)
                {
                    const uint *src_ptr = (const uint*)((char*)src + (y + dy)*stride_src) + x - x0;
                    for (dx = -x0; dx <= +x0; dx++)
                        value += kernel[(y0 + dy) * kwidth + (x0 + dx)] *
                                 (double)(*src_ptr++);
                }
                *dst_ptr = (uint)SATURATE(value, (double)0xffffffff);
            }
        }
        break;
    }
    }
}

static void convolve_3x3_8b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel)
{
    const double k11 = kernel[0], k12 = kernel[1], k13 = kernel[2];
    const double k21 = kernel[3], k22 = kernel[4], k23 = kernel[5];
    const double k31 = kernel[6], k32 = kernel[7], k33 = kernel[8];
    int x, y;
    
    #if USE_OMP == 1
    #pragma omp parallel for private(x, y)
    #endif
    for (y = 1; y < height - 1; y++)
    {
        const uchar *src_1 = (const uchar*)((char*)src + (y - 1) * stride_src);
        const uchar *src_2 = (const uchar*)((char*)src + (y    ) * stride_src);
        const uchar *src_3 = (const uchar*)((char*)src + (y + 1) * stride_src);
        uchar *dst_row = (uchar*)((char*)dst + y*stride_dst) + 1;

        for (x = 1; x < width - 1; x++)
        {
            double value = src_1[0] * k11 + src_1[1] * k12 + src_1[2] * k13 +
                           src_2[0] * k21 + src_2[1] * k22 + src_2[2] * k23 +
                           src_3[0] * k31 + src_3[1] * k32 + src_3[2] * k33 ;
            *(dst_row++) = (uchar)SATURATE(value, (double)0xff);
            src_1++;
            src_2++;
            src_3++;
        }
    }
}

static void convolve_3x3_16b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel)
{
    const double k11 = kernel[0], k12 = kernel[1], k13 = kernel[2];
    const double k21 = kernel[3], k22 = kernel[4], k23 = kernel[5];
    const double k31 = kernel[6], k32 = kernel[7], k33 = kernel[8];
    int x, y;
    
    #if USE_OMP == 1
    #pragma omp parallel for private(x, y)
    #endif
    for (y = 1; y < height - 1; y++)
    {
        const ushort *src_1 = (const ushort*)((char*)src + (y - 1) * stride_src);
        const ushort *src_2 = (const ushort*)((char*)src + (y    ) * stride_src);
        const ushort *src_3 = (const ushort*)((char*)src + (y + 1) * stride_src);
        ushort *dst_row = (ushort*)((char*)dst + y*stride_dst) + 1;

        for (x = 1; x < width - 1; x++)
        {
            double value = src_1[0] * k11 + src_1[1] * k12 + src_1[2] * k13 +
                           src_2[0] * k21 + src_2[1] * k22 + src_2[2] * k23 +
                           src_3[0] * k31 + src_3[1] * k32 + src_3[2] * k33 ;
            *(dst_row++) = (ushort)SATURATE(value, (double)0xffff);
            src_1++;
            src_2++;
            src_3++;
        }
    }
}

static void convolve_3x3_32b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel)
{
    const double k11 = kernel[0], k12 = kernel[1], k13 = kernel[2];
    const double k21 = kernel[3], k22 = kernel[4], k23 = kernel[5];
    const double k31 = kernel[6], k32 = kernel[7], k33 = kernel[8];
    int x, y;
    
    #if USE_OMP == 1
    #pragma omp parallel for private(x, y)
    #endif
    for (y = 1; y < height - 1; y++)
    {
        const uint *src_1 = (const uint*)((char*)src + (y - 1) * stride_src);
        const uint *src_2 = (const uint*)((char*)src + (y    ) * stride_src);
        const uint *src_3 = (const uint*)((char*)src + (y + 1) * stride_src);
        uint *dst_row = (uint*)((char*)dst + y*stride_dst) + 1;

        for (x = 1; x < width - 1; x++)
        {
            double value = src_1[0] * k11 + src_1[1] * k12 + src_1[2] * k13 +
                           src_2[0] * k21 + src_2[1] * k22 + src_2[2] * k23 +
                           src_3[0] * k31 + src_3[1] * k32 + src_3[2] * k33 ;
            *(dst_row++) = (uint)SATURATE(value, (double)0xffffffff);
            src_1++;
            src_2++;
            src_3++;
        }
    }
}

static void convolve_5x5_8b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel)
{
    const double k11 = kernel[ 0], k12 = kernel[ 1], k13 = kernel[ 2], k14 = kernel[ 3], k15 = kernel[ 4];
    const double k21 = kernel[ 5], k22 = kernel[ 6], k23 = kernel[ 7], k24 = kernel[ 8], k25 = kernel[ 9];
    const double k31 = kernel[10], k32 = kernel[11], k33 = kernel[12], k34 = kernel[13], k35 = kernel[14];
    const double k41 = kernel[15], k42 = kernel[16], k43 = kernel[17], k44 = kernel[18], k45 = kernel[19];
    const double k51 = kernel[20], k52 = kernel[21], k53 = kernel[22], k54 = kernel[23], k55 = kernel[24];
    int x, y;
    
    #if USE_OMP == 1
    #pragma omp parallel for private(x, y)
    #endif
    for (y = 2; y < height - 2; y++)
    {
        const uchar *src_1 = (const uchar*)((char*)src + (y - 2) * stride_src);
        const uchar *src_2 = (const uchar*)((char*)src + (y - 1) * stride_src);
        const uchar *src_3 = (const uchar*)((char*)src + (y    ) * stride_src);
        const uchar *src_4 = (const uchar*)((char*)src + (y + 1) * stride_src);
        const uchar *src_5 = (const uchar*)((char*)src + (y + 2) * stride_src);
        uchar *dst_row = (uchar*)((char*)dst + y*stride_dst) + 2;

        for (x = 2; x < width - 2; x++)
        {
            double value = src_1[0] * k11 + src_1[1] * k12 + src_1[2] * k13 + src_1[3] * k14 + src_1[4] * k15 +
                           src_2[0] * k21 + src_2[1] * k22 + src_2[2] * k23 + src_2[3] * k24 + src_2[4] * k25 +
                           src_3[0] * k31 + src_3[1] * k32 + src_3[2] * k33 + src_3[3] * k34 + src_3[4] * k35 +
                           src_4[0] * k41 + src_4[1] * k42 + src_4[2] * k43 + src_4[3] * k44 + src_4[4] * k45 +
                           src_5[0] * k51 + src_5[1] * k52 + src_5[2] * k53 + src_5[3] * k54 + src_5[4] * k55 ;
            *(dst_row++) = (uchar)SATURATE(value, (double)0xff);
            src_1++;
            src_2++;
            src_3++;
            src_4++;
            src_5++;
        }
    }
}

static void convolve_5x5_16b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel)
{
    const double k11 = kernel[ 0], k12 = kernel[ 1], k13 = kernel[ 2], k14 = kernel[ 3], k15 = kernel[ 4];
    const double k21 = kernel[ 5], k22 = kernel[ 6], k23 = kernel[ 7], k24 = kernel[ 8], k25 = kernel[ 9];
    const double k31 = kernel[10], k32 = kernel[11], k33 = kernel[12], k34 = kernel[13], k35 = kernel[14];
    const double k41 = kernel[15], k42 = kernel[16], k43 = kernel[17], k44 = kernel[18], k45 = kernel[19];
    const double k51 = kernel[20], k52 = kernel[21], k53 = kernel[22], k54 = kernel[23], k55 = kernel[24];
    int x, y;
    
    #if USE_OMP == 1
    #pragma omp parallel for private(x, y)
    #endif
    for (y = 2; y < height - 2; y++)
    {
        const ushort *src_1 = (const ushort*)((char*)src + (y - 2) * stride_src);
        const ushort *src_2 = (const ushort*)((char*)src + (y - 1) * stride_src);
        const ushort *src_3 = (const ushort*)((char*)src + (y    ) * stride_src);
        const ushort *src_4 = (const ushort*)((char*)src + (y + 1) * stride_src);
        const ushort *src_5 = (const ushort*)((char*)src + (y + 2) * stride_src);
        ushort *dst_row = (ushort*)((char*)dst + y*stride_dst) + 2;

        for (x = 2; x < width - 2; x++)
        {
            double value = src_1[0] * k11 + src_1[1] * k12 + src_1[2] * k13 + src_1[3] * k14 + src_1[4] * k15 +
                           src_2[0] * k21 + src_2[1] * k22 + src_2[2] * k23 + src_2[3] * k24 + src_2[4] * k25 +
                           src_3[0] * k31 + src_3[1] * k32 + src_3[2] * k33 + src_3[3] * k34 + src_3[4] * k35 +
                           src_4[0] * k41 + src_4[1] * k42 + src_4[2] * k43 + src_4[3] * k44 + src_4[4] * k45 +
                           src_5[0] * k51 + src_5[1] * k52 + src_5[2] * k53 + src_5[3] * k54 + src_5[4] * k55 ;
            *(dst_row++) = (ushort)SATURATE(value, (double)0xffff);
            src_1++;
            src_2++;
            src_3++;
            src_4++;
            src_5++;
        }
    }
}

static void convolve_5x5_32b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel)
{
    const double k11 = kernel[ 0], k12 = kernel[ 1], k13 = kernel[ 2], k14 = kernel[ 3], k15 = kernel[ 4];
    const double k21 = kernel[ 5], k22 = kernel[ 6], k23 = kernel[ 7], k24 = kernel[ 8], k25 = kernel[ 9];
    const double k31 = kernel[10], k32 = kernel[11], k33 = kernel[12], k34 = kernel[13], k35 = kernel[14];
    const double k41 = kernel[15], k42 = kernel[16], k43 = kernel[17], k44 = kernel[18], k45 = kernel[19];
    const double k51 = kernel[20], k52 = kernel[21], k53 = kernel[22], k54 = kernel[23], k55 = kernel[24];
    int x, y;
    
    #if USE_OMP == 1
    #pragma omp parallel for private(x, y)
    #endif
    for (y = 2; y < height - 2; y++)
    {
        const uint *src_1 = (const uint*)((char*)src + (y - 2) * stride_src);
        const uint *src_2 = (const uint*)((char*)src + (y - 1) * stride_src);
        const uint *src_3 = (const uint*)((char*)src + (y    ) * stride_src);
        const uint *src_4 = (const uint*)((char*)src + (y + 1) * stride_src);
        const uint *src_5 = (const uint*)((char*)src + (y + 2) * stride_src);
        uint *dst_row = (uint*)((char*)dst + y*stride_dst) + 2;

        for (x = 2; x < width - 2; x++)
        {
            double value = src_1[0] * k11 + src_1[1] * k12 + src_1[2] * k13 + src_1[3] * k14 + src_1[4] * k15 +
                           src_2[0] * k21 + src_2[1] * k22 + src_2[2] * k23 + src_2[3] * k24 + src_2[4] * k25 +
                           src_3[0] * k31 + src_3[1] * k32 + src_3[2] * k33 + src_3[3] * k34 + src_3[4] * k35 +
                           src_4[0] * k41 + src_4[1] * k42 + src_4[2] * k43 + src_4[3] * k44 + src_4[4] * k45 +
                           src_5[0] * k51 + src_5[1] * k52 + src_5[2] * k53 + src_5[3] * k54 + src_5[4] * k55 ;
            *(dst_row++) = (uint)SATURATE(value, (double)0xffffffff);
            src_1++;
            src_2++;
            src_3++;
            src_4++;
            src_5++;
        }
    }
}

static void convolve_7x7_8b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel)
{
    const double k11 = kernel[ 0], k12 = kernel[ 1], k13 = kernel[ 2], k14 = kernel[ 3], k15 = kernel[ 4], k16 = kernel[ 5], k17 = kernel[ 6];
    const double k21 = kernel[ 7], k22 = kernel[ 8], k23 = kernel[ 9], k24 = kernel[10], k25 = kernel[11], k26 = kernel[12], k27 = kernel[13];
    const double k31 = kernel[14], k32 = kernel[15], k33 = kernel[16], k34 = kernel[17], k35 = kernel[18], k36 = kernel[19], k37 = kernel[20];
    const double k41 = kernel[21], k42 = kernel[22], k43 = kernel[23], k44 = kernel[24], k45 = kernel[25], k46 = kernel[26], k47 = kernel[27];
    const double k51 = kernel[28], k52 = kernel[27], k53 = kernel[28], k54 = kernel[29], k55 = kernel[30], k56 = kernel[31], k57 = kernel[32];
    const double k61 = kernel[35], k62 = kernel[36], k63 = kernel[37], k64 = kernel[38], k65 = kernel[39], k66 = kernel[40], k67 = kernel[41];
    const double k71 = kernel[42], k72 = kernel[43], k73 = kernel[44], k74 = kernel[45], k75 = kernel[46], k76 = kernel[47], k77 = kernel[48];
    int x, y;
    
    #if USE_OMP == 1
    #pragma omp parallel for private(x, y)
    #endif
    for (y = 3; y < height - 3; y++)
    {
        const uchar *src_1 = (const uchar*)((char*)src + (y - 3) * stride_src);
        const uchar *src_2 = (const uchar*)((char*)src + (y - 2) * stride_src);
        const uchar *src_3 = (const uchar*)((char*)src + (y - 1) * stride_src);
        const uchar *src_4 = (const uchar*)((char*)src + (y    ) * stride_src);
        const uchar *src_5 = (const uchar*)((char*)src + (y + 1) * stride_src);
        const uchar *src_6 = (const uchar*)((char*)src + (y + 2) * stride_src);
        const uchar *src_7 = (const uchar*)((char*)src + (y + 3) * stride_src);
        uchar *dst_row = (uchar*)((char*)dst + y*stride_dst) + 3;

        for (x = 3; x < width - 3; x++)
        {
            double value = src_1[0] * k11 + src_1[1] * k12 + src_1[2] * k13 + src_1[3] * k14 + src_1[4] * k15 + src_1[5] * k16 + src_1[6] * k17 +
                           src_2[0] * k21 + src_2[1] * k22 + src_2[2] * k23 + src_2[3] * k24 + src_2[4] * k25 + src_2[5] * k26 + src_2[6] * k27 +
                           src_3[0] * k31 + src_3[1] * k32 + src_3[2] * k33 + src_3[3] * k34 + src_3[4] * k35 + src_3[5] * k36 + src_3[6] * k37 +
                           src_4[0] * k41 + src_4[1] * k42 + src_4[2] * k43 + src_4[3] * k44 + src_4[4] * k45 + src_4[5] * k46 + src_4[6] * k47 +
                           src_5[0] * k51 + src_5[1] * k52 + src_5[2] * k53 + src_5[3] * k54 + src_5[4] * k55 + src_5[5] * k56 + src_5[6] * k57 +
                           src_6[0] * k61 + src_6[1] * k62 + src_6[2] * k63 + src_6[3] * k64 + src_6[4] * k65 + src_6[5] * k66 + src_6[6] * k67 +
                           src_7[0] * k71 + src_7[1] * k72 + src_7[2] * k73 + src_7[3] * k74 + src_7[4] * k75 + src_7[5] * k76 + src_7[6] * k77 ;
            *(dst_row++) = (uchar)SATURATE(value, (double)0xff);
            src_1++;
            src_2++;
            src_3++;
            src_4++;
            src_5++;
            src_6++;
            src_7++;
        }
    }
}

static void convolve_7x7_16b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel)
{
    const double k11 = kernel[ 0], k12 = kernel[ 1], k13 = kernel[ 2], k14 = kernel[ 3], k15 = kernel[ 4], k16 = kernel[ 5], k17 = kernel[ 6];
    const double k21 = kernel[ 7], k22 = kernel[ 8], k23 = kernel[ 9], k24 = kernel[10], k25 = kernel[11], k26 = kernel[12], k27 = kernel[13];
    const double k31 = kernel[14], k32 = kernel[15], k33 = kernel[16], k34 = kernel[17], k35 = kernel[18], k36 = kernel[19], k37 = kernel[20];
    const double k41 = kernel[21], k42 = kernel[22], k43 = kernel[23], k44 = kernel[24], k45 = kernel[25], k46 = kernel[26], k47 = kernel[27];
    const double k51 = kernel[28], k52 = kernel[27], k53 = kernel[28], k54 = kernel[29], k55 = kernel[30], k56 = kernel[31], k57 = kernel[32];
    const double k61 = kernel[35], k62 = kernel[36], k63 = kernel[37], k64 = kernel[38], k65 = kernel[39], k66 = kernel[40], k67 = kernel[41];
    const double k71 = kernel[42], k72 = kernel[43], k73 = kernel[44], k74 = kernel[45], k75 = kernel[46], k76 = kernel[47], k77 = kernel[48];
    int x, y;
    
    #if USE_OMP == 1
    #pragma omp parallel for private(x, y)
    #endif
    for (y = 3; y < height - 3; y++)
    {
        const ushort *src_1 = (const ushort*)((char*)src + (y - 3) * stride_src);
        const ushort *src_2 = (const ushort*)((char*)src + (y - 2) * stride_src);
        const ushort *src_3 = (const ushort*)((char*)src + (y - 1) * stride_src);
        const ushort *src_4 = (const ushort*)((char*)src + (y    ) * stride_src);
        const ushort *src_5 = (const ushort*)((char*)src + (y + 1) * stride_src);
        const ushort *src_6 = (const ushort*)((char*)src + (y + 2) * stride_src);
        const ushort *src_7 = (const ushort*)((char*)src + (y + 3) * stride_src);
        ushort *dst_row = (ushort*)((char*)dst + y*stride_dst) + 3;

        for (x = 3; x < width - 3; x++)
        {
            double value = src_1[0] * k11 + src_1[1] * k12 + src_1[2] * k13 + src_1[3] * k14 + src_1[4] * k15 + src_1[5] * k16 + src_1[6] * k17 +
                           src_2[0] * k21 + src_2[1] * k22 + src_2[2] * k23 + src_2[3] * k24 + src_2[4] * k25 + src_2[5] * k26 + src_2[6] * k27 +
                           src_3[0] * k31 + src_3[1] * k32 + src_3[2] * k33 + src_3[3] * k34 + src_3[4] * k35 + src_3[5] * k36 + src_3[6] * k37 +
                           src_4[0] * k41 + src_4[1] * k42 + src_4[2] * k43 + src_4[3] * k44 + src_4[4] * k45 + src_4[5] * k46 + src_4[6] * k47 +
                           src_5[0] * k51 + src_5[1] * k52 + src_5[2] * k53 + src_5[3] * k54 + src_5[4] * k55 + src_5[5] * k56 + src_5[6] * k57 +
                           src_6[0] * k61 + src_6[1] * k62 + src_6[2] * k63 + src_6[3] * k64 + src_6[4] * k65 + src_6[5] * k66 + src_6[6] * k67 +
                           src_7[0] * k71 + src_7[1] * k72 + src_7[2] * k73 + src_7[3] * k74 + src_7[4] * k75 + src_7[5] * k76 + src_7[6] * k77 ;
            *(dst_row++) = (ushort)SATURATE(value, (double)0xffff);
            src_1++;
            src_2++;
            src_3++;
            src_4++;
            src_5++;
            src_6++;
            src_7++;
        }
    }
}

static void convolve_7x7_32b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel)
{
    const double k11 = kernel[ 0], k12 = kernel[ 1], k13 = kernel[ 2], k14 = kernel[ 3], k15 = kernel[ 4], k16 = kernel[ 5], k17 = kernel[ 6];
    const double k21 = kernel[ 7], k22 = kernel[ 8], k23 = kernel[ 9], k24 = kernel[10], k25 = kernel[11], k26 = kernel[12], k27 = kernel[13];
    const double k31 = kernel[14], k32 = kernel[15], k33 = kernel[16], k34 = kernel[17], k35 = kernel[18], k36 = kernel[19], k37 = kernel[20];
    const double k41 = kernel[21], k42 = kernel[22], k43 = kernel[23], k44 = kernel[24], k45 = kernel[25], k46 = kernel[26], k47 = kernel[27];
    const double k51 = kernel[28], k52 = kernel[27], k53 = kernel[28], k54 = kernel[29], k55 = kernel[30], k56 = kernel[31], k57 = kernel[32];
    const double k61 = kernel[35], k62 = kernel[36], k63 = kernel[37], k64 = kernel[38], k65 = kernel[39], k66 = kernel[40], k67 = kernel[41];
    const double k71 = kernel[42], k72 = kernel[43], k73 = kernel[44], k74 = kernel[45], k75 = kernel[46], k76 = kernel[47], k77 = kernel[48];
    int x, y;
    
    #if USE_OMP == 1
    #pragma omp parallel for private(x, y)
    #endif
    for (y = 3; y < height - 3; y++)
    {
        const uint *src_1 = (const uint*)((char*)src + (y - 3) * stride_src);
        const uint *src_2 = (const uint*)((char*)src + (y - 2) * stride_src);
        const uint *src_3 = (const uint*)((char*)src + (y - 1) * stride_src);
        const uint *src_4 = (const uint*)((char*)src + (y    ) * stride_src);
        const uint *src_5 = (const uint*)((char*)src + (y + 1) * stride_src);
        const uint *src_6 = (const uint*)((char*)src + (y + 2) * stride_src);
        const uint *src_7 = (const uint*)((char*)src + (y + 3) * stride_src);
        uint *dst_row = (uint*)((char*)dst + y*stride_dst) + 3;

        for (x = 3; x < width - 3; x++)
        {
            double value = src_1[0] * k11 + src_1[1] * k12 + src_1[2] * k13 + src_1[3] * k14 + src_1[4] * k15 + src_1[5] * k16 + src_1[6] * k17 +
                           src_2[0] * k21 + src_2[1] * k22 + src_2[2] * k23 + src_2[3] * k24 + src_2[4] * k25 + src_2[5] * k26 + src_2[6] * k27 +
                           src_3[0] * k31 + src_3[1] * k32 + src_3[2] * k33 + src_3[3] * k34 + src_3[4] * k35 + src_3[5] * k36 + src_3[6] * k37 +
                           src_4[0] * k41 + src_4[1] * k42 + src_4[2] * k43 + src_4[3] * k44 + src_4[4] * k45 + src_4[5] * k46 + src_4[6] * k47 +
                           src_5[0] * k51 + src_5[1] * k52 + src_5[2] * k53 + src_5[3] * k54 + src_5[4] * k55 + src_5[5] * k56 + src_5[6] * k57 +
                           src_6[0] * k61 + src_6[1] * k62 + src_6[2] * k63 + src_6[3] * k64 + src_6[4] * k65 + src_6[5] * k66 + src_6[6] * k67 +
                           src_7[0] * k71 + src_7[1] * k72 + src_7[2] * k73 + src_7[3] * k74 + src_7[4] * k75 + src_7[5] * k76 + src_7[6] * k77 ;
            *(dst_row++) = (uint)SATURATE(value, (double)0xffffffff);
            src_1++;
            src_2++;
            src_3++;
            src_4++;
            src_5++;
            src_6++;
            src_7++;
        }
    }
}

static void convolve_mxn_8b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel, int kwidth, int kheight)
{
    int x, y, dx, dy;
    const int x0 = kwidth / 2;
    const int y0 = kheight / 2;
    
    #if USE_OMP == 1
    #pragma omp parallel for private(x, y, dx, dy)
    #endif
    for (y = y0; y < height - y0; y++)
    {
        for (x = x0; x < width - x0; x++)
        {
            uchar *dst_ptr = (uchar*)((char*)dst + y*stride_dst) + x;
            double value = 0.0;
            for (dy = -y0; dy <= +y0; dy++)
            {
                const uchar *src_ptr = (const uchar*)((char*)src + (y + dy)*stride_src) + x - x0;
                for (dx = -x0; dx <= +x0; dx++)
                    value += kernel[(y0 + dy) * kwidth + (x0 + dx)] *
                             (double)(*src_ptr++);
            }
            *dst_ptr = (uchar)SATURATE(value, (double)0xff);
        }
    }
}

static void convolve_mxn_16b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel, int kwidth, int kheight)
{
    int x, y, dx, dy;
    const int x0 = kwidth / 2;
    const int y0 = kheight / 2;
    
    #if USE_OMP == 1
    #pragma omp parallel for private(x, y, dx, dy)
    #endif
    for (y = y0; y < height - y0; y++)
    {
        for (x = x0; x < width - x0; x++)
        {
            ushort *dst_ptr = (ushort*)((char*)dst + y*stride_dst) + x;
            double value = 0.0;
            for (dy = -y0; dy <= +y0; dy++)
            {
                const ushort *src_ptr = (const ushort*)((char*)src + (y + dy)*stride_src) + x - x0;
                for (dx = -x0; dx <= +x0; dx++)
                    value += kernel[(y0 + dy) * kwidth + (x0 + dx)] *
                             (double)(*src_ptr++);
            }
            *dst_ptr = (ushort)SATURATE(value, (double)0xffff);
        }
    }
}

static void convolve_mxn_32b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, const double *kernel, int kwidth, int kheight)
{
    int x, y, dx, dy;
    const int x0 = kwidth / 2;
    const int y0 = kheight / 2;
    
    #if USE_OMP == 1
    #pragma omp parallel for private(x, y, dx, dy)
    #endif
    for (y = y0; y < height - y0; y++)
    {
        for (x = x0; x < width - x0; x++)
        {
            uint *dst_ptr = (uint*)((char*)dst + y*stride_dst) + x;
            double value = 0.0;
            for (dy = -y0; dy <= +y0; dy++)
            {
                const uint *src_ptr = (const uint*)((char*)src + (y + dy)*stride_src) + x - x0;
                for (dx = -x0; dx <= +x0; dx++)
                    value += kernel[(y0 + dy) * kwidth + (x0 + dx)] *
                             (double)(*src_ptr++);
            }
            *dst_ptr = (uint)SATURATE(value, (double)0xffffffff);
        }
    }
}
