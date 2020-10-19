#include "astrocv.h"

//#define NO_OPTIMIZATION

static void downsample_generic(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, int bpp, int factor);
static void downsample_2x2_8b (void *dst, int stride_dst, const void *src, int stride_src, int width, int height);
static void downsample_4x4_8b (void *dst, int stride_dst, const void *src, int stride_src, int width, int height);
static void downsample_2x2_16b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height);
static void downsample_4x4_16b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height);
static void downsample_2x2_32b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height);
static void downsample_4x4_32b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height);

static void upsample_generic(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, int bpp, int factor);
static void upsample_2x2_8b (void *dst, int stride_dst, const void *src, int stride_src, int width, int height);
static void upsample_4x4_8b (void *dst, int stride_dst, const void *src, int stride_src, int width, int height);
static void upsample_2x2_16b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height);
static void upsample_4x4_16b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height);
static void upsample_2x2_32b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height);
static void upsample_4x4_32b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height);


void downsample(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, int bpp, int factor)
{
    int y;
    int dst_height = height / factor;
    int dst_width = width / factor;
    for (y = 0; y < dst_height; y++)
    {
        char *dst_row = (char*)dst + stride_dst * y;
        switch (bpp)
        {
        case 8:
            memset(dst_row, 0, dst_width * 1);
            break;
        case 16:
            memset(dst_row, 0, dst_width * 2);
            break;
        case 32:
            memset(dst_row, 0, dst_width * 4);
            break;
        default:
            break;
        }
    }

#ifndef NO_OPTIMIZATION
    switch (factor)
    {
    case 2:
    {
        switch (bpp)
        {
        case 8:
            downsample_2x2_8b(dst, stride_dst, src, stride_src, width, height);
            return;
        case 16:
            downsample_2x2_16b(dst, stride_dst, src, stride_src, width, height);
            return;
        case 32:
            downsample_2x2_32b(dst, stride_dst, src, stride_src, width, height);
            return;
        default:
            break;
        }
    }
    case 4:
    {
        switch (bpp)
        {
        case 8:
            downsample_4x4_8b(dst, stride_dst, src, stride_src, width, height);
            return;
        case 16:
            downsample_4x4_16b(dst, stride_dst, src, stride_src, width, height);
            return;
        case 32:
            downsample_4x4_32b(dst, stride_dst, src, stride_src, width, height);
            return;
        default:
            break;
        }
    }
    default:
        break;
    }
#endif
    downsample_generic(dst, stride_dst, src, stride_src, width, height, bpp, factor);
}

void upsample(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, int bpp, int factor)
{
#ifndef NO_OPTIMIZATION
    switch (factor)
    {
    case 2:
    {
        switch (bpp)
        {
        case 8:
            upsample_2x2_8b(dst, stride_dst, src, stride_src, width, height);
            return;
        case 16:
            upsample_2x2_16b(dst, stride_dst, src, stride_src, width, height);
            return;
        case 32:
            upsample_2x2_32b(dst, stride_dst, src, stride_src, width, height);
            return;
        default:
            break;
        }
    }
    case 4:
    {
        switch (bpp)
        {
        case 8:
            upsample_4x4_8b(dst, stride_dst, src, stride_src, width, height);
            return;
        case 16:
            upsample_4x4_16b(dst, stride_dst, src, stride_src, width, height);
            return;
        case 32:
            upsample_4x4_32b(dst, stride_dst, src, stride_src, width, height);
            return;
        default:
            break;
        }
    }
    default:
        break;
    }
#endif
    upsample_generic(dst, stride_dst, src, stride_src, width, height, bpp, factor);
}

static void downsample_generic(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, int bpp, int factor)
{
    int y_dst;
    const int width_dst = width / factor;
    const int height_dst = height / factor;
    const float finv = (float)(1.0 / (factor * factor));
    
    switch (bpp)
    {
    case 8:
    {
        #if USE_OMP == 1
        #pragma omp parallel for private(y_dst)
        #endif
        for (y_dst = 0; y_dst < height_dst; y_dst++)
        {
            int x_src, y_src, x_dst;
            for (x_dst = 0; x_dst < width_dst; x_dst++)
            {
                float val = 0.0;
                uchar *dst_ptr = (uchar*)((char*)dst + y_dst * stride_dst) + x_dst;
                for (y_src = 0; y_src  < factor; y_src++)
                {
                    const uchar *src_ptr = (const uchar*)((char*)src + (y_dst * factor + y_src) * stride_src) + x_dst * factor;
                    for (x_src = 0; x_src < factor; x_src++)
                        val += (float)(*src_ptr++);
                }
                val *= finv;
                *dst_ptr = (uchar)val;
            }
        }
        break;
    }
    case 16:
    {
        #if USE_OMP == 1
        #pragma omp parallel for private(y_dst)
        #endif
        for (y_dst = 0; y_dst < height_dst; y_dst++)
        {
            int x_src, y_src, x_dst;
            for (x_dst = 0; x_dst < width_dst; x_dst++)
            {
                float val = 0.0;
                ushort *dst_ptr = (ushort*)((char*)dst + y_dst * stride_dst) + x_dst;
                for (y_src = 0; y_src  < factor; y_src++)
                {
                    const ushort *src_ptr = (const ushort*)((char*)src + (y_dst * factor + y_src) * stride_src) + x_dst * factor;
                    for (x_src = 0; x_src < factor; x_src++)
                        val += (float)(*src_ptr++);
                }
                val *= finv;
                *dst_ptr = (ushort)val;
            }
        }
        break;
    }
    case 32:
    {
        #if USE_OMP == 1
        #pragma omp parallel for private(y_dst)
        #endif
        for (y_dst = 0; y_dst < height_dst; y_dst++)
        {
            int x_src, y_src, x_dst;
            for (x_dst = 0; x_dst < width_dst; x_dst++)
            {
                float val = 0.0;
                uint *dst_ptr = (uint*)((char*)dst + y_dst * stride_dst) + x_dst;
                for (y_src = 0; y_src  < factor; y_src++)
                {
                    const uint *src_ptr = (const uint*)((char*)src + (y_dst * factor + y_src) * stride_src) + x_dst * factor;
                    for (x_src = 0; x_src < factor; x_src++)
                        val += (float)(*src_ptr++);
                }
                val *= finv;
                *dst_ptr = (uint)val;
            }
        }
        break;
    }
    }
}

static void upsample_generic(void *dst, int stride_dst, const void *src, int stride_src, int width, int height, int bpp, int factor)
{
    int y_src;
    const float finv = (float)(1.0 / factor);
    const int fo2 = factor/2;
    const float dx = 0.5, dy = 0.5;
    const int ox = fo2, oy = fo2;
    
    switch (bpp)
    {
    case 8:
    {
        #if USE_OMP == 1
        #pragma omp parallel for private(y_src)
        #endif
        for (y_src = 0; y_src < height-1; y_src++)
        {
            int x_src, x_dst, y_dst;
            int start_x, start_y, end_x, end_y;
            const uchar *src_ptr_top, *src_ptr_bottom;
            
            src_ptr_top = (const uchar*)((char*)src + y_src * stride_src);
            src_ptr_bottom = (const uchar*)((char*)src + (y_src + 1) * stride_src);
            
            if (y_src == 0)
                start_y = -fo2;
            else
                start_y = 0;
            if (y_src == height - 2)
                end_y = factor + fo2;
            else
                end_y = factor;
            
            for (x_src = 0; x_src < width-1; x_src++)
            {
                float lt, rt, lb, rb;
                float a, b, c, d;
                float v0, addx, addy;
                
                if (x_src == 0)
                    start_x = -fo2;
                else
                    start_x = 0;
                if (x_src == width - 2)
                    end_x = factor + fo2;
                else
                    end_x = factor;
                
                lt = src_ptr_top[0];
                rt = src_ptr_top[1];
                lb = src_ptr_bottom[0];
                rb = src_ptr_bottom[1];
                
                a = lt;
                b = (rt - lt) * finv;
                c = (lb - lt) * finv;
                d = ((rb - rt) - (lb - lt)) * finv * finv;
                
                v0 = a + b*(dx + start_x) + c*(dy + start_y) + d*(dx + start_x)*(dy + start_y);
                addy = c + d*(dx + start_x);
                
                for (y_dst = start_y; y_dst < end_y; y_dst++)
                {
                    uchar *dst_ptr = (uchar*)((char*)dst + (y_src * factor + y_dst + oy) * stride_dst) + x_src * factor + ox + start_x;
                    float v1 = v0;
                    addx = b + d*(dy + (float)y_dst);
                    for (x_dst = start_x; x_dst < end_x; x_dst++)
                    {
                        *(dst_ptr++) = (uchar)SATURATE(v1, (float)0xff);
                        v1 += addx;
                    }
                    v0 += addy;
                }

                src_ptr_top++;
                src_ptr_bottom++;
            }
        }
        break;
    }
    case 16:
    {
        #if USE_OMP == 1
        #pragma omp parallel for private(y_src)
        #endif
        for (y_src = 0; y_src < height-1; y_src++)
        {
            int x_src, x_dst, y_dst;
            int start_x, start_y, end_x, end_y;
            const ushort *src_ptr_top, *src_ptr_bottom;
            
            src_ptr_top = (const ushort*)((char*)src + y_src * stride_src);
            src_ptr_bottom = (const ushort*)((char*)src + (y_src + 1) * stride_src);
            
            if (y_src == 0)
                start_y = -fo2;
            else
                start_y = 0;
            if (y_src == height - 2)
                end_y = factor + fo2;
            else
                end_y = factor;
            
            for (x_src = 0; x_src < width-1; x_src++)
            {
                float lt, rt, lb, rb;
                float a, b, c, d;
                float v0, addx, addy;
                
                if (x_src == 0)
                    start_x = -fo2;
                else
                    start_x = 0;
                if (x_src == width - 2)
                    end_x = factor + fo2;
                else
                    end_x = factor;
                
                lt = src_ptr_top[0];
                rt = src_ptr_top[1];
                lb = src_ptr_bottom[0];
                rb = src_ptr_bottom[1];
                
                a = lt;
                b = (rt - lt) * finv;
                c = (lb - lt) * finv;
                d = ((rb - rt) - (lb - lt)) * finv * finv;
                
                v0 = a + b*(dx + start_x) + c*(dy + start_y) + d*(dx + start_x)*(dy + start_y);
                addy = c + d*(dx + start_x);
                
                for (y_dst = start_y; y_dst < end_y; y_dst++)
                {
                    ushort *dst_ptr = (ushort*)((char*)dst + (y_src * factor + y_dst + oy) * stride_dst) + x_src * factor + ox + start_x;
                    float v1 = v0;
                    addx = b + d*(dy + (float)y_dst);
                    for (x_dst = start_x; x_dst < end_x; x_dst++)
                    {
                        *(dst_ptr++) = (ushort)SATURATE(v1, (float)0xffff);
                        v1 += addx;
                    }
                    v0 += addy;
                }

                src_ptr_top++;
                src_ptr_bottom++;
            }
        }
        break;
    }
    case 32:
    {
        #if USE_OMP == 1
        #pragma omp parallel for private(y_src)
        #endif
        for (y_src = 0; y_src < height-1; y_src++)
        {
            int x_src, x_dst, y_dst;
            int start_x, start_y, end_x, end_y;
            const uint *src_ptr_top, *src_ptr_bottom;
            
            src_ptr_top = (const uint*)((char*)src + y_src * stride_src);
            src_ptr_bottom = (const uint*)((char*)src + (y_src + 1) * stride_src);
            
            if (y_src == 0)
                start_y = -fo2;
            else
                start_y = 0;
            if (y_src == height - 2)
                end_y = factor + fo2;
            else
                end_y = factor;
            
            for (x_src = 0; x_src < width-1; x_src++)
            {
                float lt, rt, lb, rb;
                float a, b, c, d;
                float v0, addx, addy;
                
                if (x_src == 0)
                    start_x = -fo2;
                else
                    start_x = 0;
                if (x_src == width - 2)
                    end_x = factor + fo2;
                else
                    end_x = factor;
                
                lt = (float)src_ptr_top[0];
                rt = (float)src_ptr_top[1];
                lb = (float)src_ptr_bottom[0];
                rb = (float)src_ptr_bottom[1];
                
                a = lt;
                b = (rt - lt) * finv;
                c = (lb - lt) * finv;
                d = ((rb - rt) - (lb - lt)) * finv * finv;
                
                v0 = a + b*(dx + start_x) + c*(dy + start_y) + d*(dx + start_x)*(dy + start_y);
                addy = c + d*(dx + start_x);
                
                for (y_dst = start_y; y_dst < end_y; y_dst++)
                {
                    uint *dst_ptr = (uint*)((char*)dst + (y_src * factor + y_dst + oy) * stride_dst) + x_src * factor + ox + start_x;
                    float v1 = v0;
                    addx = b + d*(dy + (float)y_dst);
                    for (x_dst = start_x; x_dst < end_x; x_dst++)
                    {
                        *(dst_ptr++) = (uint)SATURATE(v1, (float)0xffffffff);
                        v1 += addx;
                    }
                    v0 += addy;
                }

                src_ptr_top++;
                src_ptr_bottom++;
            }
        }
        break;
    }
    }
}

static void downsample_2x2_8b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height)
{
    int x_dst, y_dst;
    const int width_dst = width / 2;
    const int height_dst = height / 2;

    #if USE_OMP == 1
    #pragma omp parallel for private(y_dst, x_dst)
    #endif
    for (y_dst = 0; y_dst < height_dst; y_dst++)
    {
        const uchar *src_ptr_1 = (const uchar*)((char*)src + (y_dst * 2 + 0) * stride_src);
        const uchar *src_ptr_2 = (const uchar*)((char*)src + (y_dst * 2 + 1) * stride_src);
        uchar *dst_ptr = (uchar*)((char*)dst + y_dst * stride_dst);
        for (x_dst = 0; x_dst < width_dst; x_dst++)
        {
            int sum = *(src_ptr_1++);
            sum += *(src_ptr_1++);
            sum += *(src_ptr_2++);
            sum += *(src_ptr_2++);
            *(dst_ptr++) = (sum >> 2);
        }
    }
}

static void downsample_2x2_16b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height)
{
    int x_dst, y_dst;
    const int width_dst = width / 2;
    const int height_dst = height / 2;

    #if USE_OMP == 1
    #pragma omp parallel for private(y_dst, x_dst)
    #endif
    for (y_dst = 0; y_dst < height_dst; y_dst++)
    {
        const ushort *src_ptr_1 = (const ushort*)((char*)src + (y_dst * 2 + 0) * stride_src);
        const ushort *src_ptr_2 = (const ushort*)((char*)src + (y_dst * 2 + 1) * stride_src);
        ushort *dst_ptr = (ushort*)((char*)dst + y_dst * stride_dst);
        for (x_dst = 0; x_dst < width_dst; x_dst++)
        {
            int sum = *(src_ptr_1++);
            sum += *(src_ptr_1++);
            sum += *(src_ptr_2++);
            sum += *(src_ptr_2++);
            *(dst_ptr++) = (sum >> 2);
        }
    }
}

static void downsample_2x2_32b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height)
{
    int x_dst, y_dst;
    const int width_dst = width / 2;
    const int height_dst = height / 2;

    #if USE_OMP == 1
    #pragma omp parallel for private(y_dst, x_dst)
    #endif
    for (y_dst = 0; y_dst < height_dst; y_dst++)
    {
        const uint *src_ptr_1 = (const uint*)((char*)src + (y_dst * 2 + 0) * stride_src);
        const uint *src_ptr_2 = (const uint*)((char*)src + (y_dst * 2 + 1) * stride_src);
        uint *dst_ptr = (uint*)((char*)dst + y_dst * stride_dst);
        for (x_dst = 0; x_dst < width_dst; x_dst++)
        {
            int sum = *(src_ptr_1++);
            sum += *(src_ptr_1++);
            sum += *(src_ptr_2++);
            sum += *(src_ptr_2++);
            *(dst_ptr++) = (sum >> 2);
        }
    }
}

static void downsample_4x4_8b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height)
{
    int y_dst, x_dst;
    const int width_dst = width / 4;
    const int height_dst = height / 4;
    
#if USE_SSE == 1
    int can_use_sse = (!((int)src & 0xf) && !(stride_src & 0xf) &&
                       !(width_dst & 0x3)) ? 1 : 0;
    if (can_use_sse)
    {
        __m128i dstvals, srcvals;
        const __m128i lobytes = _mm_set1_epi32(0x000000FF);
        #if USE_OMP == 1
        #pragma omp parallel for private(y_dst, x_dst, dstvals, srcvals)
        #endif
        for (y_dst = 0; y_dst < height_dst; y_dst++)
        {
            uchar *dst_ptr = (uchar*)((char*)dst + y_dst * stride_dst);
            const uchar *line1 = (const uchar*)((char*)src + (y_dst * 4) * stride_src);
            const uchar *line2 = line1 + stride_src * 1,
                        *line3 = line1 + stride_src * 2,
                        *line4 = line1 + stride_src * 3;
            for (x_dst = 0; x_dst < width_dst; x_dst += 4)
            {
                dstvals = _mm_setzero_si128();

                srcvals = _mm_loadu_si128((__m128i*)line1); line1 += 16;
                dstvals = _mm_add_epi32(dstvals, _mm_and_si128(_mm_srli_epi32(srcvals, 24), lobytes)); // 15 11  7  3
                dstvals = _mm_add_epi32(dstvals, _mm_and_si128(_mm_srli_epi32(srcvals, 16), lobytes)); // 14 10  6  2
                dstvals = _mm_add_epi32(dstvals, _mm_and_si128(_mm_srli_epi32(srcvals,  8), lobytes)); // 13  9  5  1
                dstvals = _mm_add_epi32(dstvals, _mm_and_si128(srcvals, lobytes));                     // 12  8  4  0

                srcvals = _mm_loadu_si128((__m128i*)line2); line2 += 16;
                dstvals = _mm_add_epi32(dstvals, _mm_and_si128(_mm_srli_epi32(srcvals, 24), lobytes)); // 15 11  7  3
                dstvals = _mm_add_epi32(dstvals, _mm_and_si128(_mm_srli_epi32(srcvals, 16), lobytes)); // 14 10  6  2
                dstvals = _mm_add_epi32(dstvals, _mm_and_si128(_mm_srli_epi32(srcvals,  8), lobytes)); // 13  9  5  1
                dstvals = _mm_add_epi32(dstvals, _mm_and_si128(srcvals, lobytes));                     // 12  8  4  0

                srcvals = _mm_loadu_si128((__m128i*)line3); line3 += 16;
                dstvals = _mm_add_epi32(dstvals, _mm_and_si128(_mm_srli_epi32(srcvals, 24), lobytes)); // 15 11  7  3
                dstvals = _mm_add_epi32(dstvals, _mm_and_si128(_mm_srli_epi32(srcvals, 16), lobytes)); // 14 10  6  2
                dstvals = _mm_add_epi32(dstvals, _mm_and_si128(_mm_srli_epi32(srcvals,  8), lobytes)); // 13  9  5  1
                dstvals = _mm_add_epi32(dstvals, _mm_and_si128(srcvals, lobytes));                     // 12  8  4  0

                srcvals = _mm_loadu_si128((__m128i*)line4); line4 += 16;
                dstvals = _mm_add_epi32(dstvals, _mm_and_si128(_mm_srli_epi32(srcvals, 24), lobytes)); // 15 11  7  3
                dstvals = _mm_add_epi32(dstvals, _mm_and_si128(_mm_srli_epi32(srcvals, 16), lobytes)); // 14 10  6  2
                dstvals = _mm_add_epi32(dstvals, _mm_and_si128(_mm_srli_epi32(srcvals,  8), lobytes)); // 13  9  5  1
                dstvals = _mm_add_epi32(dstvals, _mm_and_si128(srcvals, lobytes));                     // 12  8  4  0
                
                dstvals = _mm_srli_epi32(dstvals, 4);
                
                {
                    uint *u32 = (uint*)&dstvals;
                    *(dst_ptr++) = (uchar)u32[0];
                    *(dst_ptr++) = (uchar)u32[1];
                    *(dst_ptr++) = (uchar)u32[2];
                    *(dst_ptr++) = (uchar)u32[3];
                }
            }
        }
    }
    else
#endif
    {
        #if USE_OMP == 1
        #pragma omp parallel for private(y_dst, x_dst)
        #endif
        for (y_dst = 0; y_dst < height_dst; y_dst++)
        {
            const uchar *src_ptr_1 = (const uchar*)((char*)src + (y_dst * 4 + 0) * stride_src);
            const uchar *src_ptr_2 = (const uchar*)((char*)src + (y_dst * 4 + 1) * stride_src);
            const uchar *src_ptr_3 = (const uchar*)((char*)src + (y_dst * 4 + 2) * stride_src);
            const uchar *src_ptr_4 = (const uchar*)((char*)src + (y_dst * 4 + 3) * stride_src);
            uchar *dst_ptr = (uchar*)((char*)dst + y_dst * stride_dst);
            for (x_dst = 0; x_dst < width_dst; x_dst++)
            {
                int sum = *(src_ptr_1++);
                sum += *(src_ptr_1++);
                sum += *(src_ptr_1++);
                sum += *(src_ptr_1++);
                sum += *(src_ptr_2++);
                sum += *(src_ptr_2++);
                sum += *(src_ptr_2++);
                sum += *(src_ptr_2++);
                sum += *(src_ptr_3++);
                sum += *(src_ptr_3++);
                sum += *(src_ptr_3++);
                sum += *(src_ptr_3++);
                sum += *(src_ptr_4++);
                sum += *(src_ptr_4++);
                sum += *(src_ptr_4++);
                sum += *(src_ptr_4++);
                *(dst_ptr++) = (sum >> 4);
            }
        }
    }
}

static void downsample_4x4_16b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height)
{
    int x_dst, y_dst;
    const int width_dst = width / 4;
    const int height_dst = height / 4;
    
#if USE_SSE == 1
    int can_use_sse = (!((int)src & 0xf) && !(stride_src & 0xf) &&
                       !(width_dst & 0x3)) ? 1 : 0;
    if (can_use_sse)
    {
        __m128i srcvals, lovals, hivals;
        const __m128i lowords = _mm_set_epi32(0x00000000, 0x0000FFFF, 0x00000000, 0x0000FFFF);
        #if USE_OMP == 1
        #pragma omp parallel for private(y_dst, x_dst, lovals, hivals, srcvals)
        #endif
        for (y_dst = 0; y_dst < height_dst; y_dst++)
        {
            ushort *dst_ptr = (ushort*)((char*)dst + y_dst * stride_dst);
            const ushort *line1 = (const ushort*)((char*)src + (y_dst * 4 + 0) * stride_src);
            const ushort *line2 = (const ushort*)((char*)src + (y_dst * 4 + 1) * stride_src);
            const ushort *line3 = (const ushort*)((char*)src + (y_dst * 4 + 2) * stride_src);
            const ushort *line4 = (const ushort*)((char*)src + (y_dst * 4 + 3) * stride_src);
            for (x_dst = 0; x_dst < width_dst; x_dst += 4)
            {
                lovals = hivals = _mm_setzero_si128();

                srcvals = _mm_loadu_si128((__m128i*)line1); line1 += 8;
                lovals = _mm_add_epi64(lovals, _mm_and_si128(_mm_srli_epi64(srcvals, 48), lowords)); // 7  3
                lovals = _mm_add_epi64(lovals, _mm_and_si128(_mm_srli_epi64(srcvals, 32), lowords)); // 6  2
                lovals = _mm_add_epi64(lovals, _mm_and_si128(_mm_srli_epi64(srcvals, 16), lowords)); // 5  1
                lovals = _mm_add_epi64(lovals, _mm_and_si128(srcvals, lowords));                     // 4  0
                srcvals = _mm_loadu_si128((__m128i*)line1); line1 += 8;
                hivals = _mm_add_epi64(hivals, _mm_and_si128(_mm_srli_epi64(srcvals, 48), lowords)); // 7  3
                hivals = _mm_add_epi64(hivals, _mm_and_si128(_mm_srli_epi64(srcvals, 32), lowords)); // 6  2
                hivals = _mm_add_epi64(hivals, _mm_and_si128(_mm_srli_epi64(srcvals, 16), lowords)); // 5  1
                hivals = _mm_add_epi64(hivals, _mm_and_si128(srcvals, lowords));                     // 4  0

                srcvals = _mm_loadu_si128((__m128i*)line2); line2 += 8;
                lovals = _mm_add_epi64(lovals, _mm_and_si128(_mm_srli_epi64(srcvals, 48), lowords)); // 7  3
                lovals = _mm_add_epi64(lovals, _mm_and_si128(_mm_srli_epi64(srcvals, 32), lowords)); // 6  2
                lovals = _mm_add_epi64(lovals, _mm_and_si128(_mm_srli_epi64(srcvals, 16), lowords)); // 5  1
                lovals = _mm_add_epi64(lovals, _mm_and_si128(srcvals, lowords));                     // 4  0
                srcvals = _mm_loadu_si128((__m128i*)line2); line2 += 8;
                hivals = _mm_add_epi64(hivals, _mm_and_si128(_mm_srli_epi64(srcvals, 48), lowords)); // 7  3
                hivals = _mm_add_epi64(hivals, _mm_and_si128(_mm_srli_epi64(srcvals, 32), lowords)); // 6  2
                hivals = _mm_add_epi64(hivals, _mm_and_si128(_mm_srli_epi64(srcvals, 16), lowords)); // 5  1
                hivals = _mm_add_epi64(hivals, _mm_and_si128(srcvals, lowords));                     // 4  0

                srcvals = _mm_loadu_si128((__m128i*)line3); line3 += 8;
                lovals = _mm_add_epi64(lovals, _mm_and_si128(_mm_srli_epi64(srcvals, 48), lowords)); // 7  3
                lovals = _mm_add_epi64(lovals, _mm_and_si128(_mm_srli_epi64(srcvals, 32), lowords)); // 6  2
                lovals = _mm_add_epi64(lovals, _mm_and_si128(_mm_srli_epi64(srcvals, 16), lowords)); // 5  1
                lovals = _mm_add_epi64(lovals, _mm_and_si128(srcvals, lowords));                     // 4  0
                srcvals = _mm_loadu_si128((__m128i*)line3); line3 += 8;
                hivals = _mm_add_epi64(hivals, _mm_and_si128(_mm_srli_epi64(srcvals, 48), lowords)); // 7  3
                hivals = _mm_add_epi64(hivals, _mm_and_si128(_mm_srli_epi64(srcvals, 32), lowords)); // 6  2
                hivals = _mm_add_epi64(hivals, _mm_and_si128(_mm_srli_epi64(srcvals, 16), lowords)); // 5  1
                hivals = _mm_add_epi64(hivals, _mm_and_si128(srcvals, lowords));                     // 4  0

                srcvals = _mm_loadu_si128((__m128i*)line4); line4 += 8;
                lovals = _mm_add_epi64(lovals, _mm_and_si128(_mm_srli_epi64(srcvals, 48), lowords)); // 7  3
                lovals = _mm_add_epi64(lovals, _mm_and_si128(_mm_srli_epi64(srcvals, 32), lowords)); // 6  2
                lovals = _mm_add_epi64(lovals, _mm_and_si128(_mm_srli_epi64(srcvals, 16), lowords)); // 5  1
                lovals = _mm_add_epi64(lovals, _mm_and_si128(srcvals, lowords));                     // 4  0
                srcvals = _mm_loadu_si128((__m128i*)line4); line4 += 8;
                hivals = _mm_add_epi64(hivals, _mm_and_si128(_mm_srli_epi64(srcvals, 48), lowords)); // 7  3
                hivals = _mm_add_epi64(hivals, _mm_and_si128(_mm_srli_epi64(srcvals, 32), lowords)); // 6  2
                hivals = _mm_add_epi64(hivals, _mm_and_si128(_mm_srli_epi64(srcvals, 16), lowords)); // 5  1
                hivals = _mm_add_epi64(hivals, _mm_and_si128(srcvals, lowords));                     // 4  0
                
                hivals = _mm_srli_epi32(hivals, 4);
                lovals = _mm_srli_epi32(lovals, 4);
                
                {
                    uint *u32_hi = (uint*)&hivals;
                    uint *u32_lo = (uint*)&lovals;
                    *(dst_ptr++) = (ushort)u32_lo[0];
                    *(dst_ptr++) = (ushort)u32_lo[2];
                    *(dst_ptr++) = (ushort)u32_hi[0];
                    *(dst_ptr++) = (ushort)u32_hi[2];
                }
            }
        }
    }
    else
#endif
    {
        #if USE_OMP == 1
        #pragma omp parallel for private(y_dst, x_dst)
        #endif
        for (y_dst = 0; y_dst < height_dst; y_dst++)
        {
            const ushort *src_ptr_1 = (const ushort*)((char*)src + (y_dst * 4 + 0) * stride_src);
            const ushort *src_ptr_2 = (const ushort*)((char*)src + (y_dst * 4 + 1) * stride_src);
            const ushort *src_ptr_3 = (const ushort*)((char*)src + (y_dst * 4 + 2) * stride_src);
            const ushort *src_ptr_4 = (const ushort*)((char*)src + (y_dst * 4 + 3) * stride_src);
            ushort *dst_ptr = (ushort*)((char*)dst + y_dst * stride_dst);
            for (x_dst = 0; x_dst < width_dst; x_dst++)
            {
                int sum = *(src_ptr_1++);
                sum += *(src_ptr_1++);
                sum += *(src_ptr_1++);
                sum += *(src_ptr_1++);
                sum += *(src_ptr_2++);
                sum += *(src_ptr_2++);
                sum += *(src_ptr_2++);
                sum += *(src_ptr_2++);
                sum += *(src_ptr_3++);
                sum += *(src_ptr_3++);
                sum += *(src_ptr_3++);
                sum += *(src_ptr_3++);
                sum += *(src_ptr_4++);
                sum += *(src_ptr_4++);
                sum += *(src_ptr_4++);
                sum += *(src_ptr_4++);
                *(dst_ptr++) = (sum >> 4);
            }
        }
    }
}

static void downsample_4x4_32b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height)
{
    int x_dst, y_dst;
    const int width_dst = width / 4;
    const int height_dst = height / 4;
    
    #if USE_OMP == 1
    #pragma omp parallel for private(y_dst, x_dst)
    #endif
    for (y_dst = 0; y_dst < height_dst; y_dst++)
    {
        const uint *src_ptr_1 = (const uint*)((char*)src + (y_dst * 4 + 0) * stride_src);
        const uint *src_ptr_2 = (const uint*)((char*)src + (y_dst * 4 + 1) * stride_src);
        const uint *src_ptr_3 = (const uint*)((char*)src + (y_dst * 4 + 2) * stride_src);
        const uint *src_ptr_4 = (const uint*)((char*)src + (y_dst * 4 + 3) * stride_src);
        uint *dst_ptr = (uint*)((char*)dst + y_dst * stride_dst);
        for (x_dst = 0; x_dst < width_dst; x_dst++)
        {
            int sum = *(src_ptr_1++);
            sum += *(src_ptr_1++);
            sum += *(src_ptr_1++);
            sum += *(src_ptr_1++);
            sum += *(src_ptr_2++);
            sum += *(src_ptr_2++);
            sum += *(src_ptr_2++);
            sum += *(src_ptr_2++);
            sum += *(src_ptr_3++);
            sum += *(src_ptr_3++);
            sum += *(src_ptr_3++);
            sum += *(src_ptr_3++);
            sum += *(src_ptr_4++);
            sum += *(src_ptr_4++);
            sum += *(src_ptr_4++);
            sum += *(src_ptr_4++);
            *(dst_ptr++) = (sum >> 4);
        }
    }
}

static void upsample_2x2_8b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height)
{
    int x_src, y_src;

    const float finv = 0.5;
    const float dx = 0.5, dy = 0.5;

    #if USE_OMP == 1
    #pragma omp parallel for private(x_src, y_src)
    #endif
    for (y_src = 0; y_src < height-1; y_src++)
    {
        const uchar *src_ptr_top, *src_ptr_bottom;
        int start_x, start_y;

        src_ptr_top = (const uchar*)((char*)src + y_src * stride_src);
        src_ptr_bottom = (const uchar*)((char*)src + (y_src + 1) * stride_src);
        
        if (y_src == 0)
            start_y = -1;
        else
            start_y = 0;

        for (x_src = 0; x_src < width-1; x_src++)
        {
            float lt = src_ptr_top[0];
            float rt = src_ptr_top[1];
            float lb = src_ptr_bottom[0];
            float rb = src_ptr_bottom[1];

            float a = lt;
            float b = (rt - lt) * finv;
            float c = (lb - lt) * finv;
            float d = ((rb - rt) - (lb - lt)) * finv * finv;
            
            uchar *dst_ptr;
            float v0, v1, addy, addx;
            
            if (x_src == 0)
                start_x = -1;
            else
                start_x = 0;
                
            v0 = a + b*(dx + start_x) + c*(dy + start_y) + d*(dx + start_x)*(dy + start_y);
            addy = c + d*(dx + start_x);
            addx = b + d*(dy + start_y);
            
            dst_ptr = (uchar*)((char*)dst + (y_src * 2 + start_y + 1) * stride_dst) + x_src * 2 + 1 + start_x;
            
            v1 = v0;
            dst_ptr[0] = (uchar)SATURATE(v1, (float)0xff);
            v1 += addx;
            dst_ptr[1] = (uchar)SATURATE(v1, (float)0xff);
            if (x_src == 0 || x_src == width - 2)
            {
                v1 += addx;
                dst_ptr[2] = (uchar)SATURATE(v1, (float)0xff);
            }

            dst_ptr = (uchar*)((char*)dst_ptr + stride_dst);
            v0 += addy;
            addx = (float)(b + d*(dy + start_y + 1.0));

            v1 = v0;
            dst_ptr[0] = (uchar)SATURATE(v1, (float)0xff);
            v1 += addx;
            dst_ptr[1] = (uchar)SATURATE(v1, (float)0xff);
            if (x_src == 0 || x_src == width - 2)
            {
                v1 += addx;
                dst_ptr[2] = (uchar)SATURATE(v1, (float)0xff);
            }
                
            if (y_src == 0 || y_src == height - 2)
            {
                dst_ptr = (uchar*)((char*)dst_ptr + stride_dst);
                v0 += addy;
                addx = (float)(b + d*(dy + start_y + 2.0));
                
                v1 = v0;
                dst_ptr[0] = (uchar)SATURATE(v1, (float)0xff);
                v1 += addx;
                dst_ptr[1] = (uchar)SATURATE(v1, (float)0xff);
                if (x_src == 0 || x_src == width - 2)
                {
                    v1 += addx;
                    dst_ptr[2] = (uchar)SATURATE(v1, (float)0xff);
                }
            }

            src_ptr_top++;
            src_ptr_bottom++;
        }
    }
}

static void upsample_2x2_16b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height)
{
    int x_src, y_src;

    const float finv = 0.5;
    const float dx = 0.5, dy = 0.5;

    #if USE_OMP == 1
    #pragma omp parallel for private(x_src, y_src)
    #endif
    for (y_src = 0; y_src < height-1; y_src++)
    {
        const ushort *src_ptr_top, *src_ptr_bottom;
        int start_x, start_y;

        src_ptr_top = (const ushort*)((char*)src + y_src * stride_src);
        src_ptr_bottom = (const ushort*)((char*)src + (y_src + 1) * stride_src);
        
        if (y_src == 0)
            start_y = -1;
        else
            start_y = 0;

        for (x_src = 0; x_src < width-1; x_src++)
        {
            float lt = src_ptr_top[0];
            float rt = src_ptr_top[1];
            float lb = src_ptr_bottom[0];
            float rb = src_ptr_bottom[1];

            float a = lt;
            float b = (rt - lt) * finv;
            float c = (lb - lt) * finv;
            float d = ((rb - rt) - (lb - lt)) * finv * finv;
            
            ushort *dst_ptr;
            float v0, v1, addy, addx;
            
            if (x_src == 0)
                start_x = -1;
            else
                start_x = 0;
                
            v0 = a + b*(dx + start_x) + c*(dy + start_y) + d*(dx + start_x)*(dy + start_y);
            addy = c + d*(dx + start_x);
            addx = b + d*(dy + start_y);
            
            dst_ptr = (ushort*)((char*)dst + (y_src * 2 + start_y + 1) * stride_dst) + x_src * 2 + 1 + start_x;
            
            v1 = v0;
            dst_ptr[0] = (ushort)SATURATE(v1, (float)0xffff);
            v1 += addx;
            dst_ptr[1] = (ushort)SATURATE(v1, (float)0xffff);
            if (x_src == 0 || x_src == width - 2)
            {
                v1 += addx;
                dst_ptr[2] = (ushort)SATURATE(v1, (float)0xffff);
            }

            dst_ptr = (ushort*)((char*)dst_ptr + stride_dst);
            v0 += addy;
            addx = (float)(b + d*(dy + start_y + 1.0));

            v1 = v0;
            dst_ptr[0] = (ushort)SATURATE(v1, (float)0xffff);
            v1 += addx;
            dst_ptr[1] = (ushort)SATURATE(v1, (float)0xffff);
            if (x_src == 0 || x_src == width - 2)
            {
                v1 += addx;
                dst_ptr[2] = (ushort)SATURATE(v1, (float)0xffff);
            }
                
            if (y_src == 0 || y_src == height - 2)
            {
                dst_ptr = (ushort*)((char*)dst_ptr + stride_dst);
                v0 += addy;
                addx = (float)(b + d*(dy + start_y + 2.0));
                
                v1 = v0;
                dst_ptr[0] = (ushort)SATURATE(v1, (float)0xffff);
                v1 += addx;
                dst_ptr[1] = (ushort)SATURATE(v1, (float)0xffff);
                if (x_src == 0 || x_src == width - 2)
                {
                    v1 += addx;
                    dst_ptr[2] = (ushort)SATURATE(v1, (float)0xffff);
                }
            }

            src_ptr_top++;
            src_ptr_bottom++;
        }
    }
}

static void upsample_2x2_32b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height)
{
    int x_src, y_src;

    const float finv = 0.5;
    const float dx = 0.5, dy = 0.5;
    
    #if USE_OMP == 1
    #pragma omp parallel for private(x_src, y_src)
    #endif
    for (y_src = 0; y_src < height-1; y_src++)
    {
        const uint *src_ptr_top, *src_ptr_bottom;
        int start_x, start_y;

        src_ptr_top = (const uint*)((char*)src + y_src * stride_src);
        src_ptr_bottom = (const uint*)((char*)src + (y_src + 1) * stride_src);
        
        if (y_src == 0)
            start_y = -1;
        else
            start_y = 0;

        for (x_src = 0; x_src < width-1; x_src++)
        {
            float lt = (float)src_ptr_top[0];
            float rt = (float)src_ptr_top[1];
            float lb = (float)src_ptr_bottom[0];
            float rb = (float)src_ptr_bottom[1];

            float a = lt;
            float b = (rt - lt) * finv;
            float c = (lb - lt) * finv;
            float d = ((rb - rt) - (lb - lt)) * finv * finv;
            
            uint *dst_ptr;
            float v0, v1, addy, addx;
            
            if (x_src == 0)
                start_x = -1;
            else
                start_x = 0;
                
            v0 = a + b*(dx + start_x) + c*(dy + start_y) + d*(dx + start_x)*(dy + start_y);
            addy = c + d*(dx + start_x);
            addx = b + d*(dy + start_y);
            
            dst_ptr = (uint*)((char*)dst + (y_src * 2 + start_y + 1) * stride_dst) + x_src * 2 + 1 + start_x;
            
            v1 = v0;
            dst_ptr[0] = (uint)SATURATE(v1, (float)0xffffffff);
            v1 += addx;
            dst_ptr[1] = (uint)SATURATE(v1, (float)0xffffffff);
            if (x_src == 0 || x_src == width - 2)
            {
                v1 += addx;
                dst_ptr[2] = (uint)SATURATE(v1, (float)0xffffffff);
            }

            dst_ptr = (uint*)((char*)dst_ptr + stride_dst);
            v0 += addy;
            addx = (float)(b + d*(dy + start_y + 1.0));

            v1 = v0;
            dst_ptr[0] = (uint)SATURATE(v1, (float)0xffffffff);
            v1 += addx;
            dst_ptr[1] = (uint)SATURATE(v1, (float)0xffffffff);
            if (x_src == 0 || x_src == width - 2)
            {
                v1 += addx;
                dst_ptr[2] = (uint)SATURATE(v1, (float)0xffffffff);
            }
                
            if (y_src == 0 || y_src == height - 2)
            {
                dst_ptr = (uint*)((char*)dst_ptr + stride_dst);
                v0 += addy;
                addx = (float)(b + d*(dy + start_y + 2.0));
                
                v1 = v0;
                dst_ptr[0] = (uint)SATURATE(v1, (float)0xffffffff);
                v1 += addx;
                dst_ptr[1] = (uint)SATURATE(v1, (float)0xffffffff);
                if (x_src == 0 || x_src == width - 2)
                {
                    v1 += addx;
                    dst_ptr[2] = (uint)SATURATE(v1, (float)0xffffffff);
                }
            }

            src_ptr_top++;
            src_ptr_bottom++;
        }
    }
}

static void upsample_4x4_8b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height)
{
    int x_src, y_src;

    const float finv = 0.25;
    const float dx = 0.5, dy = 0.5;
    
#if USE_SSE == 1
    int can_use_sse = 1;
    __m128 mul = _mm_set_ps(3.0, 2.0, 1.0, 0.0);
    __m128i zero = _mm_set1_epi16(0);
#endif

    #if USE_OMP == 1
    #pragma omp parallel for private(x_src, y_src)
    #endif
    for (y_src = 0; y_src < height-1; y_src++)
    {
        int start_x, start_y;
        const uchar *src_ptr_top, *src_ptr_bottom;
        
        src_ptr_top = (const uchar*)((char*)src + y_src * stride_src);
        src_ptr_bottom = (const uchar*)((char*)src + (y_src + 1) * stride_src);
        
        if (y_src == 0)
            start_y = -2;
        else
            start_y = 0;
        
        for (x_src = 0; x_src < width-1; x_src++)
        {
            float lt = src_ptr_top[0];
            float rt = src_ptr_top[1];
            float lb = src_ptr_bottom[0];
            float rb = src_ptr_bottom[1];
            
            float a = lt;
            float b = (rt - lt) * finv;
            float c = (lb - lt) * finv;
            float d = ((rb - rt) - (lb - lt)) * finv * finv;
            
            uchar *dst_ptr;
            float v0, v1, addy, addx;
            
            if (x_src == 0)
                start_x = -2;
            else
                start_x = 0;
            
            v0 = a + b*(dx + start_x) + c*(dy + start_y) + d*(dx + start_x)*(dy + start_y);
            addy = c + d*(dx + start_x);
            addx = b + d*(dy + start_y);
            
            dst_ptr = (uchar*)((char*)dst + (y_src * 4 + 2 + start_y) * stride_dst) + x_src * 4 + 2 + start_x;
            
#if USE_SSE == 1
            if (can_use_sse && x_src != 0 && y_src != 0 && x_src != width-2 && y_src != height-2)
            {
                __m128 mv0 = _mm_set1_ps(v0);
                __m128 maddx = _mm_set1_ps(addx);
                __m128 maddy = _mm_set1_ps(addy);
                __m128i mv;
                uchar *u8 = (uchar*)&mv;
                
                mv = _mm_packus_epi16(_mm_cvtps_epi32(
                    _mm_add_ps(mv0, _mm_mul_ps(mul, maddx))), zero);
                    
                dst_ptr[0] = u8[0];
                dst_ptr[1] = u8[2];
                dst_ptr[2] = u8[4];
                dst_ptr[3] = u8[6];

                dst_ptr = (uchar*)((char*)dst_ptr + stride_dst);
                mv0 = _mm_add_ps(mv0, maddy);
                maddx = _mm_set1_ps(b + d*(dy + start_y + 1.0));
                
                mv = _mm_packus_epi16(_mm_cvtps_epi32(
                    _mm_add_ps(mv0, _mm_mul_ps(mul, maddx))), zero);
                dst_ptr[0] = u8[0];
                dst_ptr[1] = u8[2];
                dst_ptr[2] = u8[4];
                dst_ptr[3] = u8[6];

                dst_ptr = (uchar*)((char*)dst_ptr + stride_dst);
                mv0 = _mm_add_ps(mv0, maddy);
                maddx = _mm_set1_ps(b + d*(dy + start_y + 1.0));
                
                mv = _mm_packus_epi16(_mm_cvtps_epi32(
                    _mm_add_ps(mv0, _mm_mul_ps(mul, maddx))), zero);
                dst_ptr[0] = u8[0];
                dst_ptr[1] = u8[2];
                dst_ptr[2] = u8[4];
                dst_ptr[3] = u8[6];

                dst_ptr = (uchar*)((char*)dst_ptr + stride_dst);
                mv0 = _mm_add_ps(mv0, maddy);
                maddx = _mm_set1_ps(b + d*(dy + start_y + 1.0));
                
                mv = _mm_packus_epi16(_mm_cvtps_epi32(
                    _mm_add_ps(mv0, _mm_mul_ps(mul, maddx))), zero);
                dst_ptr[0] = u8[0];
                dst_ptr[1] = u8[2];
                dst_ptr[2] = u8[4];
                dst_ptr[3] = u8[6];
            }
            else
#endif
            {
                v1 = v0;
                dst_ptr[0] = (uchar)SATURATE(v1, (float)0xff);
                v1 += addx;
                dst_ptr[1] = (uchar)SATURATE(v1, (float)0xff);
                v1 += addx;
                dst_ptr[2] = (uchar)SATURATE(v1, (float)0xff);
                v1 += addx;
                dst_ptr[3] = (uchar)SATURATE(v1, (float)0xff);
                if (x_src == 0 || x_src == width - 2)
                {
                    v1 += addx;
                    dst_ptr[4] = (uchar)SATURATE(v1, (float)0xff);
                    v1 += addx;
                    dst_ptr[5] = (uchar)SATURATE(v1, (float)0xff);
                }

                dst_ptr = (uchar*)((char*)dst_ptr + stride_dst);
                v0 += addy;
                addx = (float)(b + d*(dy + start_y + 1.0));
                
                v1 = v0;
                dst_ptr[0] = (uchar)SATURATE(v1, (float)0xff);
                v1 += addx;
                dst_ptr[1] = (uchar)SATURATE(v1, (float)0xff);
                v1 += addx;
                dst_ptr[2] = (uchar)SATURATE(v1, (float)0xff);
                v1 += addx;
                dst_ptr[3] = (uchar)SATURATE(v1, (float)0xff);
                if (x_src == 0 || x_src == width - 2)
                {
                    v1 += addx;
                    dst_ptr[4] = (uchar)SATURATE(v1, (float)0xff);
                    v1 += addx;
                    dst_ptr[5] = (uchar)SATURATE(v1, (float)0xff);
                }

                dst_ptr = (uchar*)((char*)dst_ptr + stride_dst);
                v0 += addy;
                addx = (float)(b + d*(dy + start_y + 1.0));
                
                v1 = v0;
                dst_ptr[0] = (uchar)SATURATE(v1, (float)0xff);
                v1 += addx;
                dst_ptr[1] = (uchar)SATURATE(v1, (float)0xff);
                v1 += addx;
                dst_ptr[2] = (uchar)SATURATE(v1, (float)0xff);
                v1 += addx;
                dst_ptr[3] = (uchar)SATURATE(v1, (float)0xff);
                if (x_src == 0 || x_src == width - 2)
                {
                    v1 += addx;
                    dst_ptr[4] = (uchar)SATURATE(v1, (float)0xff);
                    v1 += addx;
                    dst_ptr[5] = (uchar)SATURATE(v1, (float)0xff);
                }

                dst_ptr = (uchar*)((char*)dst_ptr + stride_dst);
                v0 += addy;
                addx = (float)(b + d*(dy + start_y + 1.0));
                
                v1 = v0;
                dst_ptr[0] = (uchar)SATURATE(v1, (float)0xff);
                v1 += addx;
                dst_ptr[1] = (uchar)SATURATE(v1, (float)0xff);
                v1 += addx;
                dst_ptr[2] = (uchar)SATURATE(v1, (float)0xff);
                v1 += addx;
                dst_ptr[3] = (uchar)SATURATE(v1, (float)0xff);
                if (x_src == 0 || x_src == width - 2)
                {
                    v1 += addx;
                    dst_ptr[4] = (uchar)SATURATE(v1, (float)0xff);
                    v1 += addx;
                    dst_ptr[5] = (uchar)SATURATE(v1, (float)0xff);
                }

                if (y_src == 0 || y_src == height - 2)
                {
                    dst_ptr = (uchar*)((char*)dst_ptr + stride_dst);
                    v0 += addy;
                    addx = (float)(b + d*(dy + start_y + 1.0));
                    
                    v1 = v0;
                    dst_ptr[0] = (uchar)SATURATE(v1, (float)0xff);
                    v1 += addx;
                    dst_ptr[1] = (uchar)SATURATE(v1, (float)0xff);
                    v1 += addx;
                    dst_ptr[2] = (uchar)SATURATE(v1, (float)0xff);
                    v1 += addx;
                    dst_ptr[3] = (uchar)SATURATE(v1, (float)0xff);
                    if (x_src == 0 || x_src == width - 2)
                    {
                        v1 += addx;
                        dst_ptr[4] = (uchar)SATURATE(v1, (float)0xff);
                        v1 += addx;
                        dst_ptr[5] = (uchar)SATURATE(v1, (float)0xff);
                    }
                    
                    dst_ptr = (uchar*)((char*)dst_ptr + stride_dst);
                    v0 += addy;
                    addx = (float)(b + d*(dy + start_y + 1.0));
                    
                    v1 = v0;
                    dst_ptr[0] = (uchar)SATURATE(v1, (float)0xff);
                    v1 += addx;
                    dst_ptr[1] = (uchar)SATURATE(v1, (float)0xff);
                    v1 += addx;
                    dst_ptr[2] = (uchar)SATURATE(v1, (float)0xff);
                    v1 += addx;
                    dst_ptr[3] = (uchar)SATURATE(v1, (float)0xff);
                    if (x_src == 0 || x_src == width - 2)
                    {
                        v1 += addx;
                        dst_ptr[4] = (uchar)SATURATE(v1, (float)0xff);
                        v1 += addx;
                        dst_ptr[5] = (uchar)SATURATE(v1, (float)0xff);
                    }
                }
            }

            src_ptr_top++;
            src_ptr_bottom++;
        }
    }
}

static void upsample_4x4_16b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height)
{
    int x_src, y_src;

    const float finv = 0.25;
    const float dx = 0.5, dy = 0.5;

    #if USE_OMP == 1
    #pragma omp parallel for private(x_src, y_src)
    #endif
    for (y_src = 0; y_src < height-1; y_src++)
    {
        int start_x, start_y;
        const ushort *src_ptr_top, *src_ptr_bottom;
        
        src_ptr_top = (const ushort*)((char*)src + y_src * stride_src);
        src_ptr_bottom = (const ushort*)((char*)src + (y_src + 1) * stride_src);
        
        if (y_src == 0)
            start_y = -2;
        else
            start_y = 0;
        
        for (x_src = 0; x_src < width-1; x_src++)
        {
            float lt = src_ptr_top[0];
            float rt = src_ptr_top[1];
            float lb = src_ptr_bottom[0];
            float rb = src_ptr_bottom[1];
            
            float a = lt;
            float b = (rt - lt) * finv;
            float c = (lb - lt) * finv;
            float d = ((rb - rt) - (lb - lt)) * finv * finv;
            
            ushort *dst_ptr;
            float v0, v1, addy, addx;
            
            if (x_src == 0)
                start_x = -2;
            else
                start_x = 0;
            
            v0 = a + b*(dx + start_x) + c*(dy + start_y) + d*(dx + start_x)*(dy + start_y);
            addy = c + d*(dx + start_x);
            addx = b + d*(dy + start_y);
            
            dst_ptr = (ushort*)((char*)dst + (y_src * 4 + 2 + start_y) * stride_dst) + x_src * 4 + 2 + start_x;
            
            {
                v1 = v0;
                dst_ptr[0] = (ushort)SATURATE(v1, (float)0xffff);
                v1 += addx;
                dst_ptr[1] = (ushort)SATURATE(v1, (float)0xffff);
                v1 += addx;
                dst_ptr[2] = (ushort)SATURATE(v1, (float)0xffff);
                v1 += addx;
                dst_ptr[3] = (ushort)SATURATE(v1, (float)0xffff);
                if (x_src == 0 || x_src == width - 2)
                {
                    v1 += addx;
                    dst_ptr[4] = (ushort)SATURATE(v1, (float)0xffff);
                    v1 += addx;
                    dst_ptr[5] = (ushort)SATURATE(v1, (float)0xffff);
                }

                dst_ptr = (ushort*)((char*)dst_ptr + stride_dst);
                v0 += addy;
                addx = (float)(b + d*(dy + start_y + 1.0));
                
                v1 = v0;
                dst_ptr[0] = (ushort)SATURATE(v1, (float)0xffff);
                v1 += addx;
                dst_ptr[1] = (ushort)SATURATE(v1, (float)0xffff);
                v1 += addx;
                dst_ptr[2] = (ushort)SATURATE(v1, (float)0xffff);
                v1 += addx;
                dst_ptr[3] = (ushort)SATURATE(v1, (float)0xffff);
                if (x_src == 0 || x_src == width - 2)
                {
                    v1 += addx;
                    dst_ptr[4] = (ushort)SATURATE(v1, (float)0xffff);
                    v1 += addx;
                    dst_ptr[5] = (ushort)SATURATE(v1, (float)0xffff);
                }

                dst_ptr = (ushort*)((char*)dst_ptr + stride_dst);
                v0 += addy;
                addx = (float)(b + d*(dy + start_y + 1.0));
                
                v1 = v0;
                dst_ptr[0] = (ushort)SATURATE(v1, (float)0xffff);
                v1 += addx;
                dst_ptr[1] = (ushort)SATURATE(v1, (float)0xffff);
                v1 += addx;
                dst_ptr[2] = (ushort)SATURATE(v1, (float)0xffff);
                v1 += addx;
                dst_ptr[3] = (ushort)SATURATE(v1, (float)0xffff);
                if (x_src == 0 || x_src == width - 2)
                {
                    v1 += addx;
                    dst_ptr[4] = (ushort)SATURATE(v1, (float)0xffff);
                    v1 += addx;
                    dst_ptr[5] = (ushort)SATURATE(v1, (float)0xffff);
                }

                dst_ptr = (ushort*)((char*)dst_ptr + stride_dst);
                v0 += addy;
                addx = (float)(b + d*(dy + start_y + 1.0));
                
                v1 = v0;
                dst_ptr[0] = (ushort)SATURATE(v1, (float)0xffff);
                v1 += addx;
                dst_ptr[1] = (ushort)SATURATE(v1, (float)0xffff);
                v1 += addx;
                dst_ptr[2] = (ushort)SATURATE(v1, (float)0xffff);
                v1 += addx;
                dst_ptr[3] = (ushort)SATURATE(v1, (float)0xffff);
                if (x_src == 0 || x_src == width - 2)
                {
                    v1 += addx;
                    dst_ptr[4] = (ushort)SATURATE(v1, (float)0xffff);
                    v1 += addx;
                    dst_ptr[5] = (ushort)SATURATE(v1, (float)0xffff);
                }

                if (y_src == 0 || y_src == height - 2)
                {
                    dst_ptr = (ushort*)((char*)dst_ptr + stride_dst);
                    v0 += addy;
                    addx = (float)(b + d*(dy + start_y + 1.0));
                    
                    v1 = v0;
                    dst_ptr[0] = (ushort)SATURATE(v1, (float)0xffff);
                    v1 += addx;
                    dst_ptr[1] = (ushort)SATURATE(v1, (float)0xffff);
                    v1 += addx;
                    dst_ptr[2] = (ushort)SATURATE(v1, (float)0xffff);
                    v1 += addx;
                    dst_ptr[3] = (ushort)SATURATE(v1, (float)0xffff);
                    if (x_src == 0 || x_src == width - 2)
                    {
                        v1 += addx;
                        dst_ptr[4] = (ushort)SATURATE(v1, (float)0xffff);
                        v1 += addx;
                        dst_ptr[5] = (ushort)SATURATE(v1, (float)0xffff);
                    }
                    
                    dst_ptr = (ushort*)((char*)dst_ptr + stride_dst);
                    v0 += addy;
                    addx = (float)(b + d*(dy + start_y + 1.0));
                    
                    v1 = v0;
                    dst_ptr[0] = (ushort)SATURATE(v1, (float)0xffff);
                    v1 += addx;
                    dst_ptr[1] = (ushort)SATURATE(v1, (float)0xffff);
                    v1 += addx;
                    dst_ptr[2] = (ushort)SATURATE(v1, (float)0xffff);
                    v1 += addx;
                    dst_ptr[3] = (ushort)SATURATE(v1, (float)0xffff);
                    if (x_src == 0 || x_src == width - 2)
                    {
                        v1 += addx;
                        dst_ptr[4] = (ushort)SATURATE(v1, (float)0xffff);
                        v1 += addx;
                        dst_ptr[5] = (ushort)SATURATE(v1, (float)0xffff);
                    }
                }
            }

            src_ptr_top++;
            src_ptr_bottom++;
        }
    }
}

static void upsample_4x4_32b(void *dst, int stride_dst, const void *src, int stride_src, int width, int height)
{
    int x_src, y_src;

    const float finv = 0.25;
    const float dx = 0.5, dy = 0.5;

    #if USE_OMP == 1
    #pragma omp parallel for private(x_src, y_src)
    #endif
    for (y_src = 0; y_src < height-1; y_src++)
    {
        int start_x, start_y;
        const uint *src_ptr_top, *src_ptr_bottom;
        
        src_ptr_top = (const uint*)((char*)src + y_src * stride_src);
        src_ptr_bottom = (const uint*)((char*)src + (y_src + 1) * stride_src);
        
        if (y_src == 0)
            start_y = -2;
        else
            start_y = 0;
        
        for (x_src = 0; x_src < width-1; x_src++)
        {
            float lt = (float)src_ptr_top[0];
            float rt = (float)src_ptr_top[1];
            float lb = (float)src_ptr_bottom[0];
            float rb = (float)src_ptr_bottom[1];
            
            float a = lt;
            float b = (rt - lt) * finv;
            float c = (lb - lt) * finv;
            float d = ((rb - rt) - (lb - lt)) * finv * finv;
            
            uint *dst_ptr;
            float v0, v1, addy, addx;
            
            if (x_src == 0)
                start_x = -2;
            else
                start_x = 0;
            
            v0 = a + b*(dx + start_x) + c*(dy + start_y) + d*(dx + start_x)*(dy + start_y);
            addy = c + d*(dx + start_x);
            addx = b + d*(dy + start_y);
            
            dst_ptr = (uint*)((char*)dst + (y_src * 4 + 2 + start_y) * stride_dst) + x_src * 4 + 2 + start_x;
            
            {
                v1 = v0;
                dst_ptr[0] = (uint)SATURATE(v1, (float)0xffffffff);
                v1 += addx;
                dst_ptr[1] = (uint)SATURATE(v1, (float)0xffffffff);
                v1 += addx;
                dst_ptr[2] = (uint)SATURATE(v1, (float)0xffffffff);
                v1 += addx;
                dst_ptr[3] = (uint)SATURATE(v1, (float)0xffffffff);
                if (x_src == 0 || x_src == width - 2)
                {
                    v1 += addx;
                    dst_ptr[4] = (uint)SATURATE(v1, (float)0xffffffff);
                    v1 += addx;
                    dst_ptr[5] = (uint)SATURATE(v1, (float)0xffffffff);
                }

                dst_ptr = (uint*)((char*)dst_ptr + stride_dst);
                v0 += addy;
                addx = (float)(b + d*(dy + start_y + 1.0));
                
                v1 = v0;
                dst_ptr[0] = (uint)SATURATE(v1, (float)0xffffffff);
                v1 += addx;
                dst_ptr[1] = (uint)SATURATE(v1, (float)0xffffffff);
                v1 += addx;
                dst_ptr[2] = (uint)SATURATE(v1, (float)0xffffffff);
                v1 += addx;
                dst_ptr[3] = (uint)SATURATE(v1, (float)0xffffffff);
                if (x_src == 0 || x_src == width - 2)
                {
                    v1 += addx;
                    dst_ptr[4] = (uint)SATURATE(v1, (float)0xffffffff);
                    v1 += addx;
                    dst_ptr[5] = (uint)SATURATE(v1, (float)0xffffffff);
                }

                dst_ptr = (uint*)((char*)dst_ptr + stride_dst);
                v0 += addy;
                addx = (float)(b + d*(dy + start_y + 1.0));
                
                v1 = v0;
                dst_ptr[0] = (uint)SATURATE(v1, (float)0xffffffff);
                v1 += addx;
                dst_ptr[1] = (uint)SATURATE(v1, (float)0xffffffff);
                v1 += addx;
                dst_ptr[2] = (uint)SATURATE(v1, (float)0xffffffff);
                v1 += addx;
                dst_ptr[3] = (uint)SATURATE(v1, (float)0xffffffff);
                if (x_src == 0 || x_src == width - 2)
                {
                    v1 += addx;
                    dst_ptr[4] = (uint)SATURATE(v1, (float)0xffffffff);
                    v1 += addx;
                    dst_ptr[5] = (uint)SATURATE(v1, (float)0xffffffff);
                }

                dst_ptr = (uint*)((char*)dst_ptr + stride_dst);
                v0 += addy;
                addx = (float)(b + d*(dy + start_y + 1.0));
                
                v1 = v0;
                dst_ptr[0] = (uint)SATURATE(v1, (float)0xffffffff);
                v1 += addx;
                dst_ptr[1] = (uint)SATURATE(v1, (float)0xffffffff);
                v1 += addx;
                dst_ptr[2] = (uint)SATURATE(v1, (float)0xffffffff);
                v1 += addx;
                dst_ptr[3] = (uint)SATURATE(v1, (float)0xffffffff);
                if (x_src == 0 || x_src == width - 2)
                {
                    v1 += addx;
                    dst_ptr[4] = (uint)SATURATE(v1, (float)0xffffffff);
                    v1 += addx;
                    dst_ptr[5] = (uint)SATURATE(v1, (float)0xffffffff);
                }

                if (y_src == 0 || y_src == height - 2)
                {
                    dst_ptr = (uint*)((char*)dst_ptr + stride_dst);
                    v0 += addy;
                    addx = (float)(b + d*(dy + start_y + 1.0));
                    
                    v1 = v0;
                    dst_ptr[0] = (uint)SATURATE(v1, (float)0xffffffff);
                    v1 += addx;
                    dst_ptr[1] = (uint)SATURATE(v1, (float)0xffffffff);
                    v1 += addx;
                    dst_ptr[2] = (uint)SATURATE(v1, (float)0xffffffff);
                    v1 += addx;
                    dst_ptr[3] = (uint)SATURATE(v1, (float)0xffffffff);
                    if (x_src == 0 || x_src == width - 2)
                    {
                        v1 += addx;
                        dst_ptr[4] = (uint)SATURATE(v1, (float)0xffffffff);
                        v1 += addx;
                        dst_ptr[5] = (uint)SATURATE(v1, (float)0xffffffff);
                    }
                    
                    dst_ptr = (uint*)((char*)dst_ptr + stride_dst);
                    v0 += addy;
                    addx = (float)(b + d*(dy + start_y + 1.0));
                    
                    v1 = v0;
                    dst_ptr[0] = (uint)SATURATE(v1, (float)0xffffffff);
                    v1 += addx;
                    dst_ptr[1] = (uint)SATURATE(v1, (float)0xffffffff);
                    v1 += addx;
                    dst_ptr[2] = (uint)SATURATE(v1, (float)0xffffffff);
                    v1 += addx;
                    dst_ptr[3] = (uint)SATURATE(v1, (float)0xffffffff);
                    if (x_src == 0 || x_src == width - 2)
                    {
                        v1 += addx;
                        dst_ptr[4] = (uint)SATURATE(v1, (float)0xffffffff);
                        v1 += addx;
                        dst_ptr[5] = (uint)SATURATE(v1, (float)0xffff);
                    }
                }
            }

            src_ptr_top++;
            src_ptr_bottom++;
        }
    }
}
