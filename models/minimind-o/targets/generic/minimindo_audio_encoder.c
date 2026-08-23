#define _POSIX_C_SOURCE 200809L

#include "minimindo_audio_encoder.h"
#include "minimindo_parallel.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

enum {
    HEADER_BYTES=4096, F32=1, Q8=2, FFT_SIZE=512, MEL_BINS=80,
    AUDIO_LAYERS=70, AUDIO_HIDDEN=512, AUDIO_PROJECTED=768,
    AUDIO_CHUNK_CENTER=8, AUDIO_CHUNK_RIGHT=4,
    AUDIO_CACHE_FRAMES=32,
    AUDIO_CHUNK_MAX=AUDIO_CHUNK_CENTER+AUDIO_CHUNK_RIGHT
};

typedef struct {
    unsigned char magic[8];
    uint32_t version, header_bytes, layers, base_layers, tp_layers;
    uint32_t input, hidden, heads, intermediate, fsmn_kernel;
    uint32_t mel_bins, fft_bins, lfr_m, lfr_n, tensor_count, reserved;
    float norm_epsilon, position_theta;
    uint64_t tensors_offset, file_bytes;
    char sense_sha256[64], minimind_sha256[64];
} audio_header;

typedef struct { uint32_t type, rows, cols, reserved; uint64_t data_bytes; } tensor_header;
typedef struct { const tensor_header *header; const unsigned char *data; } tensor;

typedef struct {
    tensor norm1_weight, norm1_bias;
    tensor q_weight, q_bias, k_weight, k_bias, v_weight, v_bias;
    tensor fsmn, out_weight, out_bias;
    tensor norm2_weight, norm2_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias;
} encoder_layer;

struct minimindo_audio_encoder {
    int file; const unsigned char *mapping; size_t mapped_bytes;
    const audio_header *header;
    tensor mel_filters, cmvn_mean, cmvn_scale, embeddings;
    encoder_layer *layers;
    tensor after_weight, after_bias, tp_weight, tp_bias;
    tensor projector_norm_weight, projector_norm_bias;
    tensor projector_first_weight, projector_first_bias;
    tensor projector_second_weight, projector_second_bias;
    minimindo_audio_encoder_profile profile;
};

struct minimindo_audio_encoder_stream {
    minimindo_audio_encoder *model;
    int16_t *pcm;
    size_t pcm_count;
    size_t pcm_capacity;
    uint32_t emitted_frames;
    uint32_t cache_count[AUDIO_LAYERS];
    float *cache_keys;
    float *cache_values;
    minimindo_audio_encoder_profile profile;
    int ended;
};

typedef struct { double real, imaginary; } complex_value;

_Static_assert(sizeof(audio_header)==224,"MiniMind-O audio image ABI");
_Static_assert(sizeof(tensor_header)==24,"MiniMind-O tensor ABI");

static void set_error(char *error,size_t capacity,const char *format,...)
{ if(!error||!capacity)return;va_list args;va_start(args,format);vsnprintf(error,capacity,format,args);va_end(args); }
static uint64_t align64(uint64_t x){return(x+63U)&~UINT64_C(63);}
static int range_ok(uint64_t o,uint64_t n,uint64_t total){return o<=total&&n<=total-o;}
static int take(const unsigned char *mapping,uint64_t bytes,uint64_t *cursor,
                uint32_t type,uint32_t rows,uint32_t cols,tensor *out)
{
    if(!range_ok(*cursor,sizeof(tensor_header),bytes))return -1;
    const tensor_header *h=(const tensor_header *)(mapping+*cursor);
    uint64_t expected=type==F32?(uint64_t)rows*cols*4:(uint64_t)rows*(4+cols);
    if(h->type!=type||h->rows!=rows||h->cols!=cols||h->reserved||h->data_bytes!=expected)return -1;
    uint64_t data=*cursor+sizeof(*h);if(!range_ok(data,expected,bytes))return -1;
    out->header=h;out->data=mapping+data;*cursor=align64(data+expected);return *cursor<=bytes?0:-1;
}
static const float *f32(const tensor *t){return(const float *)t->data;}
static double monotonic_seconds(void)
{struct timespec value;clock_gettime(CLOCK_MONOTONIC,&value);return value.tv_sec+value.tv_nsec*1e-9;}

static float q8_dot(const int8_t *w,const float *x,uint32_t count)
{
#if defined(__aarch64__)
    float32x4_t a=vdupq_n_f32(0),b=a,c=a,d=a;uint32_t i=0;
    for(;i+16<=count;i+=16){int8x16_t p=vld1q_s8(w+i);int16x8_t lo=vmovl_s8(vget_low_s8(p)),hi=vmovl_s8(vget_high_s8(p));
        a=vfmaq_f32(a,vld1q_f32(x+i),vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
        b=vfmaq_f32(b,vld1q_f32(x+i+4),vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))));
        c=vfmaq_f32(c,vld1q_f32(x+i+8),vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
        d=vfmaq_f32(d,vld1q_f32(x+i+12),vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))));}
    float sum=vaddvq_f32(vaddq_f32(vaddq_f32(a,b),vaddq_f32(c,d)));for(;i<count;++i)sum+=w[i]*x[i];return sum;
#else
    double sum=0;for(uint32_t i=0;i<count;++i)sum+=(double)w[i]*x[i];return(float)sum;
#endif
}

static void q8_dot4(const int8_t *w,const float *x,uint32_t stride,
                    uint32_t count,float output[4])
{
#if defined(__aarch64__)
    float32x4_t a0=vdupq_n_f32(0),b0=a0,c0=a0,d0=a0;
    float32x4_t a1=a0,b1=a0,c1=a0,d1=a0;
    float32x4_t a2=a0,b2=a0,c2=a0,d2=a0;
    float32x4_t a3=a0,b3=a0,c3=a0,d3=a0;uint32_t i=0;
    for(;i+16<=count;i+=16){int8x16_t p=vld1q_s8(w+i);int16x8_t lo=vmovl_s8(vget_low_s8(p)),hi=vmovl_s8(vget_high_s8(p));
        float32x4_t w0=vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),w1=vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
        float32x4_t w2=vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),w3=vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
#define DOT4_ROW(n,base) do{const float *v=x+(size_t)(base)*stride+i;a##n=vfmaq_f32(a##n,vld1q_f32(v),w0);b##n=vfmaq_f32(b##n,vld1q_f32(v+4),w1);c##n=vfmaq_f32(c##n,vld1q_f32(v+8),w2);d##n=vfmaq_f32(d##n,vld1q_f32(v+12),w3);}while(0)
        DOT4_ROW(0,0);DOT4_ROW(1,1);DOT4_ROW(2,2);DOT4_ROW(3,3);
#undef DOT4_ROW
    }
#define DOT4_SUM(n) vaddvq_f32(vaddq_f32(vaddq_f32(a##n,b##n),vaddq_f32(c##n,d##n)))
    output[0]=DOT4_SUM(0);output[1]=DOT4_SUM(1);output[2]=DOT4_SUM(2);output[3]=DOT4_SUM(3);
#undef DOT4_SUM
    for(;i<count;++i){float weight=w[i];output[0]+=weight*x[i];output[1]+=weight*x[stride+i];output[2]+=weight*x[(size_t)2*stride+i];output[3]+=weight*x[(size_t)3*stride+i];}
#else
    for(uint32_t p=0;p<4;++p)output[p]=q8_dot(w,x+(size_t)p*stride,count);
#endif
}

static int32_t q8_i8_dot(const int8_t *weights,const int8_t *input,
                         uint32_t count)
{
#if defined(__aarch64__)
    int32x4_t sum0=vdupq_n_s32(0),sum1=sum0;uint32_t i=0;
    for(;i+16<=count;i+=16){
        int8x16_t w=vld1q_s8(weights+i),x=vld1q_s8(input+i);
        sum0=vpadalq_s16(sum0,vmull_s8(vget_low_s8(w),vget_low_s8(x)));
        sum1=vpadalq_s16(sum1,vmull_s8(vget_high_s8(w),vget_high_s8(x)));
    }
    int32_t sum=vaddvq_s32(vaddq_s32(sum0,sum1));
    for(;i<count;++i)sum+=weights[i]*input[i];
    return sum;
#else
    int32_t sum=0;for(uint32_t i=0;i<count;++i)sum+=weights[i]*input[i];return sum;
#endif
}

static void q8_i8_dot4(const int8_t *weights,const int8_t *input,
                       uint32_t stride,uint32_t count,int32_t output[4])
{
#if defined(__aarch64__)
    int32x4_t lo0=vdupq_n_s32(0),hi0=lo0,lo1=lo0,hi1=lo0;
    int32x4_t lo2=lo0,hi2=lo0,lo3=lo0,hi3=lo0;uint32_t i=0;
    for(;i+16<=count;i+=16){
        int8x16_t w=vld1q_s8(weights+i);
#define I8_DOT4_LANE(n) do { \
        int8x16_t x=vld1q_s8(input+(size_t)(n)*stride+i); \
        lo##n=vpadalq_s16(lo##n,vmull_s8(vget_low_s8(w),vget_low_s8(x))); \
        hi##n=vpadalq_s16(hi##n,vmull_s8(vget_high_s8(w),vget_high_s8(x))); \
    } while(0)
        I8_DOT4_LANE(0);I8_DOT4_LANE(1);I8_DOT4_LANE(2);I8_DOT4_LANE(3);
#undef I8_DOT4_LANE
    }
    output[0]=vaddvq_s32(vaddq_s32(lo0,hi0));
    output[1]=vaddvq_s32(vaddq_s32(lo1,hi1));
    output[2]=vaddvq_s32(vaddq_s32(lo2,hi2));
    output[3]=vaddvq_s32(vaddq_s32(lo3,hi3));
    for(;i<count;++i){int8_t w=weights[i];
        output[0]+=w*input[i];output[1]+=w*input[stride+i];
        output[2]+=w*input[(size_t)2*stride+i];
        output[3]+=w*input[(size_t)3*stride+i];}
#else
    for(uint32_t p=0;p<4;++p)
        output[p]=q8_i8_dot(weights,input+(size_t)p*stride,count);
#endif
}

static float quantize_i8(const float *input,int8_t *output,uint32_t count)
{
    float maximum=0.0f;
#if defined(__aarch64__)
    float32x4_t vmax=vdupq_n_f32(0.0f);uint32_t i=0;
    for(;i+4<=count;i+=4)vmax=vmaxq_f32(vmax,vabsq_f32(vld1q_f32(input+i)));
    maximum=vmaxvq_f32(vmax);
    for(;i<count;++i)if(fabsf(input[i])>maximum)maximum=fabsf(input[i]);
#else
    for(uint32_t i=0;i<count;++i)if(fabsf(input[i])>maximum)maximum=fabsf(input[i]);
#endif
    if(!(maximum>0.0f)){memset(output,0,count);return 0.0f;}
    float scale=maximum/127.0f,inverse=1.0f/scale;
#if defined(__aarch64__)
    float32x4_t vinverse=vdupq_n_f32(inverse);i=0;
    for(;i+16<=count;i+=16){
        int32x4_t q0=vcvtnq_s32_f32(vmulq_f32(vld1q_f32(input+i),vinverse));
        int32x4_t q1=vcvtnq_s32_f32(vmulq_f32(vld1q_f32(input+i+4),vinverse));
        int32x4_t q2=vcvtnq_s32_f32(vmulq_f32(vld1q_f32(input+i+8),vinverse));
        int32x4_t q3=vcvtnq_s32_f32(vmulq_f32(vld1q_f32(input+i+12),vinverse));
        int16x8_t lo=vcombine_s16(vqmovn_s32(q0),vqmovn_s32(q1));
        int16x8_t hi=vcombine_s16(vqmovn_s32(q2),vqmovn_s32(q3));
        vst1q_s8(output+i,vcombine_s8(vqmovn_s16(lo),vqmovn_s16(hi)));
    }
    for(;i<count;++i)output[i]=(int8_t)lrintf(input[i]*inverse);
#else
    for(uint32_t i=0;i<count;++i)output[i]=(int8_t)lrintf(input[i]*inverse);
#endif
    return scale;
}
static void row(const tensor *t,uint32_t r,const int8_t **w,float *scale)
{const unsigned char *p=t->data+(size_t)r*(4+t->header->cols);memcpy(scale,p,4);*w=(const int8_t *)(p+4);}

typedef struct {
    const tensor *matrix;
    const float *input;
    uint32_t length;
    uint32_t input_width;
    float *output;
    const float *bias;
} matrix_parallel_context;

static void matrix_rows(void *opaque,size_t begin,size_t end)
{
    matrix_parallel_context *c=opaque;const tensor *matrix=c->matrix;
    for(size_t out=begin;out<end;++out){const int8_t *w;float scale;row(matrix,(uint32_t)out,&w,&scale);
        uint32_t p=0;for(;p+4<=c->length;p+=4){float sums[4];q8_dot4(w,c->input+(size_t)p*c->input_width,c->input_width,c->input_width,sums);
            for(uint32_t lane=0;lane<4;++lane)c->output[(size_t)(p+lane)*matrix->header->rows+out]=sums[lane]*scale+(c->bias?c->bias[out]:0);}
        for(;p<c->length;++p)c->output[(size_t)p*matrix->header->rows+out]=q8_dot(w,c->input+(size_t)p*c->input_width,c->input_width)*scale+(c->bias?c->bias[out]:0);
    }
}

static void matrix_sequence(const tensor *matrix,const float *input,uint32_t length,
                            uint32_t input_width,float *output,const float *bias)
{matrix_parallel_context c={matrix,input,length,input_width,output,bias};minimindo_parallel_for(matrix->header->rows,matrix_rows,&c);}

typedef struct {
    const float *input;
    int8_t *quantized;
    float *scales;
    uint32_t width;
} quantize_parallel_context;

static void quantize_positions(void *opaque,size_t begin,size_t end)
{
    quantize_parallel_context *c=opaque;
    for(size_t p=begin;p<end;++p)
        c->scales[p]=quantize_i8(c->input+p*c->width,
                                 c->quantized+p*c->width,c->width);
}

typedef struct {
    const tensor *matrix;
    const int8_t *input;
    const float *input_scales;
    uint32_t length;
    uint32_t input_width;
    float *output;
    const float *bias;
} matrix_i8_parallel_context;

static void matrix_i8_rows(void *opaque,size_t begin,size_t end)
{
    matrix_i8_parallel_context *c=opaque;
    for(size_t out=begin;out<end;++out){
        const int8_t *weights;float weight_scale;
        row(c->matrix,(uint32_t)out,&weights,&weight_scale);
        uint32_t p=0;
        for(;p+4<=c->length;p+=4){
            int32_t sums[4];
            q8_i8_dot4(weights,c->input+(size_t)p*c->input_width,
                       c->input_width,c->input_width,sums);
            for(uint32_t lane=0;lane<4;++lane)
                c->output[(size_t)(p+lane)*c->matrix->header->rows+out]=
                    sums[lane]*weight_scale*c->input_scales[p+lane]+
                    (c->bias?c->bias[out]:0.0f);
        }
        for(;p<c->length;++p)
            c->output[(size_t)p*c->matrix->header->rows+out]=
                q8_i8_dot(weights,c->input+(size_t)p*c->input_width,
                          c->input_width)*weight_scale*c->input_scales[p]+
                (c->bias?c->bias[out]:0.0f);
    }
}

static void quantize_sequence_i8(const float *input,uint32_t length,
                                 uint32_t input_width,int8_t *quantized,
                                 float *scales)
{
    quantize_parallel_context quantize={input,quantized,scales,input_width};
    minimindo_parallel_for(length,quantize_positions,&quantize);
}

static void matrix_sequence_i8_quantized(
    const tensor *matrix,const int8_t *quantized,const float *scales,
    uint32_t length,uint32_t input_width,float *output,const float *bias)
{
    matrix_i8_parallel_context matrix_context={
        matrix,quantized,scales,length,input_width,output,bias};
    minimindo_parallel_for(matrix->header->rows,matrix_i8_rows,&matrix_context);
}

typedef struct {const float *input;float *output;uint32_t width;const float *weight;const float *bias;float epsilon;} norm_parallel_context;
static void norm_positions(void *opaque,size_t begin,size_t end)
{norm_parallel_context *c=opaque;for(size_t p=begin;p<end;++p){const float *x=c->input+p*c->width;float *y=c->output+p*c->width;double mean=0,sq=0;
    for(uint32_t i=0;i<c->width;++i){mean+=x[i];sq+=(double)x[i]*x[i];}mean/=c->width;double scale=1.0/sqrt(sq/c->width-mean*mean+c->epsilon);
    for(uint32_t i=0;i<c->width;++i)y[i]=(float)((x[i]-mean)*scale*c->weight[i]+c->bias[i]);}}

static void layer_norm_sequence(const float *input,float *output,uint32_t length,uint32_t width,
                                const float *weight,const float *bias,float epsilon)
{
    norm_parallel_context c={input,output,width,weight,bias,epsilon};minimindo_parallel_for(length,norm_positions,&c);
}

static void fft512(complex_value *values)
{
    for(uint32_t i=1,j=0;i<FFT_SIZE;++i){uint32_t bit=FFT_SIZE>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){complex_value t=values[i];values[i]=values[j];values[j]=t;}}
    for(uint32_t length=2;length<=FFT_SIZE;length<<=1){double angle=-2.0*3.14159265358979323846/length;complex_value step={cos(angle),sin(angle)};
        for(uint32_t start=0;start<FFT_SIZE;start+=length){complex_value w={1,0};for(uint32_t j=0;j<length/2;++j){complex_value u=values[start+j],v=values[start+j+length/2];
            double vr=v.real*w.real-v.imaginary*w.imaginary,vi=v.real*w.imaginary+v.imaginary*w.real;
            values[start+j]=(complex_value){u.real+vr,u.imaginary+vi};values[start+j+length/2]=(complex_value){u.real-vr,u.imaginary-vi};
            w=(complex_value){w.real*step.real-w.imaginary*step.imaginary,w.real*step.imaginary+w.imaginary*step.real};}}}
}

typedef struct {const minimindo_audio_encoder *model;const int16_t *samples;uint32_t frames;float *mel;} frontend_parallel_context;
static void frontend_frames(void *opaque,size_t begin,size_t end)
{frontend_parallel_context *c=opaque;const float *filters=f32(&c->model->mel_filters);
    for(size_t frame=begin;frame<end;++frame){complex_value values[FFT_SIZE];double mean=0;
        for(uint32_t i=0;i<400;++i)mean+=c->samples[frame*160+i];
        mean/=400;
        double previous=c->samples[frame*160]-mean;
        for(uint32_t i=0;i<400;++i){double current=c->samples[frame*160+i]-mean;double emphasized=i?current-0.97*previous:current-0.97*current;previous=current;
            values[i].real=emphasized*(0.54-0.46*cos(2.0*3.14159265358979323846*i/400.0));values[i].imaginary=0;}
        for(uint32_t i=400;i<FFT_SIZE;++i)values[i]=(complex_value){0,0};
        fft512(values);double power[256];for(uint32_t k=0;k<256;++k)power[k]=values[k].real*values[k].real+values[k].imaginary*values[k].imaginary;
        for(uint32_t m=0;m<80;++m){double sum=0;for(uint32_t k=0;k<256;++k)sum+=power[k]*filters[(size_t)m*256+k];c->mel[frame*80+m]=(float)log(sum>1.19e-7?sum:1.19e-7);}}
}

size_t minimindo_audio_encoder_frames(size_t samples)
{
    if(samples<400)return 0;
    size_t fbank=1+(samples-400)/160;
    return(fbank+5)/6;
}

static float *frontend(const minimindo_audio_encoder *model,const int16_t *samples,
                       size_t sample_count,uint32_t *out_frames)
{
    if(sample_count<400)return NULL;
    uint32_t frames=1+(uint32_t)((sample_count-400)/160);
    float *mel=malloc((size_t)frames*80*sizeof(float));if(!mel)return NULL;
    frontend_parallel_context c={model,samples,frames,mel};minimindo_parallel_for(frames,frontend_frames,&c);
    uint32_t lfr=(frames+5)/6;float *features=malloc((size_t)lfr*560*sizeof(float));if(!features){free(mel);return NULL;}
    const float *mean=f32(&model->cmvn_mean),*scale=f32(&model->cmvn_scale);
    for(uint32_t t=0;t<lfr;++t)for(uint32_t j=0;j<7;++j){int64_t source=(int64_t)t*6-3+j;if(source<0)source=0;if(source>=(int64_t)frames)source=frames-1;
        for(uint32_t m=0;m<80;++m){uint32_t col=j*80+m;features[(size_t)t*560+col]=(mel[(size_t)source*80+m]+mean[col])*scale[col];}}
    free(mel);*out_frames=lfr;return features;
}

minimindo_audio_encoder *minimindo_audio_encoder_open(const char *path,char *error,size_t capacity)
{
    minimindo_audio_encoder *m=calloc(1,sizeof(*m));if(!m)return NULL;m->file=-1;m->file=open(path,O_RDONLY);struct stat st;
    if(m->file<0||fstat(m->file,&st)||st.st_size<HEADER_BYTES){set_error(error,capacity,"cannot open audio image: %s",strerror(errno));goto bad;}
    m->mapped_bytes=(size_t)st.st_size;m->mapping=mmap(NULL,m->mapped_bytes,PROT_READ,MAP_PRIVATE,m->file,0);if(m->mapping==MAP_FAILED){m->mapping=NULL;goto bad;}
    m->header=(const audio_header *)m->mapping;static const unsigned char magic[8]={'M','M','O','S','E','N','S','1'};
    if(memcmp(m->header->magic,magic,8)||m->header->version!=1||m->header->header_bytes!=HEADER_BYTES||m->header->file_bytes!=m->mapped_bytes||
       m->header->layers!=70||m->header->base_layers!=50||m->header->tp_layers!=20||m->header->input!=560||m->header->hidden!=512||m->header->tensor_count!=1204){set_error(error,capacity,"invalid audio image header");goto bad;}
    m->layers=calloc(m->header->layers,sizeof(*m->layers));if(!m->layers)goto bad;uint64_t c=m->header->tensors_offset;
#define TAKE(x,type,rows,cols) if(take(m->mapping,m->mapped_bytes,&c,type,rows,cols,x))goto tensors
    TAKE(&m->mel_filters,F32,80,256);TAKE(&m->cmvn_mean,F32,1,560);TAKE(&m->cmvn_scale,F32,1,560);TAKE(&m->embeddings,F32,16,560);
    for(uint32_t li=0;li<70;++li){encoder_layer *l=&m->layers[li];uint32_t in=li?512:560;
        TAKE(&l->norm1_weight,F32,1,in);TAKE(&l->norm1_bias,F32,1,in);
        TAKE(&l->q_weight,Q8,512,in);TAKE(&l->q_bias,F32,1,512);TAKE(&l->k_weight,Q8,512,in);TAKE(&l->k_bias,F32,1,512);TAKE(&l->v_weight,Q8,512,in);TAKE(&l->v_bias,F32,1,512);
        TAKE(&l->fsmn,F32,512,11);TAKE(&l->out_weight,Q8,512,512);TAKE(&l->out_bias,F32,1,512);
        TAKE(&l->norm2_weight,F32,1,512);TAKE(&l->norm2_bias,F32,1,512);TAKE(&l->fc1_weight,Q8,2048,512);TAKE(&l->fc1_bias,F32,1,2048);TAKE(&l->fc2_weight,Q8,512,2048);TAKE(&l->fc2_bias,F32,1,512);}
    TAKE(&m->after_weight,F32,1,512);TAKE(&m->after_bias,F32,1,512);TAKE(&m->tp_weight,F32,1,512);TAKE(&m->tp_bias,F32,1,512);
    TAKE(&m->projector_norm_weight,F32,1,512);TAKE(&m->projector_norm_bias,F32,1,512);TAKE(&m->projector_first_weight,Q8,768,512);TAKE(&m->projector_first_bias,F32,1,768);TAKE(&m->projector_second_weight,Q8,768,768);TAKE(&m->projector_second_bias,F32,1,768);
#undef TAKE
    if(c!=m->mapped_bytes)goto tensors;
    return m;
tensors:set_error(error,capacity,"invalid audio image tensors");
bad:minimindo_audio_encoder_close(m);return NULL;
}

void minimindo_audio_encoder_close(minimindo_audio_encoder *m)
{if(!m)return;free(m->layers);if(m->mapping)munmap((void *)m->mapping,m->mapped_bytes);if(m->file>=0)close(m->file);free(m);}

static int encoder_forward(minimindo_audio_encoder *model,float *sequence,uint32_t length)
{
    const uint32_t h=512,heads=4,d=128;float *norm=malloc((size_t)length*560*4),*q=malloc((size_t)length*h*4),*k=malloc((size_t)length*h*4),*v=malloc((size_t)length*h*4);
    float *memory=malloc((size_t)length*h*4),*attention=malloc((size_t)length*h*4),*projected=malloc((size_t)length*h*4),*ff=malloc((size_t)length*2048*4);
    int8_t *quantized=malloc((size_t)length*2048);float *scales=malloc((size_t)length*sizeof(float));
    if(!norm||!q||!k||!v||!memory||!attention||!projected||!ff||!quantized||!scales)goto failed;
    for(uint32_t li=0;li<70;++li){encoder_layer *l=&model->layers[li];uint32_t in=li?512:560;
        layer_norm_sequence(sequence,norm,length,in,f32(&l->norm1_weight),f32(&l->norm1_bias),model->header->norm_epsilon);
        quantize_sequence_i8(norm,length,in,quantized,scales);
        matrix_sequence_i8_quantized(&l->q_weight,quantized,scales,length,in,q,f32(&l->q_bias));
        matrix_sequence_i8_quantized(&l->k_weight,quantized,scales,length,in,k,f32(&l->k_bias));
        matrix_sequence_i8_quantized(&l->v_weight,quantized,scales,length,in,v,f32(&l->v_bias));
        const float *fw=f32(&l->fsmn);for(uint32_t p=0;p<length;++p)for(uint32_t i=0;i<h;++i){double sum=v[(size_t)p*h+i];for(uint32_t kernel=0;kernel<11;++kernel){int64_t source=(int64_t)p-5+kernel;if(source>=0&&source<length)sum+=v[(size_t)source*h+i]*fw[(size_t)i*11+kernel];}memory[(size_t)p*h+i]=(float)sum;}
        for(uint32_t query=0;query<length;++query)for(uint32_t head=0;head<heads;++head){double maximum=-DBL_MAX;double scores[length];
            for(uint32_t source=0;source<length;++source){double score=0;for(uint32_t j=0;j<d;++j)score+=(double)q[(size_t)query*h+head*d+j]*k[(size_t)source*h+head*d+j];scores[source]=score/sqrt((double)d);if(scores[source]>maximum)maximum=scores[source];}
            double denominator=0;for(uint32_t source=0;source<length;++source){scores[source]=exp(scores[source]-maximum);denominator+=scores[source];}
            for(uint32_t j=0;j<d;++j){double sum=0;for(uint32_t source=0;source<length;++source)sum+=scores[source]*v[(size_t)source*h+head*d+j];attention[(size_t)query*h+head*d+j]=(float)(sum/denominator);}}
        quantize_sequence_i8(attention,length,h,quantized,scales);
        matrix_sequence_i8_quantized(&l->out_weight,quantized,scales,length,h,
                                     projected,f32(&l->out_bias));
        for(size_t i=0;i<(size_t)length*h;++i)projected[i]+=memory[i]+(li?sequence[i]:0);
        layer_norm_sequence(projected,norm,length,h,f32(&l->norm2_weight),f32(&l->norm2_bias),model->header->norm_epsilon);
        quantize_sequence_i8(norm,length,h,quantized,scales);
        matrix_sequence_i8_quantized(&l->fc1_weight,quantized,scales,length,h,
                                     ff,f32(&l->fc1_bias));
        for(size_t i=0;i<(size_t)length*2048;++i)if(ff[i]<0)ff[i]=0;
        quantize_sequence_i8(ff,length,2048,quantized,scales);
        matrix_sequence_i8_quantized(&l->fc2_weight,quantized,scales,length,
                                     2048,attention,f32(&l->fc2_bias));
        for(size_t i=0;i<(size_t)length*h;++i)attention[i]+=projected[i];
        if(li==49)layer_norm_sequence(attention,projected,length,h,f32(&model->after_weight),f32(&model->after_bias),model->header->norm_epsilon);else memcpy(projected,attention,(size_t)length*h*4);
        memcpy(sequence,projected,(size_t)length*h*4);
    }
    layer_norm_sequence(sequence,projected,length,h,f32(&model->tp_weight),f32(&model->tp_bias),model->header->norm_epsilon);memcpy(sequence,projected,(size_t)length*h*4);
    free(norm);free(q);free(k);free(v);free(memory);free(attention);
    free(projected);free(ff);free(quantized);free(scales);return 0;
failed:
    free(norm);free(q);free(k);free(v);free(memory);free(attention);
    free(projected);free(ff);free(quantized);free(scales);return -1;
}

typedef struct {
    const float *values;
    const float *weights;
    float *output;
    uint32_t length;
} fsmn_chunk_context;

static void fsmn_chunk_cells(void *opaque,size_t begin,size_t end)
{
    fsmn_chunk_context *c=opaque;
    for(size_t cell=begin;cell<end;++cell){
        uint32_t p=(uint32_t)(cell/AUDIO_HIDDEN);
        uint32_t i=(uint32_t)(cell%AUDIO_HIDDEN);
        double sum=c->values[(size_t)p*AUDIO_HIDDEN+i];
        for(uint32_t kernel=0;kernel<11;++kernel){
            int64_t source=(int64_t)p-5+(int64_t)kernel;
            if(source>=0&&source<c->length)
                sum+=c->values[(size_t)source*AUDIO_HIDDEN+i]*
                     c->weights[(size_t)i*11+kernel];
        }
        c->output[cell]=(float)sum;
    }
}

typedef struct {
    const float *queries;
    const float *keys;
    const float *values;
    const float *cached_keys;
    const float *cached_values;
    float *output;
    uint32_t length;
    uint32_t cached;
} attention_chunk_context;

static void attention_chunk_heads(void *opaque,size_t begin,size_t end)
{
    enum { HEADS=4, HEAD_WIDTH=128 };
    attention_chunk_context *c=opaque;
    for(size_t task=begin;task<end;++task){
        uint32_t query=(uint32_t)(task/HEADS);
        uint32_t head=(uint32_t)(task%HEADS);
        uint32_t sources=c->cached+c->length;
        double scores[AUDIO_CACHE_FRAMES+AUDIO_CHUNK_MAX];
        double maximum=-DBL_MAX;
        const float *q=c->queries+(size_t)query*AUDIO_HIDDEN+head*HEAD_WIDTH;
        for(uint32_t source=0;source<sources;++source){
            const float *k=(source<c->cached?c->cached_keys+(size_t)source*AUDIO_HIDDEN:
                            c->keys+(size_t)(source-c->cached)*AUDIO_HIDDEN)+head*HEAD_WIDTH;
            double score=0;
            for(uint32_t j=0;j<HEAD_WIDTH;++j)score+=(double)q[j]*k[j];
            scores[source]=score/sqrt((double)HEAD_WIDTH);
            if(scores[source]>maximum)maximum=scores[source];
        }
        double denominator=0;
        for(uint32_t source=0;source<sources;++source){
            scores[source]=exp(scores[source]-maximum);
            denominator+=scores[source];
        }
        float *out=c->output+(size_t)query*AUDIO_HIDDEN+head*HEAD_WIDTH;
        for(uint32_t j=0;j<HEAD_WIDTH;++j){
            double sum=0;
            for(uint32_t source=0;source<sources;++source){
                const float *v=(source<c->cached?c->cached_values+(size_t)source*AUDIO_HIDDEN:
                                c->values+(size_t)(source-c->cached)*AUDIO_HIDDEN)+head*HEAD_WIDTH;
                sum+=scores[source]*v[j];
            }
            out[j]=(float)(sum/denominator);
        }
    }
}

typedef struct { float *values; } gelu_context;
static void gelu_values(void *opaque,size_t begin,size_t end)
{
    gelu_context *c=opaque;
    for(size_t i=begin;i<end;++i){
        float x=c->values[i];
        c->values[i]=0.5f*x*(1.0f+erff(x*0.7071067811865475f));
    }
}

static int project_embeddings(minimindo_audio_encoder *model,
                              const float *sequence,uint32_t frames,
                              float *output)
{
    if(frames==0)return 0;
    float *normalized=malloc((size_t)frames*AUDIO_HIDDEN*sizeof(float));
    float *first=malloc((size_t)frames*AUDIO_PROJECTED*sizeof(float));
    if(!normalized||!first){free(normalized);free(first);return -1;}
    layer_norm_sequence(sequence,normalized,frames,AUDIO_HIDDEN,
                        f32(&model->projector_norm_weight),
                        f32(&model->projector_norm_bias),1e-5f);
    matrix_sequence(&model->projector_first_weight,normalized,frames,
                    AUDIO_HIDDEN,first,f32(&model->projector_first_bias));
    gelu_context gelu={first};
    minimindo_parallel_for((size_t)frames*AUDIO_PROJECTED,gelu_values,&gelu);
    matrix_sequence(&model->projector_second_weight,first,frames,
                    AUDIO_PROJECTED,output,
                    f32(&model->projector_second_bias));
    free(normalized);free(first);return 0;
}

static void append_chunk_cache(float *cache,const float *current,
                               uint32_t *cached,uint32_t append)
{
    _Static_assert(AUDIO_CHUNK_MAX<AUDIO_CACHE_FRAMES,
                   "one streaming chunk must fit in the K/V cache");
    uint32_t keep=*cached;
    if(keep+append>AUDIO_CACHE_FRAMES){
        uint32_t drop=keep+append-AUDIO_CACHE_FRAMES;
        keep-=drop;
        memmove(cache,cache+(size_t)drop*AUDIO_HIDDEN,
                (size_t)keep*AUDIO_HIDDEN*sizeof(float));
    }
    memcpy(cache+(size_t)keep*AUDIO_HIDDEN,current,
           (size_t)append*AUDIO_HIDDEN*sizeof(float));
    *cached=keep+append;
}

static int encoder_forward_chunk(minimindo_audio_encoder_stream *stream,
                                 const float *features,uint32_t position,
                                 uint32_t length,uint32_t right_context,
                                 float *output,double *projector_seconds)
{
    minimindo_audio_encoder *model=stream->model;
    const uint32_t h=AUDIO_HIDDEN;
    float *sequence=malloc((size_t)length*560*sizeof(float));
    float *norm=malloc((size_t)length*560*sizeof(float));
    float *q=malloc((size_t)length*h*sizeof(float));
    float *k=malloc((size_t)length*h*sizeof(float));
    float *v=malloc((size_t)length*h*sizeof(float));
    float *memory=malloc((size_t)length*h*sizeof(float));
    float *attention=malloc((size_t)length*h*sizeof(float));
    float *projected=malloc((size_t)length*h*sizeof(float));
    float *ff=malloc((size_t)length*2048*sizeof(float));
    int8_t *quantized=malloc((size_t)length*2048);
    float *scales=malloc((size_t)length*sizeof(float));
    if(!sequence||!norm||!q||!k||!v||!memory||!attention||!projected||!ff||
       !quantized||!scales)
        goto failed;
    float root=sqrtf(512.0f);
    for(uint32_t p=0;p<length;++p){
        for(uint32_t i=0;i<560;++i)
            sequence[(size_t)p*560+i]=features[(size_t)p*560+i]*root;
        for(uint32_t i=0;i<280;++i){
            double angle=(position+p+1U)*pow(10000.0,-2.0*i/560.0);
            sequence[(size_t)p*560+i]+=sinf((float)angle);
            sequence[(size_t)p*560+i+280]+=cosf((float)angle);
        }
    }
    for(uint32_t li=0;li<AUDIO_LAYERS;++li){
        encoder_layer *layer=&model->layers[li];
        uint32_t in=li?h:560;
        layer_norm_sequence(sequence,norm,length,in,
                            f32(&layer->norm1_weight),f32(&layer->norm1_bias),
                            model->header->norm_epsilon);
        quantize_sequence_i8(norm,length,in,quantized,scales);
        matrix_sequence_i8_quantized(&layer->q_weight,quantized,scales,length,
                                     in,q,f32(&layer->q_bias));
        matrix_sequence_i8_quantized(&layer->k_weight,quantized,scales,length,
                                     in,k,f32(&layer->k_bias));
        matrix_sequence_i8_quantized(&layer->v_weight,quantized,scales,length,
                                     in,v,f32(&layer->v_bias));
        fsmn_chunk_context fsmn={v,f32(&layer->fsmn),memory,length};
        minimindo_parallel_for((size_t)length*h,fsmn_chunk_cells,&fsmn);
        float *cached_k=stream->cache_keys+
            (size_t)li*AUDIO_CACHE_FRAMES*h;
        float *cached_v=stream->cache_values+
            (size_t)li*AUDIO_CACHE_FRAMES*h;
        attention_chunk_context attention_context={
            q,k,v,cached_k,cached_v,attention,length,stream->cache_count[li]};
        minimindo_parallel_for((size_t)length*4,
                               attention_chunk_heads,&attention_context);
        uint32_t append=length-right_context;
        uint32_t key_count=stream->cache_count[li];
        uint32_t value_count=stream->cache_count[li];
        append_chunk_cache(cached_k,k,&key_count,append);
        append_chunk_cache(cached_v,v,&value_count,append);
        if(value_count!=key_count)goto failed;
        stream->cache_count[li]=key_count;
        quantize_sequence_i8(attention,length,h,quantized,scales);
        matrix_sequence_i8_quantized(&layer->out_weight,quantized,scales,length,
                                     h,projected,f32(&layer->out_bias));
        for(size_t i=0;i<(size_t)length*h;++i)
            projected[i]+=memory[i]+(li?sequence[i]:0.0f);
        layer_norm_sequence(projected,norm,length,h,
                            f32(&layer->norm2_weight),f32(&layer->norm2_bias),
                            model->header->norm_epsilon);
        quantize_sequence_i8(norm,length,h,quantized,scales);
        matrix_sequence_i8_quantized(&layer->fc1_weight,quantized,scales,length,
                                     h,ff,f32(&layer->fc1_bias));
        for(size_t i=0;i<(size_t)length*2048;++i)if(ff[i]<0.0f)ff[i]=0.0f;
        quantize_sequence_i8(ff,length,2048,quantized,scales);
        matrix_sequence_i8_quantized(&layer->fc2_weight,quantized,scales,length,
                                     2048,attention,f32(&layer->fc2_bias));
        for(size_t i=0;i<(size_t)length*h;++i)attention[i]+=projected[i];
        if(li==49)
            layer_norm_sequence(attention,projected,length,h,
                                f32(&model->after_weight),f32(&model->after_bias),
                                model->header->norm_epsilon);
        else memcpy(projected,attention,(size_t)length*h*sizeof(float));
        memcpy(sequence,projected,(size_t)length*h*sizeof(float));
    }
    layer_norm_sequence(sequence,projected,length,h,f32(&model->tp_weight),
                        f32(&model->tp_bias),model->header->norm_epsilon);
    double projector_start=monotonic_seconds();
    if(project_embeddings(model,projected,length-right_context,output))
        goto failed;
    if(projector_seconds)*projector_seconds+=
        monotonic_seconds()-projector_start;
    free(sequence);free(norm);free(q);free(k);free(v);free(memory);
    free(attention);free(projected);free(ff);free(quantized);free(scales);
    return 0;
failed:
    free(sequence);free(norm);free(q);free(k);free(v);free(memory);
    free(attention);free(projected);free(ff);free(quantized);free(scales);
    return -1;
}

minimindo_audio_encoder_stream *minimindo_audio_encoder_stream_open(
    minimindo_audio_encoder *model,char *error,size_t capacity)
{
    if(!model){set_error(error,capacity,"invalid streaming audio encoder");return NULL;}
    minimindo_audio_encoder_stream *stream=calloc(1,sizeof(*stream));
    if(!stream)return NULL;
    size_t cache_values=(size_t)AUDIO_LAYERS*AUDIO_CACHE_FRAMES*AUDIO_HIDDEN;
    stream->cache_keys=calloc(cache_values,sizeof(float));
    stream->cache_values=calloc(cache_values,sizeof(float));
    if(!stream->cache_keys||!stream->cache_values){
        minimindo_audio_encoder_stream_close(stream);return NULL;
    }
    stream->model=model;
    return stream;
}

void minimindo_audio_encoder_stream_close(minimindo_audio_encoder_stream *stream)
{
    if(!stream)return;
    free(stream->pcm);free(stream->cache_keys);free(stream->cache_values);
    free(stream);
}

int minimindo_audio_encoder_stream_push_pcm16(
    minimindo_audio_encoder_stream *stream,const int16_t *samples,
    size_t sample_count,int end_of_stream,float *output,size_t output_count,
    size_t *output_frames,char *error,size_t capacity)
{
    if(output_frames)*output_frames=0;
    if(!stream||stream->ended||(!samples&&sample_count)||(!output&&output_count)){
        set_error(error,capacity,"invalid streaming audio arguments");return -1;
    }
    if(sample_count){
        if(stream->pcm_count+sample_count<stream->pcm_count){
            set_error(error,capacity,"streaming audio is too large");return -1;
        }
        size_t needed=stream->pcm_count+sample_count;
        if(needed>stream->pcm_capacity){
            size_t grown=stream->pcm_capacity?stream->pcm_capacity:8192;
            while(grown<needed)grown*=2;
            int16_t *pcm=realloc(stream->pcm,grown*sizeof(int16_t));
            if(!pcm)return -1;
            stream->pcm=pcm;stream->pcm_capacity=grown;
        }
        memcpy(stream->pcm+stream->pcm_count,samples,
               sample_count*sizeof(int16_t));
        stream->pcm_count+=sample_count;
    }
    if(stream->pcm_count<400){
        if(end_of_stream){set_error(error,capacity,"streaming audio is too short");return -1;}
        return 0;
    }
    uint32_t mel_frames=1U+(uint32_t)((stream->pcm_count-400)/160);
    uint32_t stable_frames=mel_frames>=4U?(mel_frames+2U)/6U:0U;
    if(!end_of_stream&&
       stable_frames<stream->emitted_frames+AUDIO_CHUNK_MAX)return 0;
    double stage=monotonic_seconds();
    uint32_t feature_frames=0;
    float *features=frontend(stream->model,stream->pcm,stream->pcm_count,
                             &feature_frames);
    stream->profile.frontend_ms+=(monotonic_seconds()-stage)*1000.0;
    if(!features){set_error(error,capacity,"streaming frontend failed");return -1;}
    uint32_t available=end_of_stream?feature_frames:
        stable_frames;
    size_t produced=0;
    while(available>stream->emitted_frames){
        uint32_t remaining=available-stream->emitted_frames;
        uint32_t length,right,commit;
        if(remaining>AUDIO_CHUNK_MAX||(!end_of_stream&&remaining>=AUDIO_CHUNK_MAX)){
            length=AUDIO_CHUNK_MAX;right=AUDIO_CHUNK_RIGHT;
            commit=AUDIO_CHUNK_CENTER;
        }else if(end_of_stream){
            length=remaining;right=0;commit=remaining;
        }else break;
        if(produced+(size_t)commit*AUDIO_PROJECTED>output_count){
            free(features);set_error(error,capacity,"streaming output buffer too small");
            return -1;
        }
        stage=monotonic_seconds();
        double projector_seconds=0.0;
        if(encoder_forward_chunk(stream,
                features+(size_t)stream->emitted_frames*560,
                stream->emitted_frames,length,right,output+produced,
                &projector_seconds)){
            free(features);set_error(error,capacity,"streaming SenseVoice encoder failed");
            return -1;
        }
        double chunk_ms=(monotonic_seconds()-stage)*1000.0;
        stream->profile.projector_ms+=projector_seconds*1000.0;
        stream->profile.encoder_ms+=chunk_ms-projector_seconds*1000.0;
        stream->emitted_frames+=commit;
        produced+=(size_t)commit*AUDIO_PROJECTED;
    }
    free(features);
    if(end_of_stream)stream->ended=1;
    stream->model->profile=stream->profile;
    if(output_frames)*output_frames=produced/AUDIO_PROJECTED;
    return 0;
}

size_t minimindo_audio_encoder_stream_total_frames(
    const minimindo_audio_encoder_stream *stream)
{return stream?stream->emitted_frames:0;}

void minimindo_audio_encoder_stream_profile(
    const minimindo_audio_encoder_stream *stream,
    minimindo_audio_encoder_profile *profile)
{if(profile)*profile=stream?stream->profile:(minimindo_audio_encoder_profile){0};}

int minimindo_audio_encoder_encode_pcm16(minimindo_audio_encoder *model,const int16_t *samples,size_t sample_count,
                                         float *output,size_t output_count,size_t *output_frames,char *error,size_t capacity)
{
    if(!model||!samples||!output){set_error(error,capacity,"invalid audio encode arguments");return -1;}model->profile=(minimindo_audio_encoder_profile){0};double stage=monotonic_seconds();uint32_t frames=0;float *features=frontend(model,samples,sample_count,&frames);model->profile.frontend_ms=(monotonic_seconds()-stage)*1000.0;
    if(!features||output_count<(size_t)frames*768){free(features);set_error(error,capacity,"audio buffer too small");return -1;}
    /* MiniMind-O freezes and calls SenseVoice's encoder submodule directly.
       Language/task embeddings belong to the ASR wrapper and are not part of
       this embedding path. */
    uint32_t length=frames;float *sequence=malloc((size_t)length*560*4);if(!sequence){free(features);return -1;}float root=sqrtf(512);
    for(uint32_t p=0;p<frames;++p)
        for(uint32_t i=0;i<560;++i)
            sequence[(size_t)p*560+i]=features[(size_t)p*560+i]*root;
    free(features);
    for(uint32_t p=0;p<length;++p)for(uint32_t i=0;i<280;++i){sequence[(size_t)p*560+i]+=sinf((p+1)*pow(10000.0,-2.0*i/560));sequence[(size_t)p*560+i+280]+=cosf((p+1)*pow(10000.0,-2.0*i/560));}
    stage=monotonic_seconds();if(encoder_forward(model,sequence,length)){free(sequence);set_error(error,capacity,"SenseVoice encoder failed");return -1;}model->profile.encoder_ms=(monotonic_seconds()-stage)*1000.0;stage=monotonic_seconds();
    if(project_embeddings(model,sequence,frames,output)){
        free(sequence);set_error(error,capacity,"audio projector failed");return -1;
    }
    model->profile.projector_ms=(monotonic_seconds()-stage)*1000.0;free(sequence);if(output_frames)*output_frames=frames;return 0;
}

void minimindo_audio_encoder_last_profile(const minimindo_audio_encoder *model,
                                          minimindo_audio_encoder_profile *profile)
{if(profile)*profile=model?model->profile:(minimindo_audio_encoder_profile){0};}
