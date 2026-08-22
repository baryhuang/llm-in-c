#define _POSIX_C_SOURCE 200809L

#include "minimindo_thinker.h"
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
#include <unistd.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

enum { MMO_VERSION = 1, MMO_HEADER_BYTES = 4096, MMO_F32 = 1, MMO_Q8_ROW = 2 };

#ifndef MINIMINDO_GENERATION_W8A8
#define MINIMINDO_GENERATION_W8A8 0
#endif

typedef struct {
    unsigned char magic[8];
    uint32_t version, header_bytes, layers, hidden, heads, kv_heads;
    uint32_t head_dim, intermediate, vocab, tensor_count;
    float rms_epsilon, rope_theta;
    uint64_t tensors_offset, file_bytes;
    char source_sha256[64];
} mmo_header;

typedef struct {
    uint32_t type, rows, cols, reserved;
    uint64_t data_bytes;
} mmo_tensor_header;

typedef struct {
    const mmo_tensor_header *header;
    const unsigned char *data;
} mmo_tensor;

typedef struct {
    mmo_tensor input_norm, q_norm, k_norm, post_norm;
    mmo_tensor q_proj, k_proj, v_proj, o_proj;
    mmo_tensor gate_proj, up_proj, down_proj;
} mmo_layer;

struct minimindo_thinker {
    int file;
    const unsigned char *mapping;
    size_t mapped_bytes;
    const mmo_header *header;
    mmo_tensor embedding, final_norm;
    mmo_layer *layers;
    uint32_t max_context, position;
    float *key_cache, *value_cache;
    float *hidden, *normed, *query, *key, *value;
    float *attention, *projected, *mlp_normed, *gate, *up;
    double *scores;
    double *rope_cos, *rope_sin;
};

_Static_assert(sizeof(mmo_header) == 136, "MiniMind-O image header ABI");
_Static_assert(sizeof(mmo_tensor_header) == 24, "MiniMind-O tensor ABI");

static void set_error(char *error, size_t capacity, const char *format, ...)
{
    if (error == NULL || capacity == 0) return;
    va_list args;
    va_start(args, format);
    vsnprintf(error, capacity, format, args);
    va_end(args);
}

static uint64_t align64(uint64_t value)
{
    return (value + 63U) & ~UINT64_C(63);
}

static int range_ok(uint64_t offset, uint64_t length, uint64_t total)
{
    return offset <= total && length <= total - offset;
}

static int take_tensor(const unsigned char *mapping, uint64_t file_bytes,
                       uint64_t *cursor, uint32_t type, uint32_t rows,
                       uint32_t cols, mmo_tensor *tensor)
{
    if (!range_ok(*cursor, sizeof(mmo_tensor_header), file_bytes)) return -1;
    const mmo_tensor_header *header =
        (const mmo_tensor_header *)(mapping + *cursor);
    const uint64_t expected = type == MMO_F32
        ? (uint64_t)rows * cols * sizeof(float)
        : (uint64_t)rows * (sizeof(float) + cols);
    if (header->type != type || header->rows != rows ||
        header->cols != cols || header->reserved != 0 ||
        header->data_bytes != expected) return -1;
    const uint64_t data_offset = *cursor + sizeof(*header);
    if (!range_ok(data_offset, expected, file_bytes)) return -1;
    tensor->header = header;
    tensor->data = mapping + data_offset;
    *cursor = align64(data_offset + expected);
    return *cursor <= file_bytes ? 0 : -1;
}

static const float *f32_data(const mmo_tensor *tensor)
{
    return (const float *)tensor->data;
}

static void q8_row_to_float(const mmo_tensor *tensor, uint32_t row,
                            float *output)
{
    const size_t stride = sizeof(float) + tensor->header->cols;
    const unsigned char *record = tensor->data + (size_t)row * stride;
    float scale;
    memcpy(&scale, record, sizeof(scale));
    const int8_t *values = (const int8_t *)(record + sizeof(scale));
    for (uint32_t index = 0; index < tensor->header->cols; ++index)
        output[index] = scale * values[index];
}

static float q8_f32_dot(const int8_t *weights, const float *input,
                        uint32_t count)
{
#if defined(__aarch64__)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16) {
        const int8x16_t packed = vld1q_s8(weights + index);
        const int16x8_t low = vmovl_s8(vget_low_s8(packed));
        const int16x8_t high = vmovl_s8(vget_high_s8(packed));
        const float32x4_t w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(low)));
        const float32x4_t w1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(low)));
        const float32x4_t w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(high)));
        const float32x4_t w3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(high)));
        sum0 = vfmaq_f32(sum0, vld1q_f32(input + index), w0);
        sum1 = vfmaq_f32(sum1, vld1q_f32(input + index + 4), w1);
        sum2 = vfmaq_f32(sum2, vld1q_f32(input + index + 8), w2);
        sum3 = vfmaq_f32(sum3, vld1q_f32(input + index + 12), w3);
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1),
                                     vaddq_f32(sum2, sum3)));
    for (; index < count; ++index) sum += input[index] * weights[index];
    return sum;
#else
    double sum = 0.0;
    for (uint32_t index = 0; index < count; ++index)
        sum += (double)input[index] * weights[index];
    return (float)sum;
#endif
}

#if MINIMINDO_GENERATION_W8A8
static int32_t q8_i8_dot(const int8_t *weights,const int8_t *input,
                         uint32_t count)
{
#if defined(__aarch64__)
    int32x4_t sum0=vdupq_n_s32(0),sum1=sum0,sum2=sum0,sum3=sum0;
    uint32_t index=0;
    for(;index+64U<=count;index+=64U){
        const int8x16_t w0=vld1q_s8(weights+index);
        const int8x16_t x0=vld1q_s8(input+index);
        const int8x16_t w1=vld1q_s8(weights+index+16U);
        const int8x16_t x1=vld1q_s8(input+index+16U);
        const int8x16_t w2=vld1q_s8(weights+index+32U);
        const int8x16_t x2=vld1q_s8(input+index+32U);
        const int8x16_t w3=vld1q_s8(weights+index+48U);
        const int8x16_t x3=vld1q_s8(input+index+48U);
        int16x8_t pair0=vmull_s8(vget_low_s8(w0),vget_low_s8(x0));
        pair0=vmlal_s8(pair0,vget_low_s8(w1),vget_low_s8(x1));
        int16x8_t pair1=vmull_s8(vget_high_s8(w0),vget_high_s8(x0));
        pair1=vmlal_s8(pair1,vget_high_s8(w1),vget_high_s8(x1));
        int16x8_t pair2=vmull_s8(vget_low_s8(w2),vget_low_s8(x2));
        pair2=vmlal_s8(pair2,vget_low_s8(w3),vget_low_s8(x3));
        int16x8_t pair3=vmull_s8(vget_high_s8(w2),vget_high_s8(x2));
        pair3=vmlal_s8(pair3,vget_high_s8(w3),vget_high_s8(x3));
        sum0=vpadalq_s16(sum0,pair0);sum1=vpadalq_s16(sum1,pair1);
        sum2=vpadalq_s16(sum2,pair2);sum3=vpadalq_s16(sum3,pair3);
    }
    int32_t sum=vaddvq_s32(vaddq_s32(vaddq_s32(sum0,sum1),
                                      vaddq_s32(sum2,sum3)));
    for(;index<count;++index)sum+=weights[index]*input[index];
    return sum;
#else
    int32_t sum=0;for(uint32_t index=0;index<count;++index)
        sum+=weights[index]*input[index];return sum;
#endif
}

static float quantize_i8(const float *input,int8_t *output,uint32_t count)
{
    float maximum=0.0f;
#if defined(__aarch64__)
    float32x4_t vmax0=vdupq_n_f32(0),vmax1=vmax0,vmax2=vmax0,vmax3=vmax0;
    uint32_t index=0;
    for(;index+16U<=count;index+=16U){
        vmax0=vmaxq_f32(vmax0,vabsq_f32(vld1q_f32(input+index)));
        vmax1=vmaxq_f32(vmax1,vabsq_f32(vld1q_f32(input+index+4U)));
        vmax2=vmaxq_f32(vmax2,vabsq_f32(vld1q_f32(input+index+8U)));
        vmax3=vmaxq_f32(vmax3,vabsq_f32(vld1q_f32(input+index+12U)));
    }
    maximum=vmaxvq_f32(vmaxq_f32(vmaxq_f32(vmax0,vmax1),
                                  vmaxq_f32(vmax2,vmax3)));
    for(;index<count;++index)if(fabsf(input[index])>maximum)
        maximum=fabsf(input[index]);
#else
    for(uint32_t index=0;index<count;++index)if(fabsf(input[index])>maximum)
        maximum=fabsf(input[index]);
#endif
    if(!(maximum>0.0f)){memset(output,0,count);return 0.0f;}
    const float scale=maximum/127.0f,inverse=1.0f/scale;
#if defined(__aarch64__)
    const float32x4_t vinverse=vdupq_n_f32(inverse);uint32_t output_index=0;
    for(;output_index+16U<=count;output_index+=16U){
        const int32x4_t q0=vcvtnq_s32_f32(vmulq_f32(
            vld1q_f32(input+output_index),vinverse));
        const int32x4_t q1=vcvtnq_s32_f32(vmulq_f32(
            vld1q_f32(input+output_index+4U),vinverse));
        const int32x4_t q2=vcvtnq_s32_f32(vmulq_f32(
            vld1q_f32(input+output_index+8U),vinverse));
        const int32x4_t q3=vcvtnq_s32_f32(vmulq_f32(
            vld1q_f32(input+output_index+12U),vinverse));
        const int16x8_t lo=vcombine_s16(vqmovn_s32(q0),vqmovn_s32(q1));
        const int16x8_t hi=vcombine_s16(vqmovn_s32(q2),vqmovn_s32(q3));
        vst1q_s8(output+output_index,
                 vcombine_s8(vqmovn_s16(lo),vqmovn_s16(hi)));
    }
    for(;output_index<count;++output_index)
        output[output_index]=(int8_t)lrintf(input[output_index]*inverse);
#else
    for(uint32_t index=0;index<count;++index)
        output[index]=(int8_t)lrintf(input[index]*inverse);
#endif
    return scale;
}
#endif

typedef struct {
    const mmo_tensor *tensor;
    const float *input;
    float *output;
    uint32_t length;
    const int8_t *quantized;
    float input_scale;
} q8_parallel_context;

static void q8_matvec_rows(void *opaque, size_t begin, size_t end)
{
    q8_parallel_context *context = opaque;
    const mmo_tensor *tensor = context->tensor;
#if MINIMINDO_GENERATION_W8A8
    const size_t stride=sizeof(float)+tensor->header->cols;
    for(size_t row=begin;row<end;++row){
        const unsigned char *record=tensor->data+row*stride;float scale;
        memcpy(&scale,record,sizeof(scale));
        context->output[row]=(float)q8_i8_dot(
            (const int8_t *)(record+sizeof(float)),context->quantized,
            tensor->header->cols)*scale*context->input_scale;
    }
#else
    const float *input = context->input;
    float *output = context->output;
    const size_t stride = sizeof(float) + tensor->header->cols;
    for (size_t row = begin; row < end; ++row) {
        const unsigned char *record = tensor->data + (size_t)row * stride;
        float scale;
        memcpy(&scale, record, sizeof(scale));
        const int8_t *values = (const int8_t *)(record + sizeof(scale));
        output[row] = q8_f32_dot(values, input, tensor->header->cols) * scale;
    }
#endif
}

static void q8_matvec(const mmo_tensor *tensor, const float *input,
                      float *output)
{
#if MINIMINDO_GENERATION_W8A8
    int8_t quantized[tensor->header->cols];
    const float input_scale=quantize_i8(input,quantized,tensor->header->cols);
    q8_parallel_context context={tensor,input,output,1U,quantized,input_scale};
#else
    q8_parallel_context context = {tensor, input, output, 1U,NULL,0.0f};
#endif
    minimindo_parallel_for(tensor->header->rows,q8_matvec_rows,&context);
}

static void q8_matmul_rows(void *opaque, size_t begin, size_t end)
{
    q8_parallel_context *context = opaque;
    const mmo_tensor *tensor = context->tensor;
    const uint32_t input_width = tensor->header->cols;
    const uint32_t output_width = tensor->header->rows;
    const size_t weight_stride = sizeof(float) + input_width;
    for (size_t row = begin; row < end; ++row) {
        const unsigned char *record = tensor->data + row * weight_stride;
        float scale;
        memcpy(&scale, record, sizeof(scale));
        const int8_t *weights = (const int8_t *)(record + sizeof(scale));
        for (uint32_t position = 0; position < context->length; ++position)
            context->output[(size_t)position * output_width + row] =
                q8_f32_dot(weights,
                           context->input + (size_t)position * input_width,
                           input_width) * scale;
    }
}

static void q8_matmul_sequence(const mmo_tensor *tensor, const float *input,
                               uint32_t length, float *output)
{
    if (length == 1U) {
        q8_matvec(tensor,input,output);
        return;
    }
    q8_parallel_context context = {tensor,input,output,length,NULL,0.0f};
    minimindo_parallel_for(tensor->header->rows, q8_matmul_rows, &context);
}

static void rms_norm(const float *input, const float *weight, float *output,
                     uint32_t width, float epsilon)
{
    double squares = 0.0;
    for (uint32_t index = 0; index < width; ++index)
        squares += (double)input[index] * input[index];
    const double scale = 1.0 / sqrt(squares / width + epsilon);
    for (uint32_t index = 0; index < width; ++index)
        output[index] = (float)((double)input[index] * scale * weight[index]);
}

typedef struct {
    const float *input;
    const float *weight;
    float *output;
    uint32_t width;
    float epsilon;
} norm_parallel_context;

static void rms_norm_positions(void *opaque, size_t begin, size_t end)
{
    norm_parallel_context *context = opaque;
    for (size_t position = begin; position < end; ++position)
        rms_norm(context->input + position * context->width,
                 context->weight,
                 context->output + position * context->width,
                 context->width, context->epsilon);
}

static void rms_norm_sequence(const float *input, const float *weight,
                              float *output, uint32_t length,
                              uint32_t width, float epsilon)
{
    norm_parallel_context context = {input, weight, output, width, epsilon};
    minimindo_parallel_for(length, rms_norm_positions, &context);
}

static void apply_rope(float *states, uint32_t heads, uint32_t head_dim,
                       const double *cosines, const double *sines)
{
    const uint32_t half = head_dim / 2U;
    for (uint32_t head = 0; head < heads; ++head) {
        float *vector = states + (size_t)head * head_dim;
        for (uint32_t index = 0; index < half; ++index) {
            const double cosine = cosines[index], sine = sines[index];
            const double first = vector[index], second = vector[index + half];
            vector[index] = (float)(first * cosine - second * sine);
            vector[index + half] = (float)(second * cosine + first * sine);
        }
    }
}

static double silu(double value)
{
    return value / (1.0 + exp(-value));
}

static int allocate_buffers(minimindo_thinker *model)
{
    const size_t hidden = model->header->hidden;
    const size_t kv = (size_t)model->header->kv_heads * model->header->head_dim;
    const size_t intermediate = model->header->intermediate;
    const size_t cache = (size_t)model->header->layers * model->max_context * kv;
#define ALLOC(field, count) do { model->field = calloc((count), sizeof(*model->field)); if (model->field == NULL) return -1; } while (0)
    ALLOC(key_cache, cache); ALLOC(value_cache, cache);
    ALLOC(hidden, hidden); ALLOC(normed, hidden); ALLOC(query, hidden);
    ALLOC(key, kv); ALLOC(value, kv); ALLOC(attention, hidden);
    ALLOC(projected, hidden); ALLOC(mlp_normed, hidden);
    ALLOC(gate, intermediate); ALLOC(up, intermediate);
    ALLOC(scores, model->max_context);
    ALLOC(rope_cos, model->max_context * model->header->head_dim / 2U);
    ALLOC(rope_sin, model->max_context * model->header->head_dim / 2U);
#undef ALLOC
    const uint32_t half = model->header->head_dim / 2U;
    for (uint32_t position = 0; position < model->max_context; ++position)
        for (uint32_t index = 0; index < half; ++index) {
            const double angle = position /
                pow((double)model->header->rope_theta,
                    (2.0 * index) / model->header->head_dim);
            model->rope_cos[(size_t)position * half + index] = cos(angle);
            model->rope_sin[(size_t)position * half + index] = sin(angle);
        }
    return 0;
}

minimindo_thinker *minimindo_thinker_open(const char *image_path,
                                          uint32_t max_context,
                                          char *error, size_t error_capacity)
{
    if (image_path == NULL || max_context == 0) {
        set_error(error, error_capacity, "invalid image path or context");
        return NULL;
    }
    minimindo_thinker *model = calloc(1, sizeof(*model));
    if (model == NULL) return NULL;
    model->file = -1;
    model->file = open(image_path, O_RDONLY);
    struct stat status;
    if (model->file < 0 || fstat(model->file, &status) != 0 ||
        status.st_size < MMO_HEADER_BYTES) {
        set_error(error, error_capacity, "cannot open image: %s", strerror(errno));
        minimindo_thinker_close(model); return NULL;
    }
    model->mapped_bytes = (size_t)status.st_size;
    model->mapping = mmap(NULL, model->mapped_bytes, PROT_READ, MAP_PRIVATE,
                          model->file, 0);
    if (model->mapping == MAP_FAILED) {
        model->mapping = NULL;
        set_error(error, error_capacity, "cannot mmap image: %s", strerror(errno));
        minimindo_thinker_close(model); return NULL;
    }
    model->header = (const mmo_header *)model->mapping;
    static const unsigned char magic[8] = {'M','M','O','T','S','K','1','\0'};
    if (memcmp(model->header->magic, magic, 8) != 0 ||
        model->header->version != MMO_VERSION ||
        model->header->header_bytes != MMO_HEADER_BYTES ||
        model->header->file_bytes != model->mapped_bytes ||
        model->header->tensors_offset != MMO_HEADER_BYTES ||
        model->header->hidden != model->header->heads * model->header->head_dim ||
        model->header->heads % model->header->kv_heads != 0 ||
        model->header->tensor_count != 1U + model->header->layers * 11U + 1U) {
        set_error(error, error_capacity, "invalid MiniMind-O image header");
        minimindo_thinker_close(model); return NULL;
    }
    model->layers = calloc(model->header->layers, sizeof(*model->layers));
    if (model->layers == NULL) { minimindo_thinker_close(model); return NULL; }
    uint64_t cursor = model->header->tensors_offset;
    const uint32_t h = model->header->hidden, d = model->header->head_dim;
    const uint32_t kv = model->header->kv_heads * d;
    const uint32_t m = model->header->intermediate;
    if (take_tensor(model->mapping, model->mapped_bytes, &cursor, MMO_Q8_ROW,
                    model->header->vocab, h, &model->embedding) != 0) goto bad_tensors;
    for (uint32_t index = 0; index < model->header->layers; ++index) {
        mmo_layer *layer = &model->layers[index];
#define TAKE(member, type, rows, cols) if (take_tensor(model->mapping, model->mapped_bytes, &cursor, type, rows, cols, &layer->member) != 0) goto bad_tensors
        TAKE(input_norm, MMO_F32, 1, h); TAKE(q_norm, MMO_F32, 1, d);
        TAKE(k_norm, MMO_F32, 1, d); TAKE(post_norm, MMO_F32, 1, h);
        TAKE(q_proj, MMO_Q8_ROW, h, h); TAKE(k_proj, MMO_Q8_ROW, kv, h);
        TAKE(v_proj, MMO_Q8_ROW, kv, h); TAKE(o_proj, MMO_Q8_ROW, h, h);
        TAKE(gate_proj, MMO_Q8_ROW, m, h); TAKE(up_proj, MMO_Q8_ROW, m, h);
        TAKE(down_proj, MMO_Q8_ROW, h, m);
#undef TAKE
    }
    if (take_tensor(model->mapping, model->mapped_bytes, &cursor, MMO_F32, 1, h,
                    &model->final_norm) != 0 || cursor != model->mapped_bytes)
        goto bad_tensors;
    model->max_context = max_context;
    if (allocate_buffers(model) != 0) {
        set_error(error, error_capacity, "cannot allocate runtime buffers");
        minimindo_thinker_close(model); return NULL;
    }
    return model;

bad_tensors:
    set_error(error, error_capacity, "invalid MiniMind-O tensor sequence");
    minimindo_thinker_close(model);
    return NULL;
}

void minimindo_thinker_close(minimindo_thinker *model)
{
    if (model == NULL) return;
    free(model->layers); free(model->key_cache); free(model->value_cache);
    free(model->hidden); free(model->normed); free(model->query);
    free(model->key); free(model->value); free(model->attention);
    free(model->projected); free(model->mlp_normed); free(model->gate);
    free(model->up); free(model->scores);
    free(model->rope_cos); free(model->rope_sin);
    if (model->mapping != NULL) munmap((void *)model->mapping, model->mapped_bytes);
    if (model->file >= 0) close(model->file);
    free(model);
}

void minimindo_thinker_reset(minimindo_thinker *model)
{
    if (model == NULL) return;
    model->position = 0;
}

uint32_t minimindo_thinker_vocab_size(const minimindo_thinker *model)
{
    return model == NULL ? 0 : model->header->vocab;
}

uint32_t minimindo_thinker_position(const minimindo_thinker *model)
{
    return model == NULL ? 0 : model->position;
}

uint32_t minimindo_thinker_hidden_size(const minimindo_thinker *model)
{
    return model == NULL ? 0 : model->header->hidden;
}

static int forward_hidden(minimindo_thinker *model,
                          float *logits, size_t logits_count,
                          float *bridge_states, size_t bridge_count,
                          char *error, size_t error_capacity)
{
    if (model == NULL ||
        (logits != NULL && logits_count < model->header->vocab) ||
        (bridge_states != NULL && bridge_count < model->header->hidden)) {
        set_error(error, error_capacity, "invalid forward arguments"); return -1;
    }
    if (model->position >= model->max_context) {
        set_error(error, error_capacity, "context limit reached"); return -2;
    }
    const uint32_t h = model->header->hidden, d = model->header->head_dim;
    const uint32_t heads = model->header->heads;
    const uint32_t kv_heads = model->header->kv_heads;
    const uint32_t kv_size = kv_heads * d;
    const uint32_t groups = heads / kv_heads;
    const uint32_t pos = model->position;

    for (uint32_t layer_index = 0; layer_index < model->header->layers; ++layer_index) {
        const mmo_layer *layer = &model->layers[layer_index];
        rms_norm(model->hidden, f32_data(&layer->input_norm), model->normed,
                 h, model->header->rms_epsilon);
        q8_matvec(&layer->q_proj,model->normed,model->query);
        q8_matvec(&layer->k_proj,model->normed,model->key);
        q8_matvec(&layer->v_proj,model->normed,model->value);
        for (uint32_t head = 0; head < heads; ++head)
            rms_norm(model->query + (size_t)head * d, f32_data(&layer->q_norm),
                     model->query + (size_t)head * d, d,
                     model->header->rms_epsilon);
        for (uint32_t head = 0; head < kv_heads; ++head)
            rms_norm(model->key + (size_t)head * d, f32_data(&layer->k_norm),
                     model->key + (size_t)head * d, d,
                     model->header->rms_epsilon);
        const double *rope_cos=model->rope_cos+(size_t)pos*(d/2U);
        const double *rope_sin=model->rope_sin+(size_t)pos*(d/2U);
        apply_rope(model->query, heads, d, rope_cos, rope_sin);
        apply_rope(model->key, kv_heads, d, rope_cos, rope_sin);
        const size_t cache_base =
            ((size_t)layer_index * model->max_context + pos) * kv_size;
        memcpy(model->key_cache + cache_base, model->key, kv_size * sizeof(float));
        memcpy(model->value_cache + cache_base, model->value, kv_size * sizeof(float));

        for (uint32_t head = 0; head < heads; ++head) {
            const uint32_t kv_head = head / groups;
            const float *query = model->query + (size_t)head * d;
            double maximum = -DBL_MAX;
            for (uint32_t source = 0; source <= pos; ++source) {
                const size_t base =
                    ((size_t)layer_index * model->max_context + source) * kv_size +
                    (size_t)kv_head * d;
                double score = 0.0;
                for (uint32_t index = 0; index < d; ++index)
                    score += (double)query[index] * model->key_cache[base + index];
                model->scores[source] = score / sqrt((double)d);
                if (model->scores[source] > maximum) maximum = model->scores[source];
            }
            double denominator = 0.0;
            for (uint32_t source = 0; source <= pos; ++source) {
                model->scores[source] = exp(model->scores[source] - maximum);
                denominator += model->scores[source];
            }
            float *attention = model->attention + (size_t)head * d;
            for (uint32_t index = 0; index < d; ++index) {
                double sum = 0.0;
                for (uint32_t source = 0; source <= pos; ++source) {
                    const size_t base =
                        ((size_t)layer_index * model->max_context + source) * kv_size +
                        (size_t)kv_head * d;
                    sum += model->scores[source] * model->value_cache[base + index];
                }
                attention[index] = (float)(sum / denominator);
            }
        }
        q8_matvec(&layer->o_proj, model->attention, model->projected);
        for (uint32_t index = 0; index < h; ++index)
            model->hidden[index] += model->projected[index];
        rms_norm(model->hidden, f32_data(&layer->post_norm), model->mlp_normed,
                 h, model->header->rms_epsilon);
        q8_matvec(&layer->gate_proj,model->mlp_normed,model->gate);
        q8_matvec(&layer->up_proj,model->mlp_normed,model->up);
        for (uint32_t index = 0; index < model->header->intermediate; ++index)
            model->gate[index] = (float)(silu(model->gate[index]) * model->up[index]);
        q8_matvec(&layer->down_proj, model->gate, model->projected);
        for (uint32_t index = 0; index < h; ++index)
            model->hidden[index] += model->projected[index];
        if (layer_index == 3U && bridge_states != NULL)
            memcpy(bridge_states, model->hidden, h * sizeof(float));
    }
    if (logits != NULL) {
        rms_norm(model->hidden, f32_data(&model->final_norm), model->normed,
                 h, model->header->rms_epsilon);
        q8_matvec(&model->embedding, model->normed, logits);
    }
    model->position++;
    return 0;
}

int minimindo_thinker_forward_bridge(minimindo_thinker *model,
                                     uint32_t token_id,
                                     float *logits, size_t logits_count,
                                     float *bridge_states,
                                     size_t bridge_count,
                                     char *error, size_t error_capacity)
{
    if (model == NULL || token_id >= model->header->vocab) {
        set_error(error, error_capacity, "invalid token id"); return -1;
    }
    q8_row_to_float(&model->embedding, token_id, model->hidden);
    return forward_hidden(model, logits, logits_count, bridge_states,
                          bridge_count, error, error_capacity);
}

int minimindo_thinker_forward_embedding(minimindo_thinker *model,
                                        const float *embedding,
                                        size_t embedding_count,
                                        float *logits, size_t logits_count,
                                        float *bridge_states,
                                        size_t bridge_count,
                                        char *error, size_t error_capacity)
{
    if (model == NULL || embedding == NULL ||
        embedding_count < model->header->hidden) {
        set_error(error, error_capacity, "invalid input embedding"); return -1;
    }
    memcpy(model->hidden, embedding, model->header->hidden * sizeof(float));
    return forward_hidden(model, logits, logits_count, bridge_states,
                          bridge_count, error, error_capacity);
}

int minimindo_thinker_forward(minimindo_thinker *model, uint32_t token_id,
                              float *logits, size_t logits_count,
                              char *error, size_t error_capacity)
{
    return minimindo_thinker_forward_bridge(model, token_id, logits,
                                            logits_count, NULL, 0,
                                            error, error_capacity);
}

int minimindo_thinker_prefill_sequence(
    minimindo_thinker *model,
    const uint32_t *token_ids, size_t token_count,
    const float *replacement_embeddings, const uint8_t *replacement_mask,
    float *logits, size_t logits_count,
    float *bridge_states, size_t bridge_count,
    char *error, size_t error_capacity)
{
    if (model == NULL || token_ids == NULL || token_count == 0U ||
        token_count > UINT32_MAX ||
        model->position + token_count > model->max_context ||
        (logits != NULL && logits_count < model->header->vocab) ||
        bridge_states == NULL ||
        bridge_count < token_count * (size_t)model->header->hidden) {
        set_error(error, error_capacity, "invalid sequence prefill arguments");
        return -1;
    }
    const uint32_t length = (uint32_t)token_count;
    const uint32_t h = model->header->hidden;
    const uint32_t d = model->header->head_dim;
    const uint32_t heads = model->header->heads;
    const uint32_t kv_heads = model->header->kv_heads;
    const uint32_t kv_size = kv_heads * d;
    const uint32_t groups = heads / kv_heads;
    const uint32_t start_position = model->position;
    const size_t hidden_count = (size_t)length * h;
    const size_t kv_count = (size_t)length * kv_size;
    const size_t intermediate_count =
        (size_t)length * model->header->intermediate;
    float *hidden = malloc(hidden_count * sizeof(float));
    float *normed = malloc(hidden_count * sizeof(float));
    float *query = malloc(hidden_count * sizeof(float));
    float *key = malloc(kv_count * sizeof(float));
    float *value = malloc(kv_count * sizeof(float));
    float *attention = malloc(hidden_count * sizeof(float));
    float *projected = malloc(hidden_count * sizeof(float));
    float *gate = malloc(intermediate_count * sizeof(float));
    float *up = malloc(intermediate_count * sizeof(float));
    if (hidden == NULL || normed == NULL || query == NULL || key == NULL ||
        value == NULL || attention == NULL || projected == NULL ||
        gate == NULL || up == NULL) {
        set_error(error, error_capacity, "sequence prefill allocation failed");
        free(hidden); free(normed); free(query); free(key); free(value);
        free(attention); free(projected); free(gate); free(up);
        return -1;
    }
    for (uint32_t position = 0; position < length; ++position) {
        float *destination = hidden + (size_t)position * h;
        if (replacement_mask != NULL && replacement_mask[position] != 0U) {
            if (replacement_embeddings == NULL) {
                set_error(error, error_capacity,
                          "missing sequence replacement embedding");
                free(hidden); free(normed); free(query); free(key); free(value);
                free(attention); free(projected); free(gate); free(up);
                return -1;
            }
            memcpy(destination,
                   replacement_embeddings + (size_t)position * h,
                   h * sizeof(float));
        } else {
            if (token_ids[position] >= model->header->vocab) {
                set_error(error, error_capacity, "invalid sequence token id");
                free(hidden); free(normed); free(query); free(key); free(value);
                free(attention); free(projected); free(gate); free(up);
                return -1;
            }
            q8_row_to_float(&model->embedding, token_ids[position], destination);
        }
    }

    for (uint32_t layer_index = 0; layer_index < model->header->layers;
         ++layer_index) {
        const mmo_layer *layer = &model->layers[layer_index];
        rms_norm_sequence(hidden, f32_data(&layer->input_norm), normed,
                          length, h, model->header->rms_epsilon);
        q8_matmul_sequence(&layer->q_proj, normed, length, query);
        q8_matmul_sequence(&layer->k_proj, normed, length, key);
        q8_matmul_sequence(&layer->v_proj, normed, length, value);
        for (uint32_t position = 0; position < length; ++position) {
            float *position_query = query + (size_t)position * h;
            float *position_key = key + (size_t)position * kv_size;
            for (uint32_t head = 0; head < heads; ++head)
                rms_norm(position_query + (size_t)head * d,
                         f32_data(&layer->q_norm),
                         position_query + (size_t)head * d, d,
                         model->header->rms_epsilon);
            for (uint32_t head = 0; head < kv_heads; ++head)
                rms_norm(position_key + (size_t)head * d,
                         f32_data(&layer->k_norm),
                         position_key + (size_t)head * d, d,
                         model->header->rms_epsilon);
            const uint32_t absolute_position = start_position + position;
            const double *rope_cos = model->rope_cos +
                (size_t)absolute_position * (d / 2U);
            const double *rope_sin = model->rope_sin +
                (size_t)absolute_position * (d / 2U);
            apply_rope(position_query, heads, d, rope_cos, rope_sin);
            apply_rope(position_key, kv_heads, d, rope_cos, rope_sin);
            const size_t cache_base =
                ((size_t)layer_index * model->max_context + absolute_position) *
                kv_size;
            memcpy(model->key_cache + cache_base, position_key,
                   kv_size * sizeof(float));
            memcpy(model->value_cache + cache_base,
                   value + (size_t)position * kv_size,
                   kv_size * sizeof(float));
        }
        for (uint32_t position = 0; position < length; ++position) {
            const uint32_t absolute_position = start_position + position;
            for (uint32_t head = 0; head < heads; ++head) {
                const uint32_t kv_head = head / groups;
                const float *head_query =
                    query + (size_t)position * h + (size_t)head * d;
                double maximum = -DBL_MAX;
                for (uint32_t source = 0; source <= absolute_position; ++source) {
                    const size_t base =
                        ((size_t)layer_index * model->max_context + source) *
                        kv_size + (size_t)kv_head * d;
                    double score = 0.0;
                    for (uint32_t index = 0; index < d; ++index)
                        score += (double)head_query[index] *
                                 model->key_cache[base + index];
                    model->scores[source] = score / sqrt((double)d);
                    if (model->scores[source] > maximum)
                        maximum = model->scores[source];
                }
                double denominator = 0.0;
                for (uint32_t source = 0; source <= absolute_position; ++source) {
                    model->scores[source] =
                        exp(model->scores[source] - maximum);
                    denominator += model->scores[source];
                }
                float *head_attention = attention + (size_t)position * h +
                                        (size_t)head * d;
                for (uint32_t index = 0; index < d; ++index) {
                    double sum = 0.0;
                    for (uint32_t source = 0; source <= absolute_position;
                         ++source) {
                        const size_t base =
                            ((size_t)layer_index * model->max_context + source) *
                            kv_size + (size_t)kv_head * d;
                        sum += model->scores[source] *
                               model->value_cache[base + index];
                    }
                    head_attention[index] = (float)(sum / denominator);
                }
            }
        }
        q8_matmul_sequence(&layer->o_proj, attention, length, projected);
        for (size_t index = 0; index < hidden_count; ++index)
            hidden[index] += projected[index];
        rms_norm_sequence(hidden, f32_data(&layer->post_norm), query,
                          length, h, model->header->rms_epsilon);
        q8_matmul_sequence(&layer->gate_proj, query, length, gate);
        q8_matmul_sequence(&layer->up_proj, query, length, up);
        for (size_t index = 0; index < intermediate_count; ++index)
            gate[index] = (float)(silu(gate[index]) * up[index]);
        q8_matmul_sequence(&layer->down_proj, gate, length, projected);
        for (size_t index = 0; index < hidden_count; ++index)
            hidden[index] += projected[index];
        if (layer_index == 3U)
            memcpy(bridge_states, hidden, hidden_count * sizeof(float));
    }
    const float *last_hidden = hidden + (size_t)(length - 1U) * h;
    if (logits != NULL) {
        rms_norm(last_hidden, f32_data(&model->final_norm), model->normed,
                 h, model->header->rms_epsilon);
        q8_matvec(&model->embedding, model->normed, logits);
    }
    memcpy(model->hidden, last_hidden, h * sizeof(float));
    model->position += length;
    free(hidden); free(normed); free(query); free(key); free(value);
    free(attention); free(projected); free(gate); free(up);
    return 0;
}

int minimindo_thinker_prefill_bridge(minimindo_thinker *model,
                                     uint32_t token_id,
                                     float *bridge_states,
                                     size_t bridge_count,
                                     char *error, size_t error_capacity)
{
    if (model == NULL || token_id >= model->header->vocab) {
        set_error(error, error_capacity, "invalid token id"); return -1;
    }
    q8_row_to_float(&model->embedding, token_id, model->hidden);
    return forward_hidden(model, NULL, 0, bridge_states, bridge_count,
                          error, error_capacity);
}

int minimindo_thinker_prefill_embedding(minimindo_thinker *model,
                                        const float *embedding,
                                        size_t embedding_count,
                                        float *bridge_states,
                                        size_t bridge_count,
                                        char *error, size_t error_capacity)
{
    if (model == NULL || embedding == NULL ||
        embedding_count < model->header->hidden) {
        set_error(error, error_capacity, "invalid input embedding"); return -1;
    }
    memcpy(model->hidden, embedding, model->header->hidden * sizeof(float));
    return forward_hidden(model, NULL, 0, bridge_states, bridge_count,
                          error, error_capacity);
}
