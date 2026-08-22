#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "minimax_h3_m3_e2e.h"
#include "minimax_h3.h"
#include "minimax_h3_m3_tree.h"
#include "minimax_h3_remote_safetensors.h"
#include "qwen38_tokenizer.h"

#include <errno.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <math.h>
#include <pthread.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

static const char *h3_artifact_revision =
    "e1244ad93d60c737c7e0f065a1c9372f3de7caf8";

typedef struct {
    uint32_t rows;
    uint32_t groups_per_row;
    uint32_t batch;
} h3_gemm_parameters;

typedef struct {
    uint32_t rows;
    uint32_t columns;
    uint32_t batch;
    uint32_t input_stride;
} h3_dense_parameters;

typedef struct {
    uint32_t rows;
    uint32_t columns;
    uint32_t modulation_stride;
    uint32_t shift_offset;
    uint32_t scale_offset;
    float epsilon;
} h3_norm_parameters;

typedef struct {
    uint32_t token_count;
    uint32_t query_heads;
    uint32_t key_value_heads;
    uint32_t head_dimension;
    float scale;
} h3_qwen_attention_parameters;

typedef struct {
    uint32_t sequence_rows;
    uint32_t exact_rows;
    uint32_t video_start;
    uint32_t rows_per_video_frame;
    uint32_t patch_columns;
    uint32_t tile_columns;
    uint32_t leaves_per_frame;
    uint32_t leaf_count;
    uint32_t aggregate_start;
    uint32_t aggregate_count;
} h3_tree_parameters;

typedef struct {
    uint32_t parent;
    uint32_t first_child;
    uint32_t child_count;
    uint32_t kind;
    uint32_t first_frame;
    uint32_t frame_count;
    uint32_t patch_y;
    uint32_t patch_x;
    uint32_t patch_h;
    uint32_t patch_w;
    uint32_t token_count;
    uint32_t physical_start;
} h3_tree_node_gpu;

typedef struct {
    uint32_t first_row;
    uint32_t row_count;
    uint32_t route_index;
} h3_query_block_gpu;

typedef struct {
    uint32_t batch;
    uint32_t input_length;
    uint32_t output_length;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t kernel;
    uint32_t stride;
    uint32_t padding;
    uint32_t dilation;
} h3_audio_conv_parameters;

typedef struct {
    float *video;
    float *audio;
    uint32_t video_latent_frames;
    uint32_t latent_height;
    uint32_t latent_width;
    uint32_t audio_latent_frames;
} h3_latents;

typedef struct {
    minimax_h3_remote_safetensors remote;
    void *mapping;
    size_t mapping_bytes;
    size_t file_padding;
    void *whole_buffer;
    double download_seconds;
} h3_remote_image;

typedef struct {
    minimax_h3_remote_tensor tensor;
    __strong id<MTLBuffer> buffer;
    NSUInteger offset;
} h3_tensor_binding;

typedef struct {
    h3_tensor_binding weight;
    h3_tensor_binding scales;
    h3_tensor_binding biases;
} h3_affine_binding;

typedef struct {
    h3_tensor_binding weight;
    h3_tensor_binding bias;
} h3_dense_binding;

typedef struct {
    __strong id<MTLDevice> device;
    __strong id<MTLCommandQueue> queue;
    __strong id<MTLLibrary> library;
    __strong id<MTLBinaryArchive> pipeline_archive;
    __strong id<MTLComputePipelineState> q4;
    __strong id<MTLComputePipelineState> q4_bf16;
    __strong id<MTLComputePipelineState> q8;
    __strong id<MTLComputePipelineState> dense_bf16;
    __strong id<MTLComputePipelineState> dense_bf16_f16_to_bf16;
    __strong id<MTLComputePipelineState> dense_bf16_activation;
    __strong id<MTLComputePipelineState> dense_bf16_activation_add;
    __strong id<MTLComputePipelineState> dense_bf16_mma;
    __strong id<MTLComputePipelineState> dense_bf16_mma_add;
    __strong id<MTLComputePipelineState> dense_f32;
    __strong id<MTLComputePipelineState> dense_f32_f16_to_bf16;
    __strong id<MTLComputePipelineState> dense_f32_f32_to_bf16;
    __strong id<MTLComputePipelineState> dense_f32_bf16_to_f32;
    __strong id<MTLComputePipelineState> dense_f32_bf16_activation;
    __strong id<MTLComputePipelineState> dense_f16;
    __strong id<MTLComputePipelineState> dense_f16_mma_weight_tiled_b64;
    __strong id<MTLComputePipelineState> rms_plain;
    __strong id<MTLComputePipelineState> rms_adaln;
    __strong id<MTLComputePipelineState> rms_plain_bf16;
    __strong id<MTLComputePipelineState> rms_adaln_bf16;
    __strong id<MTLComputePipelineState> rms_f16;
    __strong id<MTLComputePipelineState> layernorm_f16;
    __strong id<MTLComputePipelineState> scaled_residual_f16;
    __strong id<MTLComputePipelineState> silu_pair;
    __strong id<MTLComputePipelineState> silu_split;
    __strong id<MTLComputePipelineState> silu_split_bf16;
    __strong id<MTLComputePipelineState> residual;
    __strong id<MTLComputePipelineState> residual_bf16;
    __strong id<MTLComputePipelineState> qwen_prepare;
    __strong id<MTLComputePipelineState> qwen_attention;
    __strong id<MTLComputePipelineState> h3_rope;
    __strong id<MTLComputePipelineState> h3_prepare;
    __strong id<MTLComputePipelineState> h3_attention;
    __strong id<MTLComputePipelineState> h3_prepare_bf16;
    __strong id<MTLComputePipelineState> h3_attention_bf16;
    __strong id<MTLComputePipelineState> h3_attention_mma64_bf16;
    __strong id<MTLComputePipelineState> h3_attention_mma64_bf16_direct;
    __strong id<MTLComputePipelineState> h3_attention_mma64_bf16_flash16;
    __strong id<MTLComputePipelineState> h3_reorder_bf16_to_f16;
    __strong id<MTLComputePipelineState> h3_reorder_f16_to_bf16;
    __strong id<MTLComputePipelineState> h3_tree_leaf_summary;
    __strong id<MTLComputePipelineState> h3_tree_parent_summary;
    __strong id<MTLComputePipelineState> h3_tree_attention_mma64;
    __strong id<MTLComputePipelineState> f32_to_bf16;
    __strong id<MTLComputePipelineState> video_rope;
    __strong id<MTLComputePipelineState> video_prepare;
    __strong id<MTLComputePipelineState> video_attention;
    __strong id<MTLComputePipelineState> video_attention_tiled;
    __strong id<MTLComputePipelineState> audio_conv;
    __strong id<MTLComputePipelineState> audio_conv_transpose;
    __strong id<MTLComputePipelineState> audio_alias;
    __strong id<MTLComputePipelineState> audio_residual;
    __strong id<MTLComputePipelineState> audio_average3;
    __strong id<MTLComputePipelineState> image_silu;
    __strong id<MTLComputePipelineState> image_add;
    int reference_vae_gemm;
    int reference_vae_attention;
    int pipeline_archive_hit;
    double setup_seconds;
} h3_metal;

typedef struct {
    int enabled;
    int leaf_summaries_only;
    uint32_t approximate_step_mask;
    uint64_t approximate_layer_mask;
    h3_tree_parameters parameters;
    uint32_t query_block_count;
    uint32_t summary_node_count;
    uint32_t leaf_count;
    uint32_t frame_node_start;
    uint32_t frame_node_count;
    uint32_t temporal_node_start;
    uint32_t temporal_node_count;
    uint32_t root_index;
    __strong id<MTLBuffer> logical_to_physical;
    __strong id<MTLBuffer> nodes;
    __strong id<MTLBuffer> summary_log_counts;
    __strong id<MTLBuffer> route_offsets;
    __strong id<MTLBuffer> route_entries;
    __strong id<MTLBuffer> query_blocks;
    __strong id<MTLBuffer> physical_query;
    __strong id<MTLBuffer> physical_key;
    __strong id<MTLBuffer> physical_value;
    __strong id<MTLBuffer> physical_output;
    __strong id<MTLBuffer> summary_key;
    __strong id<MTLBuffer> summary_value;
    __strong id<MTLBuffer> lse;
} h3_tree_runtime;

@interface H3MPSConvolutionDataSource : NSObject <MPSCNNConvolutionDataSource>
@property(nonatomic, strong) MPSCNNConvolutionDescriptor *convDescriptor;
@property(nonatomic, strong) NSData *weightStorage;
@property(nonatomic, strong) NSMutableData *biasStorage;
@property(nonatomic, copy) NSString *sourceLabel;
@end

@implementation H3MPSConvolutionDataSource
- (MPSDataType)dataType { return MPSDataTypeFloat16; }
- (MPSCNNConvolutionDescriptor *)descriptor { return self.convDescriptor; }
- (void *)weights { return (void *)self.weightStorage.bytes; }
- (float *)biasTerms { return (float *)self.biasStorage.mutableBytes; }
- (BOOL)load { return YES; }
- (void)purge {}
- (NSString *)label { return self.sourceLabel; }
- (id)copyWithZone:(NSZone *)zone {
    (void)zone;
    return self;
}
@end

@interface H3MPSGroupNormDataSource : NSObject <MPSCNNGroupNormalizationDataSource>
@property(nonatomic, strong) NSMutableData *gammaStorage;
@property(nonatomic, strong) NSMutableData *betaStorage;
@property(nonatomic) NSUInteger channelCount;
@property(nonatomic) NSUInteger numberOfGroups;
@property(nonatomic, copy) NSString *sourceLabel;
@end

@implementation H3MPSGroupNormDataSource
- (float *)gamma { return (float *)self.gammaStorage.mutableBytes; }
- (float *)beta { return (float *)self.betaStorage.mutableBytes; }
- (NSUInteger)numberOfFeatureChannels { return self.channelCount; }
- (NSString *)label { return self.sourceLabel; }
- (float)epsilon { return 1e-6f; }
- (id)copyWithZone:(NSZone *)zone {
    (void)zone;
    return self;
}
@end

static void e2e_error(char *error, size_t capacity, const char *format, ...) {
    va_list arguments;
    if (error == NULL || capacity == 0u) return;
    va_start(arguments, format);
    vsnprintf(error, capacity, format, arguments);
    va_end(arguments);
}

static double e2e_now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static size_t e2e_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    return task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info,
                     &count) == KERN_SUCCESS
               ? (size_t)info.phys_footprint
               : 0u;
}

static double e2e_timeval(struct timeval value) {
    return (double)value.tv_sec + (double)value.tv_usec * 1e-6;
}

static int h3_remote_image_open_path(const char *path,
                                     const char *stage,
                                     h3_remote_image *image,
                                     char *error,
                                     size_t error_capacity) {
    h3_remote_image result;
    memset(&result, 0, sizeof(result));
    int cache_fd = open(path, O_RDONLY);
    struct stat cache_info;
    if (cache_fd < 0 || fstat(cache_fd, &cache_info) != 0 ||
        cache_info.st_size < 16) {
        if (cache_fd >= 0) close(cache_fd);
        e2e_error(error, error_capacity,
                  "local MiniMax-H3 artifact is missing: %s; inference is offline",
                  path);
        return 1;
    }
    result.mapping_bytes = (size_t)cache_info.st_size;
    result.mapping = mmap(NULL, result.mapping_bytes, PROT_READ, MAP_PRIVATE,
                          cache_fd, 0);
    close(cache_fd);
    if (result.mapping == MAP_FAILED) {
        result.mapping = NULL;
        e2e_error(error, error_capacity, "cannot map local artifact: %s",
                  path);
        return 1;
    }
    const unsigned char *bytes = result.mapping;
    uint64_t header_length = 0u;
    for (size_t padding = 0u; padding < 4u; ++padding) {
        uint64_t candidate = 0u;
        for (size_t index = 0u; index < 8u; ++index)
            candidate |= (uint64_t)bytes[padding + index] << (index * 8u);
        if (candidate != 0u && candidate <= UINT64_C(64) * 1024u * 1024u &&
            candidate <= result.mapping_bytes - padding - 9u &&
            bytes[padding + 8u] == '{') {
            result.file_padding = padding;
            header_length = candidate;
            break;
        }
    }
    if (header_length == 0u) {
        munmap(result.mapping, result.mapping_bytes);
        result.mapping = NULL;
        e2e_error(error, error_capacity, "invalid local artifact header: %s",
                  path);
        return 1;
    }
    result.remote.header = malloc((size_t)header_length + 1u);
    if (result.remote.header == NULL) {
        munmap(result.mapping, result.mapping_bytes);
        result.mapping = NULL;
        e2e_error(error, error_capacity,
                  "local artifact header allocation failed");
        return 1;
    }
    memcpy(result.remote.header, bytes + result.file_padding + 8u,
           (size_t)header_length);
    result.remote.header[header_length] = '\0';
    result.remote.header_length = (size_t)header_length;
    result.remote.payload_start = 8u + header_length;
    result.remote.file_length = result.mapping_bytes - result.file_padding;
    result.download_seconds = 0.0;
    fprintf(stderr, "stage=%s cache=hit bytes=%zu path=%s\n", stage,
            result.mapping_bytes, path);
    fflush(stderr);
    *image = result;
    return 0;
}

static int h3_remote_image_open(const char *filename,
                                const char *stage,
                                h3_remote_image *image,
                                char *error,
                                size_t error_capacity) {
    char cache_path[1024];
    if (snprintf(cache_path, sizeof(cache_path),
                 "tmp/minimax-h3-m3-cache/%s.%s.h3cache", filename,
                 h3_artifact_revision) >= (int)sizeof(cache_path)) {
        e2e_error(error, error_capacity, "MiniMax-H3 cache path is too long");
        return 1;
    }
    return h3_remote_image_open_path(cache_path, stage, image, error,
                                     error_capacity);
}

static void h3_remote_image_close(h3_remote_image *image) {
    if (image == NULL) return;
    if (image->whole_buffer != NULL) {
        id whole_buffer = (__bridge_transfer id)image->whole_buffer;
        image->whole_buffer = NULL;
        (void)whole_buffer;
    }
    if (image->mapping != NULL) munmap(image->mapping, image->mapping_bytes);
    minimax_h3_remote_safetensors_close(&image->remote);
    memset(image, 0, sizeof(*image));
}

static id<MTLComputePipelineState> h3_pipeline(id<MTLDevice> device,
                                               id<MTLLibrary> library,
                                               id<MTLBinaryArchive> archive,
                                               NSString *name,
                                               NSError **error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) return nil;
    if (archive == nil)
        return [device newComputePipelineStateWithFunction:function
                                                       error:error];
    MTLComputePipelineDescriptor *descriptor =
        [MTLComputePipelineDescriptor new];
    descriptor.label = name;
    descriptor.computeFunction = function;
    descriptor.binaryArchives = @[ archive ];
    return [device newComputePipelineStateWithDescriptor:descriptor
                                                 options:MTLPipelineOptionNone
                                              reflection:nil
                                                   error:error];
}

static int h3_metal_open(const char *path,
                         h3_metal *metal,
                         char *error_message,
                         size_t error_capacity) {
    h3_metal result;
    memset((void *)&result, 0, sizeof(result));
    double setup_started = e2e_now();
    const char *reference_vae_gemm = getenv("MINIMAX_H3_REFERENCE_VAE_GEMM");
    result.reference_vae_gemm = reference_vae_gemm != NULL &&
                                strcmp(reference_vae_gemm, "1") == 0;
    const char *reference_vae_attention =
        getenv("MINIMAX_H3_REFERENCE_VAE_ATTENTION");
    result.reference_vae_attention =
        reference_vae_attention != NULL &&
        strcmp(reference_vae_attention, "1") == 0;
    NSError *error = nil;
    result.device = MTLCreateSystemDefaultDevice();
    if (result.device != nil) {
        result.library = [result.device newLibraryWithURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]]
                                                error:&error];
        result.queue = [result.device newCommandQueue];
        NSString *archive_path = [[[NSString stringWithUTF8String:path]
            stringByDeletingPathExtension]
            stringByAppendingPathExtension:@"mtlarchive"];
        const char *use_archive = getenv("MINIMAX_H3_USE_PIPELINE_ARCHIVE");
        if (use_archive != NULL && strcmp(use_archive, "1") == 0 &&
            [[NSFileManager defaultManager] fileExistsAtPath:archive_path]) {
            MTLBinaryArchiveDescriptor *archive_descriptor =
                [MTLBinaryArchiveDescriptor new];
            archive_descriptor.url = [NSURL fileURLWithPath:archive_path];
            NSError *archive_error = nil;
            result.pipeline_archive =
                [result.device newBinaryArchiveWithDescriptor:archive_descriptor
                                                        error:&archive_error];
            result.pipeline_archive_hit = result.pipeline_archive != nil;
        }
    }
#define H3_PIPE(field, name)                                                   \
    result.field = result.library != nil                                      \
        ? h3_pipeline(result.device, result.library, result.pipeline_archive, \
                      @name, &error)                                          \
        : nil
    H3_PIPE(q4, "minimax_h3_q4_gemm_bf16_meta");
    H3_PIPE(q4_bf16, "minimax_h3_q4_gemm_bf16_activation");
    H3_PIPE(q8, "minimax_h3_q8_gemm_bf16_meta");
    H3_PIPE(dense_bf16, "minimax_h3_dense_bf16");
    H3_PIPE(dense_bf16_f16_to_bf16,
            "minimax_h3_dense_bf16_f16_to_bf16");
    H3_PIPE(dense_bf16_activation,
            "minimax_h3_dense_bf16_activation");
    H3_PIPE(dense_bf16_activation_add,
            "minimax_h3_dense_bf16_activation_add");
    H3_PIPE(dense_bf16_mma, "minimax_h3_dense_bf16_mma");
    H3_PIPE(dense_bf16_mma_add, "minimax_h3_dense_bf16_mma_add");
    H3_PIPE(dense_f32, "minimax_h3_dense_f32");
    H3_PIPE(dense_f32_f16_to_bf16,
            "minimax_h3_dense_f32_f16_to_bf16");
    H3_PIPE(dense_f32_f32_to_bf16,
            "minimax_h3_dense_f32_f32_to_bf16");
    H3_PIPE(dense_f32_bf16_to_f32,
            "minimax_h3_dense_f32_bf16_to_f32");
    H3_PIPE(dense_f32_bf16_activation,
            "minimax_h3_dense_f32_bf16_activation");
    H3_PIPE(dense_f16, "minimax_h3_dense_f16");
    H3_PIPE(dense_f16_mma_weight_tiled_b64,
            "minimax_h3_dense_f16_mma_weight_tiled_b64");
    H3_PIPE(rms_plain, "minimax_h3_rms_plain");
    H3_PIPE(rms_adaln, "minimax_h3_rms_adaln");
    H3_PIPE(rms_plain_bf16, "minimax_h3_rms_plain_bf16");
    H3_PIPE(rms_adaln_bf16, "minimax_h3_rms_adaln_bf16");
    H3_PIPE(rms_f16, "minimax_h3_rms_f16");
    H3_PIPE(layernorm_f16, "minimax_h3_layernorm_f16");
    H3_PIPE(scaled_residual_f16, "minimax_h3_scaled_residual_f16");
    H3_PIPE(silu_pair, "minimax_h3_silu_pair");
    H3_PIPE(silu_split, "minimax_h3_silu_split");
    H3_PIPE(silu_split_bf16, "minimax_h3_silu_split_bf16");
    H3_PIPE(residual, "minimax_h3_gated_residual");
    H3_PIPE(residual_bf16, "minimax_h3_gated_residual_bf16");
    H3_PIPE(qwen_prepare, "minimax_h3_qwen_prepare_qkv");
    H3_PIPE(qwen_attention, "minimax_h3_qwen_causal_attention");
    H3_PIPE(h3_rope, "minimax_h3_build_rope_f32");
    H3_PIPE(h3_prepare, "minimax_h3_prepare_qkv");
    H3_PIPE(h3_attention, "minimax_h3_hierarchical_attention");
    H3_PIPE(h3_prepare_bf16, "minimax_h3_prepare_qkv_bf16");
    H3_PIPE(h3_attention_bf16, "minimax_h3_dense_attention_bf16");
    H3_PIPE(h3_attention_mma64_bf16,
            "minimax_h3_dense_attention_mma64_bf16");
    H3_PIPE(h3_attention_mma64_bf16_direct,
            "minimax_h3_dense_attention_mma64_bf16_direct");
    H3_PIPE(h3_attention_mma64_bf16_flash16,
            "minimax_h3_dense_attention_mma64_bf16_flash16");
    H3_PIPE(h3_reorder_bf16_to_f16,
            "minimax_h3_reorder_bf16_to_f16");
    H3_PIPE(h3_reorder_f16_to_bf16,
            "minimax_h3_reorder_f16_to_bf16");
    H3_PIPE(h3_tree_leaf_summary,
            "minimax_h3_build_leaf_summaries");
    H3_PIPE(h3_tree_parent_summary,
            "minimax_h3_build_parent_summaries");
    H3_PIPE(h3_tree_attention_mma64,
            "minimax_h3_hierarchical_attention_mma64");
    H3_PIPE(f32_to_bf16, "minimax_h3_f32_to_bf16");
    H3_PIPE(video_rope, "minimax_h3_build_video_rope_f32");
    H3_PIPE(video_prepare, "minimax_h3_video_prepare_qkv");
    H3_PIPE(video_attention, "minimax_h3_video_attention");
    H3_PIPE(video_attention_tiled,
            "minimax_h3_video_attention_tiled8");
    H3_PIPE(audio_conv, "minimax_h3_audio_conv1d_f32");
    H3_PIPE(audio_conv_transpose, "minimax_h3_audio_conv_transpose1d_f32");
    H3_PIPE(audio_alias, "minimax_h3_audio_alias_snake_f32");
    H3_PIPE(audio_residual, "minimax_h3_audio_residual_f32");
    H3_PIPE(audio_average3, "minimax_h3_audio_average3_f32");
    H3_PIPE(image_silu, "minimax_h3_mps_image_silu");
    H3_PIPE(image_add, "minimax_h3_mps_image_add");
#undef H3_PIPE
    if (result.device == nil || result.library == nil || result.queue == nil ||
        result.q4 == nil || result.q4_bf16 == nil || result.q8 == nil ||
        result.dense_bf16 == nil ||
        result.dense_bf16_f16_to_bf16 == nil ||
        result.dense_bf16_activation == nil ||
        result.dense_bf16_activation_add == nil ||
        result.dense_bf16_mma == nil || result.dense_bf16_mma_add == nil ||
        result.dense_f32 == nil || result.dense_f32_f16_to_bf16 == nil ||
        result.dense_f32_f32_to_bf16 == nil ||
        result.dense_f32_bf16_to_f32 == nil ||
        result.dense_f32_bf16_activation == nil ||
        result.dense_f16 == nil ||
        result.dense_f16_mma_weight_tiled_b64 == nil ||
        result.rms_plain == nil || result.rms_adaln == nil ||
        result.rms_plain_bf16 == nil || result.rms_adaln_bf16 == nil ||
        result.silu_pair == nil ||
        result.rms_f16 == nil || result.layernorm_f16 == nil ||
        result.scaled_residual_f16 == nil ||
        result.silu_split == nil || result.silu_split_bf16 == nil ||
        result.residual == nil || result.residual_bf16 == nil ||
        result.qwen_prepare == nil || result.qwen_attention == nil ||
        result.h3_rope == nil ||
        result.h3_prepare == nil || result.h3_attention == nil ||
        result.h3_prepare_bf16 == nil || result.h3_attention_bf16 == nil ||
        result.h3_attention_mma64_bf16 == nil ||
        result.h3_attention_mma64_bf16_direct == nil ||
        result.h3_reorder_bf16_to_f16 == nil ||
        result.h3_reorder_f16_to_bf16 == nil ||
        result.h3_tree_leaf_summary == nil ||
        result.h3_tree_parent_summary == nil ||
        result.h3_tree_attention_mma64 == nil ||
        result.f32_to_bf16 == nil || result.video_rope == nil ||
        result.video_prepare == nil || result.video_attention == nil ||
        result.video_attention_tiled == nil ||
        result.audio_conv == nil || result.audio_conv_transpose == nil ||
        result.audio_alias == nil || result.audio_residual == nil ||
        result.audio_average3 == nil || result.image_silu == nil ||
        result.image_add == nil) {
        const char *description = error.localizedDescription.UTF8String;
        e2e_error(error_message, error_capacity, "Metal setup failed: %s",
                  description != NULL ? description : "missing pipeline");
        return 1;
    }
    result.setup_seconds = e2e_now() - setup_started;
    fprintf(stderr, "stage=metal-setup pipeline_archive=%s seconds=%.6f\n",
            result.pipeline_archive_hit ? "hit" : "miss",
            result.setup_seconds);
    *metal = result;
    return 0;
}

static int h3_wait_committed(id<MTLCommandBuffer> command,
                             char *error,
                             size_t error_capacity) {
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        e2e_error(error, error_capacity, "Metal command failed: %s",
                  command.error.localizedDescription.UTF8String);
        return 1;
    }
    return 0;
}

static int h3_wait(id<MTLCommandBuffer> command,
                   char *error,
                   size_t error_capacity) {
    [command commit];
    return h3_wait_committed(command, error, error_capacity);
}

static int h3_bind_tensor(h3_remote_image *image,
                          h3_metal *metal,
                          const char *name,
                          h3_tensor_binding *binding,
                          char *error,
                          size_t error_capacity) {
    minimax_h3_remote_tensor tensor;
    if (minimax_h3_remote_safetensors_find(&image->remote, name, &tensor,
                                            error, error_capacity) != 0)
        return 1;
    long page_value = sysconf(_SC_PAGESIZE);
    if (page_value <= 0) {
        e2e_error(error, error_capacity, "cannot query VM page size");
        return 1;
    }
    size_t page = (size_t)page_value;
    if (image->whole_buffer == NULL && page != 0u &&
        image->mapping_bytes <= SIZE_MAX - (page - 1u)) {
        size_t whole_length =
            (image->mapping_bytes + page - 1u) & ~(page - 1u);
        if (whole_length <= metal->device.maxBufferLength) {
            id<MTLBuffer> whole = [metal->device
                newBufferWithBytesNoCopy:image->mapping
                                  length:whole_length
                                 options:MTLResourceStorageModeShared
                             deallocator:^(__unused void *pointer,
                                           __unused NSUInteger bytes) {}];
            if (whole != nil)
                image->whole_buffer = (__bridge_retained void *)whole;
        }
    }
    if (image->whole_buffer != NULL) {
        uint64_t offset = (uint64_t)image->file_padding + tensor.data_start;
        if (offset <= NSUIntegerMax && offset <= image->mapping_bytes &&
            tensor.data_length <= image->mapping_bytes - (size_t)offset) {
            binding->tensor = tensor;
            binding->buffer = (__bridge id<MTLBuffer>)image->whole_buffer;
            binding->offset = (NSUInteger)offset;
            return 0;
        }
    }
    uintptr_t address = (uintptr_t)image->mapping + image->file_padding +
                        (uintptr_t)tensor.data_start;
    uintptr_t page_address = address & ~(uintptr_t)(page - 1u);
    size_t offset = (size_t)(address - page_address);
    uint64_t needed = (uint64_t)offset + tensor.data_length;
    if (needed > SIZE_MAX - page) {
        e2e_error(error, error_capacity, "tensor view is too large: %s", name);
        return 1;
    }
    size_t length = ((size_t)needed + page - 1u) & ~(page - 1u);
    if (length > metal->device.maxBufferLength) {
        e2e_error(error, error_capacity, "tensor exceeds Metal buffer: %s", name);
        return 1;
    }
    id<MTLBuffer> buffer = [metal->device
        newBufferWithBytesNoCopy:(void *)page_address
                          length:length
                         options:MTLResourceStorageModeShared
                     deallocator:^(__unused void *pointer,
                                   __unused NSUInteger bytes) {}];
    if (buffer == nil) {
        e2e_error(error, error_capacity, "cannot bind tensor: %s", name);
        return 1;
    }
    binding->tensor = tensor;
    binding->buffer = buffer;
    binding->offset = offset;
    return 0;
}

typedef struct {
    char name[192];
    h3_tensor_binding *binding;
    minimax_h3_remote_tensor tensor;
    void *storage;
    size_t storage_bytes;
} h3_fetch_job;

typedef struct {
    size_t job_index;
    uint64_t offset;
    size_t length;
} h3_fetch_chunk;

typedef struct {
    const minimax_h3_remote_safetensors *remote;
    h3_fetch_job *jobs;
    h3_fetch_chunk *chunks;
    size_t chunk_count;
    size_t next_chunk;
    size_t completed_chunks;
    int failed;
    pthread_mutex_t mutex;
    char message[512];
} h3_fetch_batch;

static void *h3_fetch_worker(void *opaque) {
    h3_fetch_batch *batch = opaque;
    for (;;) {
        size_t index;
        pthread_mutex_lock(&batch->mutex);
        if (batch->failed || batch->next_chunk >= batch->chunk_count) {
            pthread_mutex_unlock(&batch->mutex);
            break;
        }
        index = batch->next_chunk++;
        pthread_mutex_unlock(&batch->mutex);
        h3_fetch_chunk chunk = batch->chunks[index];
        h3_fetch_job *job = &batch->jobs[chunk.job_index];
        char local_error[512] = {0};
        const minimax_h3_remote_safetensors *source = batch->remote;
        int status = minimax_h3_remote_safetensors_read(
            source, job->tensor.data_start + chunk.offset,
            (unsigned char *)job->storage + (size_t)chunk.offset,
            chunk.length, local_error, sizeof(local_error));
        if (status != 0) {
            pthread_mutex_lock(&batch->mutex);
            if (!batch->failed) {
                batch->failed = 1;
                snprintf(batch->message, sizeof(batch->message), "%s: %s",
                         job->name,
                         local_error[0] != '\0' ? local_error
                                                  : "tensor chunk fetch failed");
            }
            pthread_mutex_unlock(&batch->mutex);
            break;
        }
        pthread_mutex_lock(&batch->mutex);
        ++batch->completed_chunks;
        pthread_mutex_unlock(&batch->mutex);
    }
    return NULL;
}

static int h3_fetch_batch_run(
    const minimax_h3_remote_safetensors *remote,
    h3_metal *metal,
    h3_fetch_job *jobs,
    size_t job_count,
    double *download_seconds,
    char *error,
    size_t error_capacity) {
    enum { worker_count = 32, chunk_bytes = 8 * 1024 * 1024 };
    long page_value = sysconf(_SC_PAGESIZE);
    size_t chunk_count = 0u;
    if (remote == NULL || metal == NULL || jobs == NULL || job_count == 0u ||
        page_value <= 0) {
        e2e_error(error, error_capacity, "invalid conditioner fetch batch");
        return 1;
    }
    size_t page = (size_t)page_value;
    for (size_t index = 0u; index < job_count; ++index) {
        h3_fetch_job *job = &jobs[index];
        if (minimax_h3_remote_safetensors_find(
                remote, job->name, &job->tensor, error,
                error_capacity) != 0 || job->tensor.data_length == 0u ||
            job->tensor.data_length > SIZE_MAX - page) {
            goto prepare_failed;
        }
        job->storage_bytes =
            ((size_t)job->tensor.data_length + page - 1u) & ~(page - 1u);
        if (job->storage_bytes > metal->device.maxBufferLength ||
            posix_memalign(&job->storage, page, job->storage_bytes) != 0 ||
            job->storage == NULL) {
            e2e_error(error, error_capacity,
                      "cannot allocate streamed tensor: %s", job->name);
            goto prepare_failed;
        }
        size_t job_chunks =
            ((size_t)job->tensor.data_length + chunk_bytes - 1u) /
            chunk_bytes;
        if (job_chunks > SIZE_MAX - chunk_count) {
            e2e_error(error, error_capacity, "conditioner chunk count overflow");
            goto prepare_failed;
        }
        chunk_count += job_chunks;
    }
    h3_fetch_chunk *chunks = calloc(chunk_count, sizeof(*chunks));
    if (chunks == NULL) {
        e2e_error(error, error_capacity, "cannot allocate conditioner chunks");
        goto prepare_failed;
    }
    size_t chunk_index = 0u;
    for (size_t job_index = 0u; job_index < job_count; ++job_index) {
        uint64_t offset = 0u;
        while (offset < jobs[job_index].tensor.data_length) {
            uint64_t remaining = jobs[job_index].tensor.data_length - offset;
            size_t length = remaining < chunk_bytes ? (size_t)remaining
                                                    : chunk_bytes;
            chunks[chunk_index++] = (h3_fetch_chunk) {
                .job_index = job_index,
                .offset = offset,
                .length = length,
            };
            offset += length;
        }
    }
    h3_fetch_batch batch = {
        .remote = remote,
        .jobs = jobs,
        .chunks = chunks,
        .chunk_count = chunk_count,
        .mutex = PTHREAD_MUTEX_INITIALIZER,
    };
    pthread_t workers[worker_count];
    size_t started = 0u;
    double download_started = e2e_now();
    for (; started < worker_count && started < chunk_count; ++started) {
        if (pthread_create(&workers[started], NULL, h3_fetch_worker,
                           &batch) != 0) {
            pthread_mutex_lock(&batch.mutex);
            batch.failed = 1;
            snprintf(batch.message, sizeof(batch.message),
                     "cannot start conditioner fetch worker %zu", started);
            pthread_mutex_unlock(&batch.mutex);
            break;
        }
    }
    for (size_t index = 0u; index < started; ++index)
        pthread_join(workers[index], NULL);
    if (download_seconds != NULL)
        *download_seconds += e2e_now() - download_started;
    pthread_mutex_destroy(&batch.mutex);
    free(chunks);
    if (batch.failed || batch.completed_chunks != chunk_count) {
        e2e_error(error, error_capacity, "%s",
                  batch.message[0] != '\0' ? batch.message
                                             : "incomplete conditioner fetch");
        goto prepare_failed;
    }
    for (size_t index = 0u; index < job_count; ++index) {
        h3_fetch_job *job = &jobs[index];
        id<MTLBuffer> buffer = [metal->device
            newBufferWithBytesNoCopy:job->storage
                              length:job->storage_bytes
                             options:MTLResourceStorageModeShared
                         deallocator:^(void *pointer,
                                       __unused NSUInteger length) {
                             free(pointer);
                         }];
        if (buffer == nil) {
            e2e_error(error, error_capacity,
                      "cannot bind streamed tensor: %s", job->name);
            goto prepare_failed;
        }
        job->binding->tensor = job->tensor;
        job->binding->buffer = buffer;
        job->binding->offset = 0u;
        job->storage = NULL;
    }
    return 0;

prepare_failed:
    for (size_t index = 0u; index < job_count; ++index) {
        free(jobs[index].storage);
        jobs[index].storage = NULL;
    }
    return 1;
}

static void h3_set(id<MTLComputeCommandEncoder> encoder,
                   h3_tensor_binding binding,
                   NSUInteger index) {
    [encoder setBuffer:binding.buffer offset:binding.offset atIndex:index];
}

static void h3_set_at(id<MTLComputeCommandEncoder> encoder,
                      h3_tensor_binding binding,
                      NSUInteger additional_offset,
                      NSUInteger index) {
    [encoder setBuffer:binding.buffer
                offset:binding.offset + additional_offset
               atIndex:index];
}

static int h3_q4_linear(h3_remote_image *image,
                        h3_metal *metal,
                        id<MTLComputeCommandEncoder> encoder,
                        const char *prefix,
                        id<MTLBuffer> input,
                        id<MTLBuffer> output,
                        uint32_t batch,
                        char *error,
                        size_t error_capacity) {
    char name[192];
    h3_tensor_binding weight = { {0}, nil, 0u };
    h3_tensor_binding scales = { {0}, nil, 0u };
    h3_tensor_binding biases = { {0}, nil, 0u };
#define H3_BIND_Q4(target, suffix)                                             \
    do {                                                                      \
        snprintf(name, sizeof(name), "%s.%s", prefix, suffix);               \
        if (h3_bind_tensor(image, metal, name, &target, error,                \
                           error_capacity) != 0) return 1;                    \
    } while (0)
    H3_BIND_Q4(weight, "weight");
    H3_BIND_Q4(scales, "scales");
    H3_BIND_Q4(biases, "biases");
#undef H3_BIND_Q4
    /* MLX packs eight 4-bit values in each U32.  One affine metadata entry
     * covers 64 values, so one scale/bias group spans eight packed words. */
    if (strcmp(weight.tensor.dtype, "U32") != 0 || weight.tensor.rank != 2u ||
        weight.tensor.shape[1] % 8u != 0u ||
        strcmp(scales.tensor.dtype, "BF16") != 0 || scales.tensor.rank != 2u ||
        strcmp(biases.tensor.dtype, "BF16") != 0 || biases.tensor.rank != 2u ||
        scales.tensor.shape[0] != weight.tensor.shape[0] ||
        biases.tensor.shape[0] != weight.tensor.shape[0] ||
        scales.tensor.shape[1] != weight.tensor.shape[1] / 8u ||
        biases.tensor.shape[1] != scales.tensor.shape[1]) {
        e2e_error(error, error_capacity, "invalid affine-Q4 tensor: %s", prefix);
        return 1;
    }
    h3_gemm_parameters parameters = {
        .rows = (uint32_t)weight.tensor.shape[0],
        .groups_per_row = (uint32_t)scales.tensor.shape[1],
        .batch = batch,
    };
    [encoder setComputePipelineState:metal->q4_bf16];
    [encoder setBuffer:input offset:0 atIndex:0];
    h3_set(encoder, weight, 1);
    h3_set(encoder, scales, 2);
    h3_set(encoder, biases, 3);
    [encoder setBuffer:output offset:0 atIndex:4];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:5];
    [encoder dispatchThreadgroups:
        MTLSizeMake((parameters.rows + 63u) / 64u, (batch + 31u) / 32u, 1u)
            threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
    return 0;
}

static int h3_q4_linear_lora(h3_remote_image *image,
                             h3_remote_image *adapter,
                             h3_metal *metal,
                             id<MTLComputeCommandEncoder> encoder,
                             const char *prefix,
                             id<MTLBuffer> input,
                             id<MTLBuffer> lora_scratch,
                             id<MTLBuffer> output,
                             uint32_t batch,
                             char *error,
                             size_t error_capacity) {
    if (h3_q4_linear(image, metal, encoder, prefix, input, output, batch,
                     error, error_capacity) != 0)
        return 1;
    if (adapter == NULL) return 0;

    char name[224];
    h3_tensor_binding down = { {0}, nil, 0u };
    h3_tensor_binding up = { {0}, nil, 0u };
    snprintf(name, sizeof(name), "%s.lora_A.weight", prefix);
    if (h3_bind_tensor(adapter, metal, name, &down, error,
                       error_capacity) != 0)
        return 1;
    snprintf(name, sizeof(name), "%s.lora_B.weight", prefix);
    if (h3_bind_tensor(adapter, metal, name, &up, error,
                       error_capacity) != 0)
        return 1;
    if (strcmp(down.tensor.dtype, "BF16") != 0 || down.tensor.rank != 2u ||
        strcmp(up.tensor.dtype, "BF16") != 0 || up.tensor.rank != 2u ||
        down.tensor.shape[0] != up.tensor.shape[1] ||
        down.tensor.shape[0] == 0u || down.tensor.shape[0] > UINT32_MAX ||
        down.tensor.shape[1] > UINT32_MAX || up.tensor.shape[0] > UINT32_MAX ||
        lora_scratch.length <
            (NSUInteger)batch * down.tensor.shape[0] * sizeof(uint16_t)) {
        e2e_error(error, error_capacity, "invalid Turbo LoRA pair: %s", prefix);
        return 1;
    }
    h3_dense_parameters parameters = {
        .rows = (uint32_t)down.tensor.shape[0],
        .columns = (uint32_t)down.tensor.shape[1],
        .batch = batch,
        .input_stride = (uint32_t)down.tensor.shape[1],
    };
    const char *lora_mma_text = getenv("MINIMAX_H3_LORA_MMA");
    const int use_lora_mma =
        lora_mma_text != NULL && strcmp(lora_mma_text, "1") == 0 &&
        down.tensor.shape[0] % 64u == 0u &&
        down.tensor.shape[1] % 8u == 0u &&
        up.tensor.shape[0] % 64u == 0u && up.tensor.shape[1] % 8u == 0u;
    uint32_t has_bias = 0u;
    [encoder setComputePipelineState:use_lora_mma
        ? metal->dense_bf16_mma : metal->dense_bf16_activation];
    [encoder setBuffer:input offset:0 atIndex:0];
    h3_set(encoder, down, 1);
    h3_set(encoder, down, 2);
    [encoder setBuffer:lora_scratch offset:0 atIndex:3];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
    [encoder setBytes:&has_bias length:sizeof(has_bias) atIndex:5];
    uint32_t groups = batch * parameters.rows;
    if (use_lora_mma) {
        [encoder dispatchThreadgroups:
            MTLSizeMake((parameters.rows + 63u) / 64u,
                        (batch + 31u) / 32u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
    } else {
        [encoder dispatchThreadgroups:MTLSizeMake((groups + 3u) / 4u, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
    }

    parameters.rows = (uint32_t)up.tensor.shape[0];
    parameters.columns = (uint32_t)up.tensor.shape[1];
    parameters.input_stride = parameters.columns;
    [encoder setComputePipelineState:use_lora_mma
        ? metal->dense_bf16_mma_add : metal->dense_bf16_activation_add];
    [encoder setBuffer:lora_scratch offset:0 atIndex:0];
    h3_set(encoder, up, 1);
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    groups = batch * parameters.rows;
    if (use_lora_mma) {
        [encoder dispatchThreadgroups:
            MTLSizeMake((parameters.rows + 63u) / 64u,
                        (batch + 31u) / 32u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
    } else {
        [encoder dispatchThreadgroups:MTLSizeMake((groups + 3u) / 4u, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
    }
    return 0;
}

static int h3_dense_bind(h3_remote_image *image,
                         h3_metal *metal,
                         const char *prefix,
                         h3_dense_binding *binding,
                         char *error,
                         size_t error_capacity) {
    char name[192];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    if (h3_bind_tensor(image, metal, name, &binding->weight, error,
                       error_capacity) != 0)
        return 1;
    snprintf(name, sizeof(name), "%s.bias", prefix);
    if (h3_bind_tensor(image, metal, name, &binding->bias, error,
                       error_capacity) != 0)
        return 1;
    h3_tensor_binding weight = binding->weight;
    h3_tensor_binding bias = binding->bias;
    if (weight.tensor.rank != 2u || bias.tensor.rank != 1u ||
        weight.tensor.shape[0] != bias.tensor.shape[0] ||
        (strcmp(weight.tensor.dtype, "BF16") != 0 &&
         strcmp(weight.tensor.dtype, "F32") != 0 &&
         strcmp(weight.tensor.dtype, "F16") != 0)) {
        e2e_error(error, error_capacity, "invalid dense tensor: %s", prefix);
        return 1;
    }
    return 0;
}

static int h3_dense_linear_bound(const h3_dense_binding *binding,
                                 h3_metal *metal,
                                 id<MTLComputeCommandEncoder> encoder,
                                 id<MTLBuffer> input,
                                 id<MTLBuffer> output,
                                 uint32_t batch,
                                 uint32_t input_stride,
                                 char *error,
                                 size_t error_capacity) {
    h3_tensor_binding weight = binding->weight;
    h3_tensor_binding bias = binding->bias;
    if (weight.tensor.shape[1] > input_stride) {
        e2e_error(error, error_capacity,
                  "dense input stride %u is smaller than %llu columns",
                  input_stride,
                  (unsigned long long)weight.tensor.shape[1]);
        return 1;
    }
    h3_dense_parameters parameters = {
        .rows = (uint32_t)weight.tensor.shape[0],
        .columns = (uint32_t)weight.tensor.shape[1],
        .batch = batch,
        .input_stride = input_stride,
    };
    uint32_t has_bias = 1u;
    id<MTLComputePipelineState> pipeline = metal->dense_f32;
    if (strcmp(weight.tensor.dtype, "BF16") == 0) pipeline = metal->dense_bf16;
    int f16_mma = !metal->reference_vae_gemm &&
                  strcmp(weight.tensor.dtype, "F16") == 0 &&
                  parameters.rows % 64u == 0u &&
                  parameters.columns % 64u == 0u;
    if (strcmp(weight.tensor.dtype, "F16") == 0) {
        pipeline = f16_mma ? metal->dense_f16_mma_weight_tiled_b64
                           : metal->dense_f16;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    h3_set(encoder, weight, 1);
    h3_set(encoder, bias, 2);
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
    [encoder setBytes:&has_bias length:sizeof(has_bias) atIndex:5];
    if (f16_mma) {
        [encoder dispatchThreadgroups:
            MTLSizeMake(parameters.rows / 64u, (batch + 63u) / 64u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    } else {
        uint32_t groups = batch * parameters.rows;
        [encoder dispatchThreadgroups:MTLSizeMake((groups + 3u) / 4u, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
    }
    return 0;
}

typedef enum {
    H3_DENSE_INPUT_F16 = 0,
    H3_DENSE_INPUT_BF16 = 1,
    H3_DENSE_INPUT_F32 = 2,
} h3_dense_input_format;

static int h3_dense_linear_bfloat_output(
    h3_remote_image *image,
    h3_metal *metal,
    id<MTLComputeCommandEncoder> encoder,
    const char *prefix,
    id<MTLBuffer> input,
    id<MTLBuffer> output,
    uint32_t batch,
    uint32_t input_stride,
    h3_dense_input_format input_format,
    char *error,
    size_t error_capacity) {
    char name[192];
    h3_tensor_binding weight = { {0}, nil, 0u };
    h3_tensor_binding bias = { {0}, nil, 0u };
    snprintf(name, sizeof(name), "%s.weight", prefix);
    if (h3_bind_tensor(image, metal, name, &weight, error, error_capacity) != 0)
        return 1;
    snprintf(name, sizeof(name), "%s.bias", prefix);
    if (h3_bind_tensor(image, metal, name, &bias, error, error_capacity) != 0)
        return 1;
    int weights_are_bf16 = strcmp(weight.tensor.dtype, "BF16") == 0;
    int weights_are_f32 = strcmp(weight.tensor.dtype, "F32") == 0;
    if (weight.tensor.rank != 2u || bias.tensor.rank != 1u ||
        (!weights_are_bf16 && !weights_are_f32) ||
        strcmp(bias.tensor.dtype, weight.tensor.dtype) != 0 ||
        weight.tensor.shape[0] != bias.tensor.shape[0] ||
        weight.tensor.shape[1] > input_stride) {
        e2e_error(error, error_capacity,
                  "invalid bfloat-output dense tensor: %s", prefix);
        return 1;
    }
    h3_dense_parameters parameters = {
        .rows = (uint32_t)weight.tensor.shape[0],
        .columns = (uint32_t)weight.tensor.shape[1],
        .batch = batch,
        .input_stride = input_stride,
    };
    uint32_t has_bias = 1u;
    id<MTLComputePipelineState> pipeline;
    if (input_format == H3_DENSE_INPUT_F32 && !weights_are_f32) {
        e2e_error(error, error_capacity,
                  "unsupported F32-input dense tensor: %s", prefix);
        return 1;
    }
    if (weights_are_bf16)
        pipeline = input_format == H3_DENSE_INPUT_BF16
            ? metal->dense_bf16_activation
            : metal->dense_bf16_f16_to_bf16;
    else if (input_format == H3_DENSE_INPUT_F32)
        pipeline = metal->dense_f32_f32_to_bf16;
    else
        pipeline = input_format == H3_DENSE_INPUT_BF16
            ? metal->dense_f32_bf16_activation
            : metal->dense_f32_f16_to_bf16;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    h3_set(encoder, weight, 1);
    h3_set(encoder, bias, 2);
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
    [encoder setBytes:&has_bias length:sizeof(has_bias) atIndex:5];
    uint32_t groups = batch * parameters.rows;
    [encoder dispatchThreadgroups:MTLSizeMake((groups + 3u) / 4u, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
    return 0;
}

static int h3_dense_linear_f32_output(
    h3_remote_image *image,
    h3_metal *metal,
    id<MTLComputeCommandEncoder> encoder,
    const char *prefix,
    id<MTLBuffer> input,
    id<MTLBuffer> output,
    uint32_t batch,
    uint32_t input_stride,
    char *error,
    size_t error_capacity) {
    char name[192];
    h3_tensor_binding weight = { {0}, nil, 0u };
    h3_tensor_binding bias = { {0}, nil, 0u };
    snprintf(name, sizeof(name), "%s.weight", prefix);
    if (h3_bind_tensor(image, metal, name, &weight, error, error_capacity) != 0)
        return 1;
    snprintf(name, sizeof(name), "%s.bias", prefix);
    if (h3_bind_tensor(image, metal, name, &bias, error, error_capacity) != 0)
        return 1;
    if (weight.tensor.rank != 2u || bias.tensor.rank != 1u ||
        strcmp(weight.tensor.dtype, "F32") != 0 ||
        strcmp(bias.tensor.dtype, "F32") != 0 ||
        weight.tensor.shape[0] != bias.tensor.shape[0] ||
        weight.tensor.shape[1] > input_stride) {
        e2e_error(error, error_capacity,
                  "invalid F32-output dense tensor: %s", prefix);
        return 1;
    }
    h3_dense_parameters parameters = {
        .rows = (uint32_t)weight.tensor.shape[0],
        .columns = (uint32_t)weight.tensor.shape[1],
        .batch = batch,
        .input_stride = input_stride,
    };
    uint32_t has_bias = 1u;
    [encoder setComputePipelineState:metal->dense_f32_bf16_to_f32];
    [encoder setBuffer:input offset:0 atIndex:0];
    h3_set(encoder, weight, 1);
    h3_set(encoder, bias, 2);
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
    [encoder setBytes:&has_bias length:sizeof(has_bias) atIndex:5];
    uint32_t groups = batch * parameters.rows;
    [encoder dispatchThreadgroups:MTLSizeMake((groups + 3u) / 4u, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
    return 0;
}

static int h3_rms_adaln_bound(h3_remote_image *weights,
                              h3_metal *metal,
                              id<MTLComputeCommandEncoder> encoder,
                              const char *norm_name,
                              h3_tensor_binding modulation,
                              uint32_t step,
                              uint32_t modality_rows,
                              uint32_t modulation_stride,
                              uint32_t shift_offset,
                              uint32_t scale_offset,
                              id<MTLBuffer> row_indices,
                              id<MTLBuffer> input,
                              id<MTLBuffer> output,
                              uint32_t rows,
                              char *error,
                              size_t error_capacity) {
    h3_tensor_binding norm = { {0}, nil, 0u };
    if (h3_bind_tensor(weights, metal, norm_name, &norm, error,
                       error_capacity) != 0)
        return 1;
    NSUInteger step_bytes = (NSUInteger)step * modality_rows *
                            modulation_stride * sizeof(uint16_t);
    if ((uint64_t)step_bytes + (uint64_t)modality_rows * modulation_stride * 2u >
        modulation.tensor.data_length) {
        e2e_error(error, error_capacity,
                  "AdaLN step outside compiled modulation tensor");
        return 1;
    }
    h3_norm_parameters parameters = {
        .rows = rows,
        .columns = 5376u,
        .modulation_stride = modulation_stride,
        .shift_offset = shift_offset,
        .scale_offset = scale_offset,
        .epsilon = 1e-5f,
    };
    [encoder setComputePipelineState:metal->rms_adaln_bf16];
    [encoder setBuffer:input offset:0 atIndex:0];
    h3_set(encoder, norm, 1);
    h3_set_at(encoder, modulation, step_bytes, 2);
    [encoder setBuffer:row_indices offset:0 atIndex:3];
    [encoder setBuffer:output offset:0 atIndex:4];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    return 0;
}

static void h3_dense_attention(h3_metal *metal,
                               id<MTLComputeCommandEncoder> encoder,
                               id<MTLBuffer> query,
                               id<MTLBuffer> key,
                               id<MTLBuffer> value,
                               id<MTLBuffer> output,
                               uint32_t rows) {
    [encoder setComputePipelineState:metal->h3_attention_bf16];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key offset:0 atIndex:1];
    [encoder setBuffer:value offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake((rows * 56u + 3u) / 4u, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
}

static int h3_tree_runtime_build(
    const minimax_h3_m3_e2e_options *options,
    size_t text_rows,
    h3_metal *metal,
    h3_tree_runtime *runtime,
    char *error,
    size_t error_capacity) {
    h3_tree_runtime result = {0};
    const char *mode = getenv("MINIMAX_H3_TREE_ATTENTION");
    if (mode == NULL || mode[0] == '\0' || strcmp(mode, "0") == 0) {
        *runtime = result;
        return 0;
    }
    if (strcmp(mode, "1") != 0 && strcmp(mode, "conservative") != 0 &&
        strcmp(mode, "quality") != 0) {
        e2e_error(error, error_capacity,
                  "MINIMAX_H3_TREE_ATTENTION must be 0, 1, conservative or quality");
        return 2;
    }
    const int frame_safe = strcmp(mode, "quality") == 0;
    uint32_t approximate_step_mask = frame_safe ? UINT32_C(0x8) : UINT32_MAX;
    uint64_t approximate_layer_mask = frame_safe
        ? (UINT64_C(0x3ff) << 40u)
        : UINT64_MAX;
    if (frame_safe) {
        const char *step_mask_text = getenv("MINIMAX_H3_TREE_STEP_MASK");
        const char *layer_mask_text = getenv("MINIMAX_H3_TREE_LAYER_MASK");
        char *end = NULL;
        if (step_mask_text != NULL && step_mask_text[0] != '\0') {
            errno = 0;
            unsigned long parsed = strtoul(step_mask_text, &end, 0);
            if (errno != 0 || end == step_mask_text || *end != '\0' ||
                parsed > UINT32_MAX) {
                e2e_error(error, error_capacity,
                          "MINIMAX_H3_TREE_STEP_MASK is not a uint32 mask");
                return 2;
            }
            approximate_step_mask = (uint32_t)parsed;
        }
        if (layer_mask_text != NULL && layer_mask_text[0] != '\0') {
            errno = 0;
            unsigned long long parsed = strtoull(layer_mask_text, &end, 0);
            if (errno != 0 || end == layer_mask_text || *end != '\0') {
                e2e_error(error, error_capacity,
                          "MINIMAX_H3_TREE_LAYER_MASK is not a uint64 mask");
                return 2;
            }
            approximate_layer_mask = (uint64_t)parsed &
                ((UINT64_C(1) << 50u) - UINT64_C(1));
        }
    }

    minimax_h3_geometry geometry;
    minimax_h3_t2va_layout layout;
    minimax_h3_m3_tree_plan plan;
    double *positions = NULL;
    uint8_t *tags = NULL;
    minimax_h3_m3_tree_node *nodes = NULL;
    uint32_t *route_offsets = NULL;
    uint32_t *route_entries = NULL;
    uint32_t *logical_to_physical = NULL;
    h3_query_block_gpu *query_blocks = NULL;
    size_t node_count = 0u;
    size_t route_entry_count = 0u;
    size_t maximum_route_entries = 0u;
    size_t query_block_count = 0u;
    int status = 1;

    if (minimax_h3_geometry_init(&geometry, options->width, options->height,
                                 options->frames, text_rows) !=
            MINIMAX_H3_OK ||
        geometry.sequence_rows > UINT32_MAX) {
        e2e_error(error, error_capacity, "cannot compile H3 tree geometry");
        goto cleanup;
    }
    positions = calloc(geometry.sequence_rows * 3u, sizeof(*positions));
    tags = calloc(geometry.sequence_rows, sizeof(*tags));
    if (positions == NULL || tags == NULL ||
        minimax_h3_build_t2va_layout(&geometry, NULL, positions, tags,
                                     geometry.sequence_rows, &layout) !=
            MINIMAX_H3_OK ||
        minimax_h3_m3_tree_node_count(&geometry, &node_count) !=
            MINIMAX_H3_OK) {
        e2e_error(error, error_capacity, "cannot compile H3 tree layout");
        goto cleanup;
    }
    nodes = calloc(node_count, sizeof(*nodes));
    if (nodes == NULL ||
        minimax_h3_m3_tree_plan_make(&geometry, &layout, nodes, node_count,
                                     &plan) != MINIMAX_H3_OK ||
        (frame_safe
             ? minimax_h3_m3_tree_frame_safe_route_entry_count(
                   &geometry, &layout, nodes, &plan, &route_entry_count,
                   &maximum_route_entries)
             : minimax_h3_m3_tree_route_entry_count(
                   &geometry, &layout, nodes, &plan, &route_entry_count,
                   &maximum_route_entries)) != MINIMAX_H3_OK ||
        plan.exact_count > UINT32_MAX || plan.leaf_count > UINT32_MAX ||
        plan.node_count > UINT32_MAX) {
        e2e_error(error, error_capacity, "cannot compile H3 tree routes");
        goto cleanup;
    }
    query_block_count = (plan.exact_count + 63u) / 64u + plan.leaf_count;
    if (query_block_count > UINT32_MAX) {
        e2e_error(error, error_capacity, "H3 tree query plan is too large");
        goto cleanup;
    }
    route_offsets = calloc(plan.leaf_count + 2u, sizeof(*route_offsets));
    route_entries = calloc(route_entry_count, sizeof(*route_entries));
    logical_to_physical = calloc(geometry.sequence_rows,
                                 sizeof(*logical_to_physical));
    query_blocks = calloc(query_block_count, sizeof(*query_blocks));
    if (route_offsets == NULL || route_entries == NULL ||
        logical_to_physical == NULL || query_blocks == NULL ||
        (frame_safe
             ? minimax_h3_m3_tree_frame_safe_routes_make(
                   &geometry, &layout, nodes, &plan, route_offsets,
                   plan.leaf_count + 2u, route_entries, route_entry_count)
             : minimax_h3_m3_tree_routes_make(
                   &geometry, &layout, nodes, &plan, route_offsets,
                   plan.leaf_count + 2u, route_entries,
                   route_entry_count)) != MINIMAX_H3_OK) {
        e2e_error(error, error_capacity, "cannot materialize H3 tree routes");
        goto cleanup;
    }

    {
        size_t block = 0u;
        size_t physical_row = plan.video_start;
        for (size_t row = 0u; row < plan.exact_count; ++row)
            logical_to_physical[row] = (uint32_t)row;
        for (size_t row = 0u; row < plan.exact_count; row += 64u) {
            size_t count = plan.exact_count - row;
            if (count > 64u) count = 64u;
            query_blocks[block++] = (h3_query_block_gpu) {
                .first_row = (uint32_t)row,
                .row_count = (uint32_t)count,
                .route_index = (uint32_t)plan.leaf_count,
            };
        }
        for (size_t leaf = 0u; leaf < plan.leaf_count; ++leaf) {
            const minimax_h3_m3_tree_node *node = &nodes[leaf];
            if (node->token_count == 0u || node->token_count > 64u) {
                e2e_error(error, error_capacity,
                          "H3 tree leaf exceeds 64 query rows");
                goto cleanup;
            }
            query_blocks[block++] = (h3_query_block_gpu) {
                .first_row = (uint32_t)physical_row,
                .row_count = node->token_count,
                .route_index = (uint32_t)leaf,
            };
            for (uint16_t y = 0u; y < node->patch_h; ++y) {
                for (uint16_t x = 0u; x < node->patch_w; ++x) {
                    size_t logical_row = 0u;
                    if (minimax_h3_m3_tree_leaf_row(
                            &geometry, &layout, node, y, x,
                            &logical_row) != MINIMAX_H3_OK ||
                        logical_row >= geometry.sequence_rows ||
                        physical_row >= geometry.sequence_rows) {
                        e2e_error(error, error_capacity,
                                  "H3 tree row permutation is invalid");
                        goto cleanup;
                    }
                    logical_to_physical[logical_row] =
                        (uint32_t)physical_row++;
                }
            }
        }
        if (block != query_block_count ||
            physical_row != geometry.sequence_rows) {
            e2e_error(error, error_capacity,
                      "H3 tree row permutation is incomplete");
            goto cleanup;
        }
        for (size_t entry = 0u; entry < route_entry_count; ++entry) {
            if ((route_entries[entry] &
                 MINIMAX_H3_M3_TREE_SUMMARY_ENTRY) == 0u) {
                uint32_t logical_row = route_entries[entry];
                if (logical_row >= geometry.sequence_rows) {
                    e2e_error(error, error_capacity,
                              "H3 tree route row is invalid");
                    goto cleanup;
                }
                route_entries[entry] =
                    logical_to_physical[logical_row];
            }
        }
    }

    size_t tensor_elements = geometry.sequence_rows * 56u * 128u;
    size_t tensor_bytes = tensor_elements * sizeof(uint16_t);
    size_t summary_elements = plan.node_count * 56u * 128u;
    size_t summary_bytes = summary_elements * sizeof(uint16_t);
    result.logical_to_physical = [metal->device
        newBufferWithBytes:logical_to_physical
                      length:geometry.sequence_rows * sizeof(uint32_t)
                     options:MTLResourceStorageModeShared];
    result.route_offsets = [metal->device
        newBufferWithBytes:route_offsets
                      length:(plan.leaf_count + 2u) * sizeof(uint32_t)
                     options:MTLResourceStorageModeShared];
    result.route_entries = [metal->device
        newBufferWithBytes:route_entries
                      length:route_entry_count * sizeof(uint32_t)
                     options:MTLResourceStorageModeShared];
    result.query_blocks = [metal->device
        newBufferWithBytes:query_blocks
                      length:query_block_count * sizeof(h3_query_block_gpu)
                     options:MTLResourceStorageModeShared];
    result.nodes = [metal->device
        newBufferWithLength:plan.node_count * sizeof(h3_tree_node_gpu)
                    options:MTLResourceStorageModeShared];
    result.summary_log_counts = [metal->device
        newBufferWithLength:plan.node_count * sizeof(float)
                    options:MTLResourceStorageModeShared];
    result.physical_query = [metal->device newBufferWithLength:tensor_bytes
                                                    options:MTLResourceStorageModePrivate];
    result.physical_key = [metal->device newBufferWithLength:tensor_bytes
                                                  options:MTLResourceStorageModePrivate];
    result.physical_value = [metal->device newBufferWithLength:tensor_bytes
                                                    options:MTLResourceStorageModePrivate];
    result.physical_output = [metal->device newBufferWithLength:tensor_bytes
                                                     options:MTLResourceStorageModePrivate];
    result.summary_key = [metal->device newBufferWithLength:summary_bytes
                                                options:MTLResourceStorageModePrivate];
    result.summary_value = [metal->device newBufferWithLength:summary_bytes
                                                  options:MTLResourceStorageModePrivate];
    result.lse = [metal->device
        newBufferWithLength:geometry.sequence_rows * 56u * sizeof(float)
                    options:MTLResourceStorageModePrivate];
    if (result.logical_to_physical == nil || result.route_offsets == nil ||
        result.route_entries == nil || result.query_blocks == nil ||
        result.nodes == nil || result.summary_log_counts == nil ||
        result.physical_query == nil || result.physical_key == nil ||
        result.physical_value == nil || result.physical_output == nil ||
        result.summary_key == nil || result.summary_value == nil ||
        result.lse == nil) {
        e2e_error(error, error_capacity,
                  "cannot allocate compiled H3 tree execution image");
        goto cleanup;
    }

    {
        h3_tree_node_gpu *gpu_nodes = result.nodes.contents;
        float *log_counts = result.summary_log_counts.contents;
        size_t physical_start = plan.video_start;
        for (size_t index = 0u; index < plan.node_count; ++index) {
            gpu_nodes[index] = (h3_tree_node_gpu) {
                .parent = nodes[index].parent,
                .first_child = nodes[index].first_child,
                .child_count = nodes[index].child_count,
                .kind = nodes[index].kind,
                .first_frame = nodes[index].first_frame,
                .frame_count = nodes[index].frame_count,
                .patch_y = nodes[index].patch_y,
                .patch_x = nodes[index].patch_x,
                .patch_h = nodes[index].patch_h,
                .patch_w = nodes[index].patch_w,
                .token_count = nodes[index].token_count,
                .physical_start = index < plan.leaf_count
                                      ? (uint32_t)physical_start
                                      : 0u,
            };
            if (index < plan.leaf_count)
                physical_start += nodes[index].token_count;
            log_counts[index] = logf((float)nodes[index].token_count);
        }
    }

    result.parameters = (h3_tree_parameters) {
        .sequence_rows = (uint32_t)geometry.sequence_rows,
        .exact_rows = (uint32_t)plan.exact_count,
        .video_start = (uint32_t)layout.video_start,
        .rows_per_video_frame = (uint32_t)geometry.rows_per_video_frame,
        .patch_columns = plan.patch_columns,
        .tile_columns = plan.tile_columns,
        .leaves_per_frame =
            (uint32_t)((size_t)plan.tile_rows * plan.tile_columns),
        .leaf_count = (uint32_t)plan.leaf_count,
    };
    result.query_block_count = (uint32_t)query_block_count;
    result.summary_node_count = (uint32_t)plan.node_count;
    result.leaf_count = (uint32_t)plan.leaf_count;
    result.frame_node_start = (uint32_t)plan.frame_node_start;
    result.frame_node_count = (uint32_t)plan.frame_node_count;
    result.temporal_node_start = (uint32_t)plan.temporal_node_start;
    result.temporal_node_count = (uint32_t)plan.temporal_node_count;
    result.root_index = (uint32_t)plan.root_index;
    result.leaf_summaries_only = frame_safe;
    result.approximate_step_mask = approximate_step_mask;
    result.approximate_layer_mask = approximate_layer_mask;
    result.enabled = 1;
    fprintf(stderr,
            "stage=tree-precompile profile=%s rows=%zu exact_rows=%zu nodes=%zu "
            "query_blocks=%zu route_entries=%zu max_route=%zu "
            "step_mask=0x%08x layer_mask=0x%013llx "
            "tensor_summary_bytes=%zu\n",
            frame_safe ? "quality" : "hierarchical", geometry.sequence_rows,
            plan.exact_count, plan.node_count,
            query_block_count, route_entry_count, maximum_route_entries,
            approximate_step_mask,
            (unsigned long long)approximate_layer_mask,
            tensor_bytes * 4u + summary_bytes * 2u);
    fflush(stderr);
    *runtime = result;
    status = 0;

cleanup:
    free(positions);
    free(tags);
    free(nodes);
    free(route_offsets);
    free(route_entries);
    free(logical_to_physical);
    free(query_blocks);
    return status;
}

static void h3_tree_attention_encode(
    h3_tree_runtime *tree,
    h3_metal *metal,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> query,
    id<MTLBuffer> key,
    id<MTLBuffer> value,
    id<MTLBuffer> output,
    uint32_t rows) {
    uint32_t elements = rows * 56u * 128u;
    const id<MTLBuffer> logical[3] = {query, key, value};
    const id<MTLBuffer> physical[3] = {
        tree->physical_query, tree->physical_key, tree->physical_value
    };
    [encoder setComputePipelineState:metal->h3_reorder_bf16_to_f16];
    for (unsigned tensor = 0u; tensor < 3u; ++tensor) {
        [encoder setBuffer:logical[tensor] offset:0 atIndex:0];
        [encoder setBuffer:tree->logical_to_physical offset:0 atIndex:1];
        [encoder setBuffer:physical[tensor] offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(elements, 1u, 1u)
                 threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    }

    h3_tree_parameters parameters = tree->parameters;
    [encoder setComputePipelineState:metal->h3_tree_leaf_summary];
    [encoder setBuffer:tree->physical_key offset:0 atIndex:0];
    [encoder setBuffer:tree->physical_value offset:0 atIndex:1];
    [encoder setBuffer:tree->nodes offset:0 atIndex:2];
    [encoder setBuffer:tree->summary_key offset:0 atIndex:4];
    [encoder setBuffer:tree->summary_value offset:0 atIndex:5];
    parameters.aggregate_start = 0u;
    parameters.aggregate_count = tree->leaf_count;
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake((size_t)tree->leaf_count * 56u,
                                               1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];

    if (!tree->leaf_summaries_only) {
        [encoder setComputePipelineState:metal->h3_tree_parent_summary];
        parameters.aggregate_start = tree->frame_node_start;
        parameters.aggregate_count = tree->frame_node_count;
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
        [encoder dispatchThreadgroups:
            MTLSizeMake((size_t)tree->frame_node_count * 56u, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        parameters.aggregate_start = tree->temporal_node_start;
        parameters.aggregate_count = tree->temporal_node_count;
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
        [encoder dispatchThreadgroups:
            MTLSizeMake((size_t)tree->temporal_node_count * 56u, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        parameters.aggregate_start = tree->root_index;
        parameters.aggregate_count = 1u;
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(56u, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    }

    parameters = tree->parameters;
    [encoder setComputePipelineState:metal->h3_tree_attention_mma64];
    [encoder setBuffer:tree->physical_query offset:0 atIndex:0];
    [encoder setBuffer:tree->physical_key offset:0 atIndex:1];
    [encoder setBuffer:tree->physical_value offset:0 atIndex:2];
    [encoder setBuffer:tree->summary_key offset:0 atIndex:3];
    [encoder setBuffer:tree->summary_value offset:0 atIndex:4];
    [encoder setBuffer:tree->summary_log_counts offset:0 atIndex:5];
    [encoder setBuffer:tree->route_offsets offset:0 atIndex:6];
    [encoder setBuffer:tree->route_entries offset:0 atIndex:7];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:8];
    [encoder setBuffer:tree->physical_output offset:0 atIndex:9];
    [encoder setBuffer:tree->query_blocks offset:0 atIndex:10];
    [encoder setBuffer:tree->lse offset:0 atIndex:11];
    [encoder dispatchThreadgroups:MTLSizeMake(tree->query_block_count, 56u,
                                               1u)
                threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];

    [encoder setComputePipelineState:metal->h3_reorder_f16_to_bf16];
    [encoder setBuffer:tree->physical_output offset:0 atIndex:0];
    [encoder setBuffer:tree->logical_to_physical offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(elements, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
}

static void h3_dense_attention_mma64(h3_metal *metal,
                                     h3_tree_runtime *tree,
                                     id<MTLComputeCommandEncoder> encoder,
                                     id<MTLBuffer> query,
                                     id<MTLBuffer> key,
                                     id<MTLBuffer> value,
                                     id<MTLBuffer> output,
                                     uint32_t rows,
                                     unsigned layer,
                                     uint32_t step) {
    const int approximate =
        tree != NULL && tree->enabled &&
        (!tree->leaf_summaries_only ||
         (step < 32u && layer < 64u &&
          (tree->approximate_step_mask & (UINT32_C(1) << step)) != 0u &&
          (tree->approximate_layer_mask & (UINT64_C(1) << layer)) != 0u));
    if (approximate) {
        h3_tree_attention_encode(tree, metal, encoder, query, key, value,
                                 output, rows);
        return;
    }
    const char *reference = getenv("MINIMAX_H3_REFERENCE_ATTENTION");
    if (reference != NULL && strcmp(reference, "1") == 0) {
        h3_dense_attention(metal, encoder, query, key, value, output, rows);
        return;
    }
    /* Single-pass online-softmax kernel: same mathematics as the
     * two-pass kernel, fp32 accumulation, bf16-rounding-level output
     * differences only; MINIMAX_H3_FLASH=0 restores the two-pass
     * kernel for comparison. */
    const char *flash = getenv("MINIMAX_H3_FLASH");
    [encoder setComputePipelineState:
        flash != NULL && strcmp(flash, "0") == 0
            ? metal->h3_attention_mma64_bf16_direct
            : metal->h3_attention_mma64_bf16_flash16];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key offset:0 atIndex:1];
    [encoder setBuffer:value offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake((rows + 63u) / 64u, 56u, 1u)
                threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];

}

static int h3_q8_linear_bound(h3_affine_binding binding,
                              h3_metal *metal,
                              id<MTLComputeCommandEncoder> encoder,
                              const char *prefix,
                              id<MTLBuffer> input,
                              id<MTLBuffer> output,
                              uint32_t batch,
                              char *error,
                              size_t error_capacity) {
    h3_tensor_binding weight = binding.weight;
    h3_tensor_binding scales = binding.scales;
    h3_tensor_binding biases = binding.biases;
    /* MLX packs four 8-bit values in each U32.  One affine metadata entry
     * covers 64 values, so one scale/bias group spans sixteen packed words. */
    if (strcmp(weight.tensor.dtype, "U32") != 0 || weight.tensor.rank != 2u ||
        weight.tensor.shape[1] % 16u != 0u ||
        strcmp(scales.tensor.dtype, "BF16") != 0 || scales.tensor.rank != 2u ||
        strcmp(biases.tensor.dtype, "BF16") != 0 || biases.tensor.rank != 2u ||
        scales.tensor.shape[0] != weight.tensor.shape[0] ||
        biases.tensor.shape[0] != weight.tensor.shape[0] ||
        scales.tensor.shape[1] != weight.tensor.shape[1] / 16u ||
        biases.tensor.shape[1] != scales.tensor.shape[1]) {
        e2e_error(error, error_capacity, "invalid affine-Q8 tensor: %s", prefix);
        return 1;
    }
    h3_gemm_parameters parameters = {
        .rows = (uint32_t)weight.tensor.shape[0],
        .groups_per_row = (uint32_t)scales.tensor.shape[1],
        .batch = batch,
    };
    [encoder setComputePipelineState:metal->q8];
    [encoder setBuffer:input offset:0 atIndex:0];
    h3_set(encoder, weight, 1);
    h3_set(encoder, scales, 2);
    h3_set(encoder, biases, 3);
    [encoder setBuffer:output offset:0 atIndex:4];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:5];
    [encoder dispatchThreadgroups:
        MTLSizeMake((parameters.rows + 63u) / 64u, (batch + 31u) / 32u, 1u)
            threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
    return 0;
}

static int h3_rms_plain_bound(h3_tensor_binding weight,
                              h3_metal *metal,
                              id<MTLComputeCommandEncoder> encoder,
                              id<MTLBuffer> input,
                              id<MTLBuffer> output,
                              uint32_t rows,
                              uint32_t columns) {
    h3_norm_parameters parameters = {
        .rows = rows,
        .columns = columns,
        .epsilon = columns == 5120u ? 1e-6f : 1e-5f,
    };
    [encoder setComputePipelineState:metal->rms_plain];
    [encoder setBuffer:input offset:0 atIndex:0];
    h3_set(encoder, weight, 1);
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    return 0;
}

static int h3_rms_plain(h3_remote_image *image,
                        h3_metal *metal,
                        id<MTLComputeCommandEncoder> encoder,
                        const char *weight_name,
                        id<MTLBuffer> input,
                        id<MTLBuffer> output,
                        uint32_t rows,
                        uint32_t columns,
                        char *error,
                        size_t error_capacity) {
    h3_tensor_binding weight;
    if (h3_bind_tensor(image, metal, weight_name, &weight, error,
                       error_capacity) != 0)
        return 1;
    h3_norm_parameters parameters = {
        .rows = rows,
        .columns = columns,
        .epsilon = 1e-5f,
    };
    [encoder setComputePipelineState:metal->rms_plain_bf16];
    [encoder setBuffer:input offset:0 atIndex:0];
    h3_set(encoder, weight, 1);
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    return 0;
}

static void h3_plain_residual_bf16(h3_metal *metal,
                                   id<MTLComputeCommandEncoder> encoder,
                                   id<MTLBuffer> residual,
                                   id<MTLBuffer> update,
                                   id<MTLBuffer> dummy_modulation,
                                   id<MTLBuffer> dummy_indices,
                                   uint32_t rows,
                                   uint32_t columns) {
    uint32_t zero = 0u;
    [encoder setComputePipelineState:metal->residual_bf16];
    [encoder setBuffer:residual offset:0 atIndex:0];
    [encoder setBuffer:update offset:0 atIndex:1];
    [encoder setBuffer:dummy_modulation offset:0 atIndex:2];
    [encoder setBuffer:dummy_indices offset:0 atIndex:3];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    [encoder setBytes:&columns length:sizeof(columns) atIndex:5];
    [encoder setBytes:&zero length:sizeof(zero) atIndex:6];
    [encoder setBytes:&zero length:sizeof(zero) atIndex:7];
    [encoder dispatchThreads:MTLSizeMake((size_t)rows * columns, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
}

static void h3_plain_residual(h3_metal *metal,
                              id<MTLComputeCommandEncoder> encoder,
                              id<MTLBuffer> residual,
                              id<MTLBuffer> update,
                              id<MTLBuffer> dummy_modulation,
                              id<MTLBuffer> dummy_indices,
                              uint32_t rows,
                              uint32_t columns) {
    uint32_t zero = 0u;
    [encoder setComputePipelineState:metal->residual];
    [encoder setBuffer:residual offset:0 atIndex:0];
    [encoder setBuffer:update offset:0 atIndex:1];
    [encoder setBuffer:dummy_modulation offset:0 atIndex:2];
    [encoder setBuffer:dummy_indices offset:0 atIndex:3];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    [encoder setBytes:&columns length:sizeof(columns) atIndex:5];
    [encoder setBytes:&zero length:sizeof(zero) atIndex:6];
    [encoder setBytes:&zero length:sizeof(zero) atIndex:7];
    [encoder dispatchThreads:MTLSizeMake((size_t)rows * columns, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
}

static uint16_t h3_float_to_half(float value) {
    __fp16 half = (__fp16)value;
    uint16_t bits;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

static float h3_bfloat(uint16_t bits) {
    uint32_t expanded = (uint32_t)bits << 16u;
    float value;
    memcpy(&value, &expanded, sizeof(value));
    return value;
}

static int h3_bfloat_buffer_is_finite(id<MTLBuffer> buffer,
                                      size_t count,
                                      size_t *first_bad,
                                      float *largest_magnitude) {
    const uint16_t *values = buffer.contents;
    float largest = 0.0f;
    for (size_t index = 0u; index < count; ++index) {
        float value = h3_bfloat(values[index]);
        if (!isfinite(value)) {
            if (first_bad != NULL) *first_bad = index;
            if (largest_magnitude != NULL) *largest_magnitude = largest;
            return 0;
        }
        largest = fmaxf(largest, fabsf(value));
    }
    if (first_bad != NULL) *first_bad = SIZE_MAX;
    if (largest_magnitude != NULL) *largest_magnitude = largest;
    return 1;
}

static int h3_f32_buffer_is_finite(id<MTLBuffer> buffer,
                                   size_t count,
                                   size_t *first_bad,
                                   float *largest_magnitude) {
    const float *values = buffer.contents;
    float largest = 0.0f;
    for (size_t index = 0u; index < count; ++index) {
        float value = values[index];
        if (!isfinite(value)) {
            if (first_bad != NULL) *first_bad = index;
            if (largest_magnitude != NULL) *largest_magnitude = largest;
            return 0;
        }
        largest = fmaxf(largest, fabsf(value));
    }
    if (first_bad != NULL) *first_bad = SIZE_MAX;
    if (largest_magnitude != NULL) *largest_magnitude = largest;
    return 1;
}

static int h3_text_encode(const uint32_t *token_ids,
                          size_t token_count,
                          h3_metal *metal,
                          uint16_t **states,
                          double *download_seconds,
                          double *encode_seconds,
                          size_t *peak_footprint,
                          char *error,
                          size_t error_capacity) {
    minimax_h3_remote_safetensors remote = {0};
    char url[2300];
    const char *text_encoder_url = getenv("MINIMAX_H3_TEXT_ENCODER_URL");
    if (text_encoder_url != NULL &&
        strncmp(text_encoder_url, "file://", 7u) == 0) {
        if (snprintf(url, sizeof(url), "%s", text_encoder_url) >=
            (int)sizeof(url)) {
            e2e_error(error, error_capacity,
                      "local text encoder path is too long");
            return 1;
        }
    } else {
        e2e_error(error, error_capacity,
                  "MINIMAX_H3_TEXT_ENCODER_URL must be a local file:// path; inference is offline");
        return 1;
    }
    *download_seconds = 0.0;
    *encode_seconds = 0.0;
    double header_started = e2e_now();
    if (minimax_h3_remote_safetensors_open(url, &remote, error,
                                            error_capacity) != 0)
        return 1;
    *download_seconds += e2e_now() - header_started;
    size_t padded = (token_count + 31u) & ~(size_t)31u;
    size_t hidden_bytes = padded * 5120u * 2u;
    id<MTLBuffer> hidden = [metal->device newBufferWithLength:hidden_bytes
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> normalized = [metal->device newBufferWithLength:hidden_bytes
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> q_projected = [metal->device
        newBufferWithLength:padded * 8192u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> k_projected = [metal->device
        newBufferWithLength:padded * 1024u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> v_projected = [metal->device
        newBufferWithLength:padded * 1024u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> query = [metal->device
        newBufferWithLength:padded * 8192u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> key = [metal->device
        newBufferWithLength:padded * 1024u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> value = [metal->device
        newBufferWithLength:padded * 1024u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> attended = [metal->device
        newBufferWithLength:padded * 8192u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> attention_output = [metal->device
        newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> gate = [metal->device
        newBufferWithLength:padded * 25600u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> up = [metal->device
        newBufferWithLength:padded * 25600u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> gated = [metal->device
        newBufferWithLength:padded * 25600u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> mlp_output = [metal->device
        newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dummy = [metal->device newBufferWithLength:4u
                                                   options:MTLResourceStorageModeShared];
    if (hidden == nil || normalized == nil || q_projected == nil ||
        k_projected == nil || v_projected == nil || query == nil || key == nil ||
        value == nil || attended == nil || attention_output == nil || gate == nil ||
        up == nil || gated == nil || mlp_output == nil || dummy == nil) {
        e2e_error(error, error_capacity, "text activation allocation failed");
        minimax_h3_remote_safetensors_close(&remote);
        return 1;
    }
    memset(hidden.contents, 0, hidden_bytes);
    memset(dummy.contents, 0, dummy.length);
    minimax_h3_remote_tensor embedding;
    if (minimax_h3_remote_safetensors_find(
            &remote, "model.embed_tokens.weight", &embedding, error,
            error_capacity) != 0 || embedding.rank != 2u ||
        embedding.shape[0] != 151936u || embedding.shape[1] != 5120u ||
        strcmp(embedding.dtype, "BF16") != 0) {
        minimax_h3_remote_safetensors_close(&remote);
        return 1;
    }
    uint16_t *hidden_values = hidden.contents;
    uint16_t embedding_row[5120];
    for (size_t token = 0u; token < token_count; ++token) {
        if (token_ids[token] >= embedding.shape[0]) {
            e2e_error(error, error_capacity, "token id outside H3 vocabulary");
            minimax_h3_remote_safetensors_close(&remote);
            return 1;
        }
        double row_download_started = e2e_now();
        if (minimax_h3_remote_safetensors_read(
                &remote,
                embedding.data_start + (uint64_t)token_ids[token] *
                                               5120u * sizeof(uint16_t),
                embedding_row, sizeof(embedding_row), error,
                error_capacity) != 0) {
            minimax_h3_remote_safetensors_close(&remote);
            return 1;
        }
        *download_seconds += e2e_now() - row_download_started;
        double row_convert_started = e2e_now();
        for (size_t column = 0u; column < 5120u; ++column)
            hidden_values[token * 5120u + column] =
                h3_float_to_half(h3_bfloat(embedding_row[column]));
        *encode_seconds += e2e_now() - row_convert_started;
    }
    for (unsigned layer = 0u; layer < 50u; ++layer) {
        @autoreleasepool {
            h3_tensor_binding input_norm = {0};
            h3_tensor_binding post_norm = {0};
            h3_tensor_binding q_norm = {0};
            h3_tensor_binding k_norm = {0};
            h3_affine_binding q_projection = {0};
            h3_affine_binding k_projection = {0};
            h3_affine_binding v_projection = {0};
            h3_affine_binding o_projection = {0};
            h3_affine_binding gate_projection = {0};
            h3_affine_binding up_projection = {0};
            h3_affine_binding down_projection = {0};
#define H3_QUEUE_QWEN_TENSOR(target, format)                                  \
    do {                                                                      \
        h3_fetch_job *job = &jobs[job_count++];                              \
        snprintf(job->name, sizeof(job->name), format, layer);               \
        job->binding = &target;                                               \
    } while (0)
            h3_fetch_job jobs[25] = {0};
            size_t job_count = 0u;
            H3_QUEUE_QWEN_TENSOR(
                input_norm, "model.layers.%u.input_layernorm.weight");
            H3_QUEUE_QWEN_TENSOR(
                post_norm, "model.layers.%u.post_attention_layernorm.weight");
            H3_QUEUE_QWEN_TENSOR(
                q_norm, "model.layers.%u.self_attn.q_norm.weight");
            H3_QUEUE_QWEN_TENSOR(
                k_norm, "model.layers.%u.self_attn.k_norm.weight");
#define H3_QUEUE_QWEN_AFFINE(target, format)                                  \
    do {                                                                      \
        H3_QUEUE_QWEN_TENSOR(target.weight, format ".weight");              \
        H3_QUEUE_QWEN_TENSOR(target.scales, format ".scales");              \
        H3_QUEUE_QWEN_TENSOR(target.biases, format ".biases");              \
    } while (0)
            H3_QUEUE_QWEN_AFFINE(
                q_projection, "model.layers.%u.self_attn.q_proj");
            H3_QUEUE_QWEN_AFFINE(
                k_projection, "model.layers.%u.self_attn.k_proj");
            H3_QUEUE_QWEN_AFFINE(
                v_projection, "model.layers.%u.self_attn.v_proj");
            H3_QUEUE_QWEN_AFFINE(
                o_projection, "model.layers.%u.self_attn.o_proj");
            H3_QUEUE_QWEN_AFFINE(
                gate_projection, "model.layers.%u.mlp.gate_proj");
            H3_QUEUE_QWEN_AFFINE(
                up_projection, "model.layers.%u.mlp.up_proj");
            H3_QUEUE_QWEN_AFFINE(
                down_projection, "model.layers.%u.mlp.down_proj");
#undef H3_QUEUE_QWEN_AFFINE
#undef H3_QUEUE_QWEN_TENSOR
            if (job_count != 25u ||
                h3_fetch_batch_run(&remote, metal, jobs, job_count,
                                   download_seconds, error,
                                   error_capacity) != 0) {
                minimax_h3_remote_safetensors_close(&remote);
                return 1;
            }
            *peak_footprint = MAX(*peak_footprint, e2e_footprint());
            double layer_encode_started = e2e_now();

            double stage_started = e2e_now();
            id<MTLCommandBuffer> command = [metal->queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            h3_rms_plain_bound(input_norm, metal, encoder, hidden, normalized,
                               (uint32_t)token_count, 5120u);
            int stage_status =
                h3_q8_linear_bound(q_projection, metal, encoder, "q_proj",
                                    normalized, q_projected,
                                    (uint32_t)token_count, error,
                                    error_capacity) != 0 ||
                h3_q8_linear_bound(k_projection, metal, encoder, "k_proj",
                                    normalized, k_projected,
                                    (uint32_t)token_count, error,
                                    error_capacity) != 0 ||
                h3_q8_linear_bound(v_projection, metal, encoder, "v_proj",
                                    normalized, v_projected,
                                    (uint32_t)token_count, error,
                                    error_capacity) != 0;
            [encoder endEncoding];
            if (stage_status || h3_wait(command, error, error_capacity) != 0) {
                minimax_h3_remote_safetensors_close(&remote);
                return 1;
            }
            double qkv_seconds = e2e_now() - stage_started;

            h3_qwen_attention_parameters attention_parameters = {
                .token_count = (uint32_t)token_count,
                .query_heads = 64u,
                .key_value_heads = 8u,
                .head_dimension = 128u,
                .scale = 0.08838834764831845f,
            };
            stage_started = e2e_now();
            command = [metal->queue commandBuffer];
            encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:metal->qwen_prepare];
            [encoder setBuffer:q_projected offset:0 atIndex:0];
            [encoder setBuffer:k_projected offset:0 atIndex:1];
            [encoder setBuffer:v_projected offset:0 atIndex:2];
            h3_set(encoder, q_norm, 3);
            h3_set(encoder, k_norm, 4);
            [encoder setBuffer:query offset:0 atIndex:5];
            [encoder setBuffer:key offset:0 atIndex:6];
            [encoder setBuffer:value offset:0 atIndex:7];
            [encoder setBytes:&attention_parameters
                        length:sizeof(attention_parameters) atIndex:8];
            [encoder dispatchThreadgroups:
                MTLSizeMake(token_count, 72u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
            [encoder setComputePipelineState:metal->qwen_attention];
            [encoder setBuffer:query offset:0 atIndex:0];
            [encoder setBuffer:key offset:0 atIndex:1];
            [encoder setBuffer:value offset:0 atIndex:2];
            [encoder setBuffer:attended offset:0 atIndex:3];
            [encoder setBytes:&attention_parameters
                        length:sizeof(attention_parameters) atIndex:4];
            [encoder dispatchThreadgroups:
                MTLSizeMake((token_count * 64u + 3u) / 4u, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
            if (h3_q8_linear_bound(o_projection, metal, encoder, "o_proj",
                                    attended, attention_output,
                                    (uint32_t)token_count, error,
                                    error_capacity) != 0) {
                [encoder endEncoding];
                minimax_h3_remote_safetensors_close(&remote);
                return 1;
            }
            h3_plain_residual(metal, encoder, hidden, attention_output, dummy,
                              dummy, (uint32_t)token_count, 5120u);
            [encoder endEncoding];
            if (h3_wait(command, error, error_capacity) != 0) {
                minimax_h3_remote_safetensors_close(&remote);
                return 1;
            }
            double attention_seconds = e2e_now() - stage_started;

            stage_started = e2e_now();
            command = [metal->queue commandBuffer];
            encoder = [command computeCommandEncoder];
            h3_rms_plain_bound(post_norm, metal, encoder, hidden, normalized,
                               (uint32_t)token_count, 5120u);
            stage_status =
                h3_q8_linear_bound(gate_projection, metal, encoder,
                                    "gate_proj", normalized, gate,
                                    (uint32_t)token_count, error,
                                    error_capacity) != 0 ||
                h3_q8_linear_bound(up_projection, metal, encoder, "up_proj",
                                    normalized, up, (uint32_t)token_count,
                                    error, error_capacity) != 0;
            uint32_t element_count = (uint32_t)(token_count * 25600u);
            [encoder setComputePipelineState:metal->silu_pair];
            [encoder setBuffer:gate offset:0 atIndex:0];
            [encoder setBuffer:up offset:0 atIndex:1];
            [encoder setBuffer:gated offset:0 atIndex:2];
            [encoder setBytes:&element_count length:sizeof(element_count)
                       atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(element_count, 1u, 1u)
                 threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
            stage_status = stage_status ||
                h3_q8_linear_bound(down_projection, metal, encoder,
                                    "down_proj", gated, mlp_output,
                                    (uint32_t)token_count, error,
                                    error_capacity) != 0;
            h3_plain_residual(metal, encoder, hidden, mlp_output, dummy, dummy,
                              (uint32_t)token_count, 5120u);
            [encoder endEncoding];
            if (stage_status || h3_wait(command, error, error_capacity) != 0) {
                minimax_h3_remote_safetensors_close(&remote);
                return 1;
            }
            double mlp_seconds = e2e_now() - stage_started;
            *encode_seconds += e2e_now() - layer_encode_started;
            fprintf(stderr,
                    "stage=text-layer layer=%u/50 qkv_seconds=%.6f "
                    "attention_seconds=%.6f mlp_seconds=%.6f\n",
                    layer + 1u, qkv_seconds, attention_seconds, mlp_seconds);
            fflush(stderr);
        }
        fprintf(stderr,
                "stage=text-encode layer=%u/50 download_seconds=%.6f "
                "encode_seconds=%.6f footprint=%zu\n",
                layer + 1u, *download_seconds, *encode_seconds,
                e2e_footprint());
        fflush(stderr);
    }
    *states = malloc(token_count * 5120u * 2u);
    if (*states == NULL) {
        e2e_error(error, error_capacity, "cannot allocate text-state output");
        minimax_h3_remote_safetensors_close(&remote);
        return 1;
    }
    memcpy(*states, hidden.contents, token_count * 5120u * 2u);
    minimax_h3_remote_safetensors_close(&remote);
    const char *delete_after_load =
        getenv("MINIMAX_H3_TEXT_ENCODER_DELETE_AFTER_LOAD");
    if (delete_after_load != NULL && strcmp(delete_after_load, "1") == 0 &&
        strncmp(url, "file://", 7u) == 0 && unlink(url + 7u) != 0) {
        free(*states);
        *states = NULL;
        e2e_error(error, error_capacity,
                  "cannot remove consumed text encoder cache: %s",
                  strerror(errno));
        return 1;
    }
    return 0;
}

static int h3_cached_residual_bound(h3_tensor_binding modulation,
                                    h3_metal *metal,
                                    id<MTLComputeCommandEncoder> encoder,
                                    uint32_t step,
                                    uint32_t gate_offset,
                                    id<MTLBuffer> row_indices,
                                    id<MTLBuffer> residual,
                                    id<MTLBuffer> update,
                                    uint32_t rows,
                                    char *error,
                                    size_t error_capacity) {
    const uint32_t stride = 6u * 5376u;
    NSUInteger step_bytes = (NSUInteger)step * 9u * stride * 2u;
    if ((uint64_t)step_bytes + UINT64_C(9) * stride * 2u >
        modulation.tensor.data_length) {
        e2e_error(error, error_capacity,
                  "gate step outside compiled modulation tensor");
        return 1;
    }
    [encoder setComputePipelineState:metal->residual_bf16];
    [encoder setBuffer:residual offset:0 atIndex:0];
    [encoder setBuffer:update offset:0 atIndex:1];
    h3_set_at(encoder, modulation, step_bytes, 2);
    [encoder setBuffer:row_indices offset:0 atIndex:3];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    uint32_t columns = 5376u;
    [encoder setBytes:&columns length:sizeof(columns) atIndex:5];
    [encoder setBytes:&stride length:sizeof(stride) atIndex:6];
    [encoder setBytes:&gate_offset length:sizeof(gate_offset) atIndex:7];
    [encoder dispatchThreads:MTLSizeMake((size_t)rows * columns, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    return 0;
}

static int h3_prepare_attention(h3_remote_image *weights,
                                h3_metal *metal,
                                id<MTLComputeCommandEncoder> encoder,
                                const char *prefix,
                                id<MTLBuffer> projected,
                                id<MTLBuffer> rotary,
                                uint32_t apply_rotary,
                                id<MTLBuffer> query,
                                id<MTLBuffer> key,
                                id<MTLBuffer> value,
                                uint32_t rows,
                                char *error,
                                size_t error_capacity) {
    char name[192];
    h3_tensor_binding q_norm = { {0}, nil, 0u };
    h3_tensor_binding k_norm = { {0}, nil, 0u };
    snprintf(name, sizeof(name), "%s.q_norm.weight", prefix);
    if (h3_bind_tensor(weights, metal, name, &q_norm, error,
                       error_capacity) != 0)
        return 1;
    snprintf(name, sizeof(name), "%s.k_norm.weight", prefix);
    if (h3_bind_tensor(weights, metal, name, &k_norm, error,
                       error_capacity) != 0)
        return 1;
    [encoder setComputePipelineState:metal->h3_prepare_bf16];
    [encoder setBuffer:projected offset:0 atIndex:0];
    h3_set(encoder, q_norm, 1);
    h3_set(encoder, k_norm, 2);
    [encoder setBuffer:rotary offset:0 atIndex:3];
    [encoder setBuffer:query offset:0 atIndex:4];
    [encoder setBuffer:key offset:0 atIndex:5];
    [encoder setBuffer:value offset:0 atIndex:6];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:7];
    [encoder setBytes:&apply_rotary length:sizeof(apply_rotary) atIndex:8];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 112u, 1u)
                threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    return 0;
}

static int h3_plain_block(h3_remote_image *weights,
                          h3_remote_image *adapter,
                          h3_metal *metal,
                          const char *prefix,
                          id<MTLBuffer> hidden,
                          id<MTLBuffer> normalized,
                          id<MTLBuffer> qkv,
                          id<MTLBuffer> query,
                          id<MTLBuffer> key,
                          id<MTLBuffer> value,
                          id<MTLBuffer> attended,
                          id<MTLBuffer> attention_output,
                          id<MTLBuffer> lora_scratch,
                          id<MTLBuffer> fc1,
                          id<MTLBuffer> gated,
                          id<MTLBuffer> mlp_output,
                          id<MTLBuffer> positions,
                          id<MTLBuffer> zero,
                          uint32_t rows,
                          char *error,
                          size_t error_capacity) {
    char name[192];
    char projection[192];
    id<MTLCommandBuffer> command = [metal->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    snprintf(name, sizeof(name), "%s.norm1.weight", prefix);
    if (h3_rms_plain(weights, metal, encoder, name, hidden, normalized, rows,
                     5376u, error, error_capacity) != 0)
        goto failed;
    snprintf(projection, sizeof(projection), "%s.attn.qkv_proj", prefix);
    if (h3_q4_linear_lora(weights, adapter, metal, encoder, projection,
                          normalized, lora_scratch, qkv, rows, error,
                          error_capacity) != 0)
        goto failed;
    snprintf(projection, sizeof(projection), "%s.attn", prefix);
    if (h3_prepare_attention(weights, metal, encoder, projection, qkv,
                             positions, 0u, query, key, value, rows, error,
                             error_capacity) != 0)
        goto failed;
    h3_dense_attention(metal, encoder, query, key, value, attended, rows);
    snprintf(projection, sizeof(projection), "%s.attn.out_proj", prefix);
    if (h3_q4_linear_lora(weights, adapter, metal, encoder, projection,
                          attended, lora_scratch, attention_output, rows,
                          error, error_capacity) != 0)
        goto failed;
    h3_plain_residual_bf16(metal, encoder, hidden, attention_output, zero,
                           zero, rows, 5376u);
    snprintf(name, sizeof(name), "%s.norm2.weight", prefix);
    if (h3_rms_plain(weights, metal, encoder, name, hidden, normalized, rows,
                     5376u, error, error_capacity) != 0)
        goto failed;
    snprintf(projection, sizeof(projection), "%s.mlp.fc1", prefix);
    if (h3_q4_linear_lora(weights, adapter, metal, encoder, projection,
                          normalized, lora_scratch, fc1, rows, error,
                          error_capacity) != 0)
        goto failed;
    uint32_t width = 14336u;
    [encoder setComputePipelineState:metal->silu_split_bf16];
    [encoder setBuffer:fc1 offset:0 atIndex:0];
    [encoder setBuffer:gated offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
    [encoder setBytes:&width length:sizeof(width) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake((size_t)rows * width, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    snprintf(projection, sizeof(projection), "%s.mlp.fc2", prefix);
    if (h3_q4_linear_lora(weights, adapter, metal, encoder, projection, gated,
                          lora_scratch, mlp_output, rows, error,
                          error_capacity) != 0)
        goto failed;
    h3_plain_residual_bf16(metal, encoder, hidden, mlp_output, zero, zero,
                           rows, 5376u);
    [encoder endEncoding];
    return h3_wait(command, error, error_capacity);
failed:
    [encoder endEncoding];
    return 1;
}

static int h3_denoise_block(h3_remote_image *weights,
                            h3_remote_image *adapter,
                            h3_tensor_binding modulation,
                            h3_metal *metal,
                            h3_tree_runtime *tree,
                            id<MTLComputeCommandEncoder> encoder,
                            unsigned layer,
                            uint32_t step,
                            id<MTLBuffer> row_indices,
                            id<MTLBuffer> hidden,
                            id<MTLBuffer> normalized,
                            id<MTLBuffer> qkv,
                            id<MTLBuffer> query,
                            id<MTLBuffer> key,
                            id<MTLBuffer> value,
                            id<MTLBuffer> attended,
                            id<MTLBuffer> attention_output,
                            id<MTLBuffer> lora_scratch,
                            id<MTLBuffer> fc1,
                            id<MTLBuffer> gated,
                            id<MTLBuffer> mlp_output,
                            id<MTLBuffer> positions,
                            uint32_t rows,
                            char *error,
                            size_t error_capacity) {
    char prefix[192];
    char name[192];
    snprintf(prefix, sizeof(prefix), "blocks.%u", layer);

    snprintf(name, sizeof(name), "%s.norm1.weight", prefix);
    if (h3_rms_adaln_bound(weights, metal, encoder, name, modulation, step,
                           9u, 6u * 5376u, 0u, 5376u, row_indices, hidden,
                           normalized, rows, error, error_capacity) != 0)
        return 1;
    snprintf(name, sizeof(name), "%s.attn.qkv_proj", prefix);
    if (h3_q4_linear_lora(weights, adapter, metal, encoder, name, normalized,
                          lora_scratch, qkv, rows, error,
                          error_capacity) != 0)
        return 1;
    snprintf(name, sizeof(name), "%s.attn", prefix);
    if (h3_prepare_attention(weights, metal, encoder, name, qkv, positions,
                             1u, query, key, value, rows, error,
                             error_capacity) != 0)
        return 1;

    h3_dense_attention_mma64(metal, tree, encoder, query, key, value,
                             attended, rows, layer, step);
    snprintf(name, sizeof(name), "%s.attn.out_proj", prefix);
    if (h3_q4_linear_lora(weights, adapter, metal, encoder, name, attended,
                          lora_scratch, attention_output, rows, error,
                          error_capacity) != 0)
        return 1;
    if (h3_cached_residual_bound(modulation, metal, encoder, step,
                                 2u * 5376u, row_indices, hidden,
                                 attention_output, rows, error,
                                 error_capacity) != 0)
        return 1;

    snprintf(name, sizeof(name), "%s.norm2.weight", prefix);
    if (h3_rms_adaln_bound(weights, metal, encoder, name, modulation, step,
                           9u, 6u * 5376u, 3u * 5376u, 4u * 5376u,
                           row_indices, hidden, normalized, rows, error,
                           error_capacity) != 0)
        return 1;
    snprintf(name, sizeof(name), "%s.mlp.fc1", prefix);
    if (h3_q4_linear_lora(weights, adapter, metal, encoder, name, normalized,
                          lora_scratch, fc1, rows, error,
                          error_capacity) != 0)
        return 1;
    uint32_t width = 14336u;
    [encoder setComputePipelineState:metal->silu_split_bf16];
    [encoder setBuffer:fc1 offset:0 atIndex:0];
    [encoder setBuffer:gated offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
    [encoder setBytes:&width length:sizeof(width) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake((size_t)rows * width, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    snprintf(name, sizeof(name), "%s.mlp.fc2", prefix);
    if (h3_q4_linear_lora(weights, adapter, metal, encoder, name, gated,
                          lora_scratch, mlp_output, rows, error,
                          error_capacity) != 0)
        return 1;
    if (h3_cached_residual_bound(modulation, metal, encoder, step,
                                 5u * 5376u, row_indices, hidden, mlp_output,
                                 rows, error, error_capacity) != 0)
        return 1;
    return 0;
}

static uint64_t h3_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12u;
    x ^= x << 25u;
    x ^= x >> 27u;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static float h3_rng_normal(uint64_t *state) {
    double a = ((h3_rng_next(state) >> 11u) + 1.0) /
               9007199254740993.0;
    double b = ((h3_rng_next(state) >> 11u) + 1.0) /
               9007199254740993.0;
    return (float)(sqrt(-2.0 * log(a)) *
                   cos(6.28318530717958647692 * b));
}

enum {
    H3_TURBO_STEPS = 4,
    H3_TURBO_TIMESTAMPS = 3,
    H3_TURBO_MODALITY_ROWS = 9,
    H3_TURBO_ADALN_WIDTH = 32256,
    H3_TURBO_TIME_WIDTH = 2688,
};

typedef struct {
    uint32_t lower_step;
    uint32_t lower_kind;
    uint32_t upper_step;
    uint32_t upper_kind;
    float fraction;
} h3_turbo_interpolation;

typedef struct {
    h3_tensor_binding blocks[50];
    h3_tensor_binding final;
    float video_sigmas[H3_TURBO_STEPS + 1];
    float audio_sigmas[H3_TURBO_STEPS + 1];
    double compile_seconds;
} h3_turbo_cache;

static uint16_t h3_float_to_bfloat(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t rounding = UINT32_C(0x7fff) + ((bits >> 16u) & 1u);
    return (uint16_t)((bits + rounding) >> 16u);
}

static const void *h3_cpu_tensor(h3_remote_image *image,
                                 const char *name,
                                 minimax_h3_remote_tensor *tensor,
                                 char *error,
                                 size_t error_capacity) {
    if (minimax_h3_remote_safetensors_find(&image->remote, name, tensor,
                                            error, error_capacity) != 0)
        return NULL;
    uint64_t offset = (uint64_t)image->file_padding + tensor->data_start;
    if (offset > image->mapping_bytes ||
        tensor->data_length > image->mapping_bytes - offset) {
        e2e_error(error, error_capacity, "tensor exceeds local artifact: %s",
                  name);
        return NULL;
    }
    return (const uint8_t *)image->mapping + (size_t)offset;
}

static float h3_shifted_sigma(float base, float shift) {
    return shift * base / (1.0f + (shift - 1.0f) * base);
}

static h3_turbo_interpolation h3_turbo_interpolation_make(
    const float source_video_sigmas[31],
    const float source_audio_sigmas[31],
    float target) {
    h3_turbo_interpolation result = {0};
    float lower_value = -INFINITY;
    float upper_value = INFINITY;
    for (uint32_t step = 0u; step < 30u; ++step) {
        float candidates[3] = {
            1.0f - source_video_sigmas[step],
            1.0f - source_audio_sigmas[step],
            fmaxf(1.0f - source_video_sigmas[step], 0.999f),
        };
        for (uint32_t kind = 0u; kind < 3u; ++kind) {
            float value = candidates[kind];
            if (value <= target && value > lower_value) {
                lower_value = value;
                result.lower_step = step;
                result.lower_kind = kind;
            }
            if (value >= target && value < upper_value) {
                upper_value = value;
                result.upper_step = step;
                result.upper_kind = kind;
            }
        }
    }
    if (!isfinite(lower_value)) {
        result.lower_step = result.upper_step;
        result.lower_kind = result.upper_kind;
        lower_value = upper_value;
    }
    if (!isfinite(upper_value)) {
        result.upper_step = result.lower_step;
        result.upper_kind = result.lower_kind;
        upper_value = lower_value;
    }
    result.fraction = upper_value == lower_value
                          ? 0.0f
                          : (target - lower_value) /
                                (upper_value - lower_value);
    return result;
}

static float h3_interpolate_f32(const float *source,
                                size_t row_width,
                                h3_turbo_interpolation interpolation,
                                size_t column) {
    size_t lower = ((size_t)interpolation.lower_step * 3u +
                    interpolation.lower_kind) * row_width + column;
    size_t upper = ((size_t)interpolation.upper_step * 3u +
                    interpolation.upper_kind) * row_width + column;
    return source[lower] +
           (source[upper] - source[lower]) * interpolation.fraction;
}

static float h3_interpolate_bf16(const uint16_t *source,
                                 size_t step_rows,
                                 size_t row_width,
                                 h3_turbo_interpolation interpolation,
                                 size_t row_in_kind,
                                 size_t column) {
    size_t lower = ((size_t)interpolation.lower_step * step_rows +
                    interpolation.lower_kind * 3u + row_in_kind) *
                       row_width +
                   column;
    size_t upper = ((size_t)interpolation.upper_step * step_rows +
                    interpolation.upper_kind * 3u + row_in_kind) *
                       row_width +
                   column;
    float a = h3_bfloat(source[lower]);
    float b = h3_bfloat(source[upper]);
    return a + (b - a) * interpolation.fraction;
}

static int h3_turbo_pair(h3_remote_image *adapter,
                         const char *prefix,
                         uint32_t rank,
                         uint32_t input_width,
                         uint32_t output_width,
                         const uint16_t **down,
                         const uint16_t **up,
                         char *error,
                         size_t error_capacity) {
    char name[224];
    minimax_h3_remote_tensor down_tensor;
    minimax_h3_remote_tensor up_tensor;
    snprintf(name, sizeof(name), "%s.lora_A.weight", prefix);
    *down = h3_cpu_tensor(adapter, name, &down_tensor, error, error_capacity);
    if (*down == NULL) return 1;
    snprintf(name, sizeof(name), "%s.lora_B.weight", prefix);
    *up = h3_cpu_tensor(adapter, name, &up_tensor, error, error_capacity);
    if (*up == NULL) return 1;
    if (strcmp(down_tensor.dtype, "BF16") != 0 || down_tensor.rank != 2u ||
        down_tensor.shape[0] != rank ||
        down_tensor.shape[1] != input_width ||
        strcmp(up_tensor.dtype, "BF16") != 0 || up_tensor.rank != 2u ||
        up_tensor.shape[0] != output_width ||
        up_tensor.shape[1] != rank) {
        e2e_error(error, error_capacity,
                  "unexpected Turbo AdaLN LoRA shape: %s", prefix);
        return 1;
    }
    return 0;
}

static int h3_turbo_cache_build(h3_remote_image *source_cache,
                                h3_remote_image *adapter,
                                h3_metal *metal,
                                h3_turbo_cache *cache,
                                char *error,
                                size_t error_capacity) {
    double started = e2e_now();
    minimax_h3_remote_tensor tensor;
    const float *source_video = h3_cpu_tensor(
        source_cache, "video_sigmas", &tensor, error, error_capacity);
    if (source_video == NULL || strcmp(tensor.dtype, "F32") != 0 ||
        tensor.rank != 1u || tensor.shape[0] != 31u)
        return 1;
    const float *source_audio = h3_cpu_tensor(
        source_cache, "audio_sigmas", &tensor, error, error_capacity);
    if (source_audio == NULL || strcmp(tensor.dtype, "F32") != 0 ||
        tensor.rank != 1u || tensor.shape[0] != 31u)
        return 1;
    const float *source_time = h3_cpu_tensor(
        source_cache, "time_embeddings", &tensor, error, error_capacity);
    if (source_time == NULL || strcmp(tensor.dtype, "F32") != 0 ||
        tensor.rank != 3u || tensor.shape[0] != 30u ||
        tensor.shape[1] != 3u || tensor.shape[2] != H3_TURBO_TIME_WIDTH)
        return 1;

    h3_turbo_interpolation interpolation[H3_TURBO_STEPS]
                                              [H3_TURBO_TIMESTAMPS];
    uint16_t activated[H3_TURBO_STEPS][H3_TURBO_TIMESTAMPS]
                      [H3_TURBO_TIME_WIDTH];
    for (uint32_t point = 0u; point <= H3_TURBO_STEPS; ++point) {
        float base = (float)(H3_TURBO_STEPS - point) /
                     (float)H3_TURBO_STEPS;
        cache->video_sigmas[point] = h3_shifted_sigma(base, 12.0f);
        cache->audio_sigmas[point] = h3_shifted_sigma(base, 3.0f);
    }
    for (uint32_t step = 0u; step < H3_TURBO_STEPS; ++step) {
        float targets[3] = {
            1.0f - cache->video_sigmas[step],
            1.0f - cache->audio_sigmas[step],
            fmaxf(1.0f - cache->video_sigmas[step], 0.999f),
        };
        for (uint32_t kind = 0u; kind < 3u; ++kind) {
            interpolation[step][kind] = h3_turbo_interpolation_make(
                source_video, source_audio, targets[kind]);
            for (uint32_t column = 0u; column < H3_TURBO_TIME_WIDTH;
                 ++column) {
                float value = h3_interpolate_f32(
                    source_time, H3_TURBO_TIME_WIDTH,
                    interpolation[step][kind], column);
                activated[step][kind][column] =
                    h3_float_to_bfloat(value / (1.0f + expf(-value)));
            }
        }
    }

    for (uint32_t layer = 0u; layer < 50u; ++layer) {
        char name[128];
        snprintf(name, sizeof(name), "blocks.%u.modulations", layer);
        const uint16_t *source = h3_cpu_tensor(
            source_cache, name, &tensor, error, error_capacity);
        if (source == NULL || strcmp(tensor.dtype, "BF16") != 0 ||
            tensor.rank != 3u || tensor.shape[0] != 30u ||
            tensor.shape[1] != H3_TURBO_MODALITY_ROWS ||
            tensor.shape[2] != H3_TURBO_ADALN_WIDTH)
            return 1;
        size_t output_count = (size_t)H3_TURBO_STEPS *
                              H3_TURBO_MODALITY_ROWS *
                              H3_TURBO_ADALN_WIDTH;
        id<MTLBuffer> output = [metal->device
            newBufferWithLength:output_count * sizeof(uint16_t)
                        options:MTLResourceStorageModeShared];
        if (output == nil) {
            e2e_error(error, error_capacity,
                      "cannot allocate Turbo AdaLN block %u", layer);
            return 1;
        }
        uint16_t *destination = output.contents;
        for (uint32_t step = 0u; step < H3_TURBO_STEPS; ++step)
            for (uint32_t kind = 0u; kind < 3u; ++kind)
                for (uint32_t modality = 0u; modality < 3u; ++modality)
                    for (uint32_t column = 0u;
                         column < H3_TURBO_ADALN_WIDTH; ++column) {
                        size_t output_index =
                            ((size_t)step * 9u + kind * 3u + modality) *
                                H3_TURBO_ADALN_WIDTH +
                            column;
                        destination[output_index] = h3_float_to_bfloat(
                            h3_interpolate_bf16(
                                source, 9u, H3_TURBO_ADALN_WIDTH,
                                interpolation[step][kind], modality,
                                column));
                    }

        snprintf(name, sizeof(name), "blocks.%u.adaln_proj.linear", layer);
        const uint16_t *down;
        const uint16_t *up;
        if (h3_turbo_pair(adapter, name, 16u, H3_TURBO_TIME_WIDTH,
                          3u * H3_TURBO_ADALN_WIDTH, &down, &up, error,
                          error_capacity) != 0)
            return 1;
        uint16_t rank_values[16];
        for (uint32_t step = 0u; step < H3_TURBO_STEPS; ++step)
            for (uint32_t kind = 0u; kind < 3u; ++kind) {
                for (uint32_t rank = 0u; rank < 16u; ++rank) {
                    float sum = 0.0f;
                    for (uint32_t column = 0u;
                         column < H3_TURBO_TIME_WIDTH; ++column)
                        sum += h3_bfloat(activated[step][kind][column]) *
                               h3_bfloat(down[(size_t)rank *
                                                  H3_TURBO_TIME_WIDTH +
                                              column]);
                    rank_values[rank] = h3_float_to_bfloat(sum);
                }
                for (uint32_t output_row = 0u;
                     output_row < 3u * H3_TURBO_ADALN_WIDTH; ++output_row) {
                    float sum = 0.0f;
                    for (uint32_t rank = 0u; rank < 16u; ++rank)
                        sum += h3_bfloat(rank_values[rank]) *
                               h3_bfloat(up[(size_t)output_row * 16u + rank]);
                    uint32_t modality = output_row / H3_TURBO_ADALN_WIDTH;
                    uint32_t column = output_row -
                                      modality * H3_TURBO_ADALN_WIDTH;
                    size_t output_index =
                        ((size_t)step * 9u + kind * 3u + modality) *
                            H3_TURBO_ADALN_WIDTH +
                        column;
                    destination[output_index] = h3_float_to_bfloat(
                        h3_bfloat(destination[output_index]) + sum);
                }
            }
        cache->blocks[layer].buffer = output;
        cache->blocks[layer].offset = 0u;
        snprintf(cache->blocks[layer].tensor.dtype,
                 sizeof(cache->blocks[layer].tensor.dtype), "BF16");
        cache->blocks[layer].tensor.rank = 3u;
        cache->blocks[layer].tensor.shape[0] = H3_TURBO_STEPS;
        cache->blocks[layer].tensor.shape[1] = H3_TURBO_MODALITY_ROWS;
        cache->blocks[layer].tensor.shape[2] = H3_TURBO_ADALN_WIDTH;
        cache->blocks[layer].tensor.data_length =
            output_count * sizeof(uint16_t);
        fprintf(stderr, "stage=turbo-compile adaln_layer=%u/50\n", layer + 1u);
        fflush(stderr);
    }

    const uint16_t *source_final = h3_cpu_tensor(
        source_cache, "final_modulations", &tensor, error, error_capacity);
    if (source_final == NULL || strcmp(tensor.dtype, "BF16") != 0 ||
        tensor.rank != 3u || tensor.shape[0] != 30u ||
        tensor.shape[1] != 3u || tensor.shape[2] != 10752u)
        return 1;
    size_t final_count = (size_t)H3_TURBO_STEPS * 3u * 10752u;
    id<MTLBuffer> final = [metal->device
        newBufferWithLength:final_count * sizeof(uint16_t)
                    options:MTLResourceStorageModeShared];
    if (final == nil) {
        e2e_error(error, error_capacity,
                  "cannot allocate Turbo final AdaLN table");
        return 1;
    }
    uint16_t *final_values = final.contents;
    for (uint32_t step = 0u; step < H3_TURBO_STEPS; ++step)
        for (uint32_t kind = 0u; kind < 3u; ++kind)
            for (uint32_t column = 0u; column < 10752u; ++column) {
                size_t output_index = ((size_t)step * 3u + kind) * 10752u +
                                      column;
                final_values[output_index] = h3_float_to_bfloat(
                    h3_interpolate_bf16(source_final, 3u, 10752u,
                                        interpolation[step][kind], 0u,
                                        column));
            }
    const uint16_t *final_down;
    const uint16_t *final_up;
    if (h3_turbo_pair(adapter, "final_layer.adaln_proj.linear", 16u,
                      H3_TURBO_TIME_WIDTH, 10752u, &final_down, &final_up,
                      error, error_capacity) != 0)
        return 1;
    uint16_t final_rank[16];
    for (uint32_t step = 0u; step < H3_TURBO_STEPS; ++step)
        for (uint32_t kind = 0u; kind < 3u; ++kind) {
            for (uint32_t rank = 0u; rank < 16u; ++rank) {
                float sum = 0.0f;
                for (uint32_t column = 0u; column < H3_TURBO_TIME_WIDTH;
                     ++column)
                    sum += h3_bfloat(activated[step][kind][column]) *
                           h3_bfloat(final_down[(size_t)rank *
                                                    H3_TURBO_TIME_WIDTH +
                                                column]);
                final_rank[rank] = h3_float_to_bfloat(sum);
            }
            for (uint32_t output_row = 0u; output_row < 10752u;
                 ++output_row) {
                float sum = 0.0f;
                for (uint32_t rank = 0u; rank < 16u; ++rank)
                    sum += h3_bfloat(final_rank[rank]) *
                           h3_bfloat(final_up[(size_t)output_row * 16u +
                                              rank]);
                size_t output_index = ((size_t)step * 3u + kind) * 10752u +
                                      output_row;
                final_values[output_index] = h3_float_to_bfloat(
                    h3_bfloat(final_values[output_index]) + sum);
            }
        }
    cache->final.buffer = final;
    cache->final.offset = 0u;
    snprintf(cache->final.tensor.dtype, sizeof(cache->final.tensor.dtype),
             "BF16");
    cache->final.tensor.rank = 3u;
    cache->final.tensor.shape[0] = H3_TURBO_STEPS;
    cache->final.tensor.shape[1] = 3u;
    cache->final.tensor.shape[2] = 10752u;
    cache->final.tensor.data_length = final_count * sizeof(uint16_t);
    cache->compile_seconds = e2e_now() - started;
    fprintf(stderr, "stage=turbo-compile status=complete seconds=%.6f\n",
            cache->compile_seconds);
    fflush(stderr);
    return 0;
}

static int h3_transformer_run(const uint16_t *text_states,
                              size_t text_rows,
                              const minimax_h3_m3_e2e_options *options,
                              const float *first_condition,
                              const float *last_condition,
                              h3_metal *metal,
                              h3_latents *latents,
                              double *download_seconds,
                              double *turbo_compile_seconds,
                              double *rope_precompute_seconds,
                              double *denoise_seconds,
                              size_t *peak_footprint,
                              char *error,
                              size_t error_capacity) {
    h3_remote_image weights = {0};
    h3_remote_image cache = {0};
    h3_remote_image adapter = {0};
    h3_turbo_cache turbo_cache = {0};
    h3_tree_runtime tree_runtime = {0};
    *turbo_compile_seconds = 0.0;
    *rope_precompute_seconds = 0.0;
    int status = 1;
    float *condition_video = NULL;
    uint32_t latent_height = options->height / 16u;
    uint32_t latent_width = options->width / 16u;
    if (options->frames % 17u != 5u || options->width % 32u != 0u ||
        options->height % 32u != 0u || latent_height == 0u ||
        latent_width == 0u) {
        e2e_error(error, error_capacity,
                  "H3 geometry requires frames=17*n+5 and width/height divisible by 32");
        return 2;
    }
    uint32_t video_frames = ((options->frames - 5u) / 17u) * 5u + 2u;
    uint32_t audio_frames = (uint32_t)llround(
        (double)options->frames / 24.0 * 40.0);
    uint32_t rows_per_video_frame = (latent_height / 2u) *
                                    (latent_width / 2u);
    uint32_t condition_images = (first_condition != NULL ? 1u : 0u) +
                                (last_condition != NULL ? 1u : 0u);
    uint32_t condition_rows = condition_images * rows_per_video_frame;
    uint32_t target_video_rows = video_frames * rows_per_video_frame;
    uint32_t video_rows = condition_rows + target_video_rows;
    uint32_t audio_rows = audio_frames * 2u;
    uint32_t rows = (uint32_t)text_rows + condition_rows + audio_rows +
                    target_video_rows;
    uint32_t padded = (rows + 31u) & ~31u;
    if (rows == 0u || rows > UINT32_MAX / 5376u) {
        e2e_error(error, error_capacity, "invalid packed H3 row count");
        return 2;
    }
    const char *tree_mode = getenv("MINIMAX_H3_TREE_ATTENTION");
    if (condition_images != 0u && tree_mode != NULL &&
        tree_mode[0] != '\0' && strcmp(tree_mode, "0") != 0) {
        e2e_error(error, error_capacity,
                  "tree attention is not available for FL2VA conditioning");
        return 2;
    }
    if (h3_tree_runtime_build(options, text_rows, metal, &tree_runtime,
                              error, error_capacity) != 0)
        return 1;
    if (h3_remote_image_open("transformer.safetensors", "transformer",
                             &weights, error, error_capacity) != 0)
        return 1;
    if (h3_remote_image_open("adaln_cache.safetensors", "adaln-cache",
                             &cache, error, error_capacity) != 0) {
        h3_remote_image_close(&weights);
        return 1;
    }
    const char *adapter_url = getenv("MINIMAX_H3_TURBO_ADAPTER");
    int turbo_enabled = adapter_url != NULL && adapter_url[0] != '\0';
    if (turbo_enabled) {
        if (strncmp(adapter_url, "file://", 7u) != 0 ||
            adapter_url[7] != '/') {
            e2e_error(error, error_capacity,
                      "MINIMAX_H3_TURBO_ADAPTER must be an absolute file:// URL");
            h3_remote_image_close(&cache);
            h3_remote_image_close(&weights);
            return 1;
        }
        if (h3_remote_image_open_path(adapter_url + 7u, "turbo-adapter",
                                      &adapter, error, error_capacity) != 0 ||
            h3_turbo_cache_build(&cache, &adapter, metal, &turbo_cache,
                                 error, error_capacity) != 0) {
            h3_remote_image_close(&adapter);
            h3_remote_image_close(&cache);
            h3_remote_image_close(&weights);
            return 1;
        }
        *turbo_compile_seconds = turbo_cache.compile_seconds;
    }
    h3_tensor_binding block_modulations[50] = {0};
    h3_tensor_binding final_modulation = {0};
    for (unsigned layer = 0u; layer < 50u; ++layer) {
        if (turbo_enabled) {
            block_modulations[layer] = turbo_cache.blocks[layer];
            continue;
        }
        char modulation_name[96];
        snprintf(modulation_name, sizeof(modulation_name),
                 "blocks.%u.modulations", layer);
        if (h3_bind_tensor(&cache, metal, modulation_name,
                           &block_modulations[layer], error,
                           error_capacity) != 0) {
            h3_remote_image_close(&adapter);
            h3_remote_image_close(&cache);
            h3_remote_image_close(&weights);
            return 1;
        }
    }
    if (turbo_enabled) {
        final_modulation = turbo_cache.final;
    } else if (h3_bind_tensor(&cache, metal, "final_modulations",
                              &final_modulation, error,
                              error_capacity) != 0) {
        h3_remote_image_close(&adapter);
        h3_remote_image_close(&cache);
        h3_remote_image_close(&weights);
        return 1;
    }
    *download_seconds = weights.download_seconds + cache.download_seconds;
    *peak_footprint = MAX(*peak_footprint, e2e_footprint());

    size_t hidden_bytes = (size_t)padded * 5376u * 2u;
    size_t heads_bytes = (size_t)padded * 56u * 128u * 2u;
    id<MTLBuffer> hidden = [metal->device newBufferWithLength:hidden_bytes
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> normalized = [metal->device newBufferWithLength:hidden_bytes
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> qkv = [metal->device
        newBufferWithLength:(size_t)padded * 21504u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> query = [metal->device newBufferWithLength:heads_bytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> key = [metal->device newBufferWithLength:heads_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> value = [metal->device newBufferWithLength:heads_bytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> attended = [metal->device newBufferWithLength:heads_bytes
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> attention_output = [metal->device newBufferWithLength:hidden_bytes
                                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> lora_scratch = [metal->device
        newBufferWithLength:(size_t)padded * 64u * sizeof(uint16_t)
                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> fc1 = [metal->device
        newBufferWithLength:(size_t)padded * 28672u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> gated = [metal->device
        newBufferWithLength:(size_t)padded * 14336u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> mlp_output = [metal->device newBufferWithLength:hidden_bytes
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> positions = [metal->device
        newBufferWithLength:(size_t)rows * 3u * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> rotary = [metal->device
        newBufferWithLength:(size_t)rows * 48u * 2u * sizeof(float)
                    options:MTLResourceStorageModePrivate];
    id<MTLBuffer> row_indices = [metal->device newBufferWithLength:rows
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> final_indices = [metal->device newBufferWithLength:rows
                                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> zero = [metal->device newBufferWithLength:4096u
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> text_input = [metal->device
        newBufferWithLength:((text_rows + 31u) & ~(size_t)31u) * 5120u * 2u
                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> text_hidden = [metal->device
        newBufferWithLength:((text_rows + 31u) & ~(size_t)31u) * 5376u * 2u
                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> video_input = [metal->device
        newBufferWithLength:(size_t)video_rows * 96u * sizeof(float)
                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> audio_input = [metal->device
        newBufferWithLength:(size_t)audio_rows * 32u * sizeof(float)
                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> video_hidden = [metal->device
        newBufferWithLength:(size_t)video_rows * 5376u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> audio_hidden = [metal->device
        newBufferWithLength:(size_t)audio_rows * 5376u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> video_velocity = [metal->device
        newBufferWithLength:(size_t)video_rows * 96u * sizeof(float)
                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> audio_velocity = [metal->device
        newBufferWithLength:(size_t)audio_rows * 32u * sizeof(float)
                    options:MTLResourceStorageModeShared];
    h3_tensor_binding video_sigmas = { {0}, nil, 0u };
    h3_tensor_binding audio_sigmas = { {0}, nil, 0u };
    if (hidden == nil || normalized == nil || qkv == nil || query == nil ||
        key == nil || value == nil || attended == nil ||
        attention_output == nil ||
        lora_scratch == nil ||
        fc1 == nil || gated == nil || mlp_output == nil || positions == nil ||
        rotary == nil ||
        row_indices == nil || final_indices == nil || zero == nil || text_input == nil ||
        text_hidden == nil || video_input == nil || audio_input == nil ||
        video_hidden == nil || audio_hidden == nil || video_velocity == nil ||
        audio_velocity == nil) {
        e2e_error(error, error_capacity, "transformer activation allocation failed");
        goto cleanup;
    }
    memset(zero.contents, 0, zero.length);
    memset(text_input.contents, 0, text_input.length);
    memcpy(text_input.contents, text_states, text_rows * 5120u * 2u);
    float *position_values = positions.contents;
    uint8_t *indices = row_indices.contents;
    uint8_t *final_index_values = final_indices.contents;
    for (uint32_t row = 0u; row < (uint32_t)text_rows; ++row) {
        position_values[row * 3u] = (float)row;
        position_values[row * 3u + 1u] = 0.0f;
        position_values[row * 3u + 2u] = 0.0f;
        indices[row] = 1u;
        final_index_values[row] = 0u;
    }
    double origin = (double)text_rows;
    double square_root_area = sqrt((double)latent_height * latent_width);
    double height_ratio = latent_height / square_root_area;
    double width_ratio = latent_width / square_root_area;
    uint32_t height_count = latent_height / 2u;
    uint32_t width_count = latent_width / 2u;
    double height_left = (1.0 - height_ratio) * 0.5;
    double width_left = (1.0 - width_ratio) * 0.5;
    double audio_left = width_left * 32.0;
    double audio_right = (width_left +
        (double)(width_count - 1u) * width_ratio / width_count) * 32.0;
    uint32_t condition_index = 0u;
    const double span[5] = {1.0, 4.0, 4.0, 4.0, 4.0};
    double final_temporal = origin;
    for (uint32_t frame = 1u; frame < video_frames; ++frame)
        final_temporal += (5.0 / 3.0) * span[(frame - 1u) % 5u];
    for (uint32_t anchor = 0u; anchor < 2u; ++anchor) {
        const float *condition = anchor == 0u ? first_condition
                                              : last_condition;
        if (condition == NULL) continue;
        double anchor_time = anchor == 0u ? origin : final_temporal;
        for (uint32_t spatial = 0u; spatial < rows_per_video_frame;
             ++spatial) {
            uint32_t row = (uint32_t)text_rows +
                           condition_index * rows_per_video_frame + spatial;
            uint32_t spatial_y = spatial / width_count;
            uint32_t spatial_x = spatial % width_count;
            position_values[row * 3u] = (float)anchor_time;
            position_values[row * 3u + 1u] = (float)((height_left +
                (double)spatial_y * height_ratio / height_count) * 32.0);
            position_values[row * 3u + 2u] = (float)((width_left +
                (double)spatial_x * width_ratio / width_count) * 32.0);
            indices[row] = 6u;
            final_index_values[row] = 2u;
        }
        ++condition_index;
    }
    for (uint32_t channel = 0u; channel < 2u; ++channel) {
        for (uint32_t frame = 0u; frame < audio_frames; ++frame) {
            uint32_t row = (uint32_t)text_rows + condition_rows +
                           channel * audio_frames + frame;
            position_values[row * 3u] = (float)(origin + frame);
            position_values[row * 3u + 1u] = 0.0f;
            position_values[row * 3u + 2u] =
                (float)(channel == 0u ? audio_left : audio_right);
            indices[row] = 5u;
            final_index_values[row] = 1u;
        }
    }
    double temporal = origin;
    for (uint32_t frame = 0u; frame < video_frames; ++frame) {
        for (uint32_t spatial = 0u; spatial < rows_per_video_frame; ++spatial) {
            uint32_t row = (uint32_t)text_rows + condition_rows + audio_rows +
                           frame * rows_per_video_frame + spatial;
            position_values[row * 3u] = (float)temporal;
            uint32_t spatial_y = spatial / width_count;
            uint32_t spatial_x = spatial % width_count;
            position_values[row * 3u + 1u] = (float)((height_left +
                (double)spatial_y * height_ratio / height_count) * 32.0);
            position_values[row * 3u + 2u] = (float)((width_left +
                (double)spatial_x * width_ratio / width_count) * 32.0);
            indices[row] = 0u;
            final_index_values[row] = 0u;
        }
        temporal += (5.0 / 3.0) * span[frame % 5u];
    }

    {
        double rope_started = e2e_now();
        id<MTLCommandBuffer> command = [metal->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:metal->h3_rope];
        [encoder setBuffer:positions offset:0 atIndex:0];
        [encoder setBuffer:rotary offset:0 atIndex:1];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake((size_t)rows * 48u, 1u, 1u)
                 threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
        if (h3_wait(command, error, error_capacity) != 0) goto cleanup;
        *rope_precompute_seconds = e2e_now() - rope_started;
        fprintf(stderr,
                "stage=rope-precompute rows=%u bytes=%zu seconds=%.6f\n",
                rows, (size_t)rows * 48u * 2u * sizeof(float),
                *rope_precompute_seconds);
        fflush(stderr);
    }

    {
        id<MTLCommandBuffer> command = [metal->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (h3_dense_linear_bfloat_output(
                &weights, metal, encoder, "condition_proj", text_input,
                text_hidden, (uint32_t)text_rows, 5120u, H3_DENSE_INPUT_F16, error,
                error_capacity) != 0) {
            [encoder endEncoding];
            goto cleanup;
        }
        [encoder endEncoding];
        if (h3_wait(command, error, error_capacity) != 0) goto cleanup;
    }
    for (unsigned layer = 0u; layer < 2u; ++layer) {
        char prefix[96];
        snprintf(prefix, sizeof(prefix), "token_refiner.blocks.%u", layer);
        if (h3_plain_block(&weights, turbo_enabled ? &adapter : NULL, metal,
                           prefix, text_hidden,
                           normalized, qkv, query, key, value, attended,
                           attention_output, lora_scratch, fc1, gated,
                           mlp_output, zero, zero,
                           (uint32_t)text_rows, error, error_capacity) != 0)
            goto cleanup;
    }
    {
        id<MTLCommandBuffer> command = [metal->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (h3_rms_plain(&weights, metal, encoder,
                         "token_refiner.final_norm.weight", text_hidden,
                         normalized, (uint32_t)text_rows, 5376u, error,
                         error_capacity) != 0) {
            [encoder endEncoding];
            goto cleanup;
        }
        [encoder endEncoding];
        if (h3_wait(command, error, error_capacity) != 0) goto cleanup;
        memcpy(text_hidden.contents, normalized.contents,
               text_rows * 5376u * 2u);
    }
    {
        size_t first_bad = 0u;
        float largest = 0.0f;
        if (!h3_bfloat_buffer_is_finite(text_hidden, text_rows * 5376u,
                                        &first_bad, &largest)) {
            e2e_error(error, error_capacity,
                      "non-finite token-refiner output at index %zu "
                      "(largest finite magnitude %.9g)",
                      first_bad, largest);
            goto cleanup;
        }
    }

    size_t video_values = (size_t)24u * video_frames * latent_height * latent_width;
    size_t condition_values_per_image = (size_t)24u * latent_height *
                                        latent_width;
    size_t condition_value_count = condition_values_per_image *
                                   condition_images;
    size_t audio_values = (size_t)32u * 2u * audio_frames;
    float *video = malloc(video_values * sizeof(float));
    condition_video = condition_value_count != 0u
                          ? malloc(condition_value_count * sizeof(float))
                          : NULL;
    float *audio = malloc(audio_values * sizeof(float));
    if (video == NULL || audio == NULL ||
        (condition_value_count != 0u && condition_video == NULL)) {
        free(video);
        free(audio);
        e2e_error(error, error_capacity, "latent allocation failed");
        goto cleanup;
    }
    uint64_t rng = options->seed != 0u ? options->seed : UINT64_C(1);
    condition_index = 0u;
    for (uint32_t anchor = 0u; anchor < 2u; ++anchor) {
        const float *condition = anchor == 0u ? first_condition
                                              : last_condition;
        if (condition == NULL) continue;
        float *destination = condition_video +
                             condition_index * condition_values_per_image;
        for (size_t index = 0u; index < condition_values_per_image; ++index)
            destination[index] = 0.999f * condition[index] +
                                 0.001f * h3_rng_normal(&rng);
        ++condition_index;
    }
    for (size_t index = 0u; index < video_values; ++index)
        video[index] = h3_rng_normal(&rng);
    for (size_t index = 0u; index < audio_values; ++index)
        audio[index] = h3_rng_normal(&rng);

    const float *video_sigma;
    const float *audio_sigma;
    uint32_t step_count = turbo_enabled ? H3_TURBO_STEPS : 30u;
    if (turbo_enabled) {
        video_sigma = turbo_cache.video_sigmas;
        audio_sigma = turbo_cache.audio_sigmas;
    } else {
        if (h3_bind_tensor(&cache, metal, "video_sigmas", &video_sigmas,
                           error, error_capacity) != 0 ||
            h3_bind_tensor(&cache, metal, "audio_sigmas", &audio_sigmas,
                           error, error_capacity) != 0) {
            free(video);
            free(audio);
            goto cleanup;
        }
        video_sigma = (const float *)((const uint8_t *)
            video_sigmas.buffer.contents + video_sigmas.offset);
        audio_sigma = (const float *)((const uint8_t *)
            audio_sigmas.buffer.contents + audio_sigmas.offset);
    }
    double started = e2e_now();
    const char *validate_groups_text =
        getenv("MINIMAX_H3_VALIDATE_LAYER_GROUPS");
    const int validate_layer_groups = validate_groups_text != NULL &&
                                      strcmp(validate_groups_text, "1") == 0;
    const char *pipeline_groups_text =
        getenv("MINIMAX_H3_PIPELINE_LAYER_GROUPS");
    const int pipeline_layer_groups = !validate_layer_groups &&
        (pipeline_groups_text == NULL ||
         strcmp(pipeline_groups_text, "0") != 0);
    const char *lora_mma_text = getenv("MINIMAX_H3_LORA_MMA");
    const int lora_mma_enabled =
        lora_mma_text != NULL && strcmp(lora_mma_text, "1") == 0;
    fprintf(stderr,
            "stage=denoise-schedule layer_groups=%s lora=%s validation=%s\n",
            pipeline_layer_groups ? "pipelined" : "synchronous",
            lora_mma_enabled ? "bf16-mma" : "simd-reduction",
            validate_layer_groups ? "layer-group" : "step");
    for (uint32_t step = 0u; step < step_count; ++step) {
        float *vp = video_input.contents;
        uint32_t patch_row = 0u;
        for (uint32_t condition = 0u; condition < condition_images;
             ++condition) {
            const float *condition_source = condition_video +
                (size_t)condition * condition_values_per_image;
            for (uint32_t y = 0u; y < latent_height; y += 2u)
                for (uint32_t x = 0u; x < latent_width; x += 2u) {
                    uint32_t column = 0u;
                    for (uint32_t channel = 0u; channel < 24u; ++channel)
                        for (uint32_t py = 0u; py < 2u; ++py)
                            for (uint32_t px = 0u; px < 2u; ++px) {
                                size_t source = ((size_t)channel *
                                    latent_height + y + py) * latent_width +
                                    x + px;
                                vp[(size_t)patch_row * 96u + column++] =
                                    condition_source[source];
                            }
                    ++patch_row;
                }
        }
        for (uint32_t frame = 0u; frame < video_frames; ++frame) {
            for (uint32_t y = 0u; y < latent_height; y += 2u) {
                for (uint32_t x = 0u; x < latent_width; x += 2u) {
                    uint32_t column = 0u;
                    for (uint32_t channel = 0u; channel < 24u; ++channel)
                        for (uint32_t py = 0u; py < 2u; ++py)
                            for (uint32_t px = 0u; px < 2u; ++px) {
                                size_t source = (((size_t)channel * video_frames + frame) *
                                                 latent_height + y + py) * latent_width + x + px;
                                vp[(size_t)patch_row * 96u + column++] = video[source];
                            }
                    ++patch_row;
                }
            }
        }
        float *ap = audio_input.contents;
        for (uint32_t channel = 0u; channel < 2u; ++channel)
            for (uint32_t frame = 0u; frame < audio_frames; ++frame)
                for (uint32_t feature = 0u; feature < 32u; ++feature) {
                    size_t source = ((size_t)feature * 2u + channel) *
                                    audio_frames + frame;
                    size_t destination = ((size_t)channel * audio_frames + frame) *
                                         32u + feature;
                    ap[destination] = audio[source];
                }
        {
            id<MTLCommandBuffer> command = [metal->queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (h3_dense_linear_bfloat_output(
                    &weights, metal, encoder, "video_patch_proj", video_input,
                    video_hidden, video_rows, 96u, H3_DENSE_INPUT_F32, error,
                    error_capacity) != 0 ||
                h3_dense_linear_bfloat_output(
                    &weights, metal, encoder, "audio_patch_proj", audio_input,
                    audio_hidden, audio_rows, 32u, H3_DENSE_INPUT_F32, error,
                    error_capacity) != 0) {
                [encoder endEncoding];
                free(video);
                free(audio);
                goto cleanup;
            }
            [encoder endEncoding];
            if (h3_wait(command, error, error_capacity) != 0) {
                free(video);
                free(audio);
                goto cleanup;
            }
        }
        memset(hidden.contents, 0, hidden_bytes);
        memcpy(hidden.contents, text_hidden.contents, text_rows * 5376u * 2u);
        memcpy((uint16_t *)hidden.contents + text_rows * 5376u,
               video_hidden.contents, (size_t)condition_rows * 5376u * 2u);
        memcpy((uint16_t *)hidden.contents +
                   ((size_t)text_rows + condition_rows) * 5376u,
               audio_hidden.contents, (size_t)audio_rows * 5376u * 2u);
        memcpy((uint16_t *)hidden.contents +
                   ((size_t)text_rows + condition_rows + audio_rows) * 5376u,
               (uint16_t *)video_hidden.contents +
                   (size_t)condition_rows * 5376u,
               (size_t)target_video_rows * 5376u * 2u);
        const unsigned layers_per_command = 4u;
        id<MTLCommandBuffer> group_commands[13] = { nil };
        unsigned group_layer_starts[13] = { 0u };
        unsigned group_layer_ends[13] = { 0u };
        unsigned group_count = 0u;
        for (unsigned layer_start = 0u; layer_start < 50u;
             layer_start += layers_per_command) {
            unsigned layer_end = layer_start + layers_per_command;
            if (layer_end > 50u) layer_end = 50u;
            double group_started = e2e_now();
            id<MTLCommandBuffer> command = [metal->queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            int group_status = 0;
            for (unsigned layer = layer_start; layer < layer_end; ++layer) {
                if (h3_denoise_block(&weights,
                                     turbo_enabled ? &adapter : NULL,
                                     block_modulations[layer], metal,
                                     &tree_runtime, encoder, layer, step,
                                     row_indices, hidden, normalized, qkv,
                                     query, key, value, attended,
                                     attention_output, lora_scratch, fc1,
                                     gated, mlp_output, rotary, rows, error,
                                     error_capacity) != 0) {
                    group_status = 1;
                    break;
                }
            }
            [encoder endEncoding];
            if (group_status) {
                free(video);
                free(audio);
                goto cleanup;
            }
            if (pipeline_layer_groups) {
                group_commands[group_count] = command;
                group_layer_starts[group_count] = layer_start;
                group_layer_ends[group_count] = layer_end;
                ++group_count;
                [command commit];
                continue;
            }
            if (h3_wait(command, error, error_capacity) != 0) {
                free(video);
                free(audio);
                goto cleanup;
            }
            if (validate_layer_groups) {
                size_t first_bad = 0u;
                float largest = 0.0f;
                if (!h3_bfloat_buffer_is_finite(
                        hidden, (size_t)rows * 5376u, &first_bad, &largest)) {
                    free(video);
                    free(audio);
                    e2e_error(
                        error, error_capacity,
                        "non-finite transformer hidden at step %u layer %u "
                        "index %zu (largest finite magnitude %.9g)",
                        step, layer_end - 1u, first_bad, largest);
                    goto cleanup;
                }
            }
            fprintf(stderr,
                    "stage=denoise-layer-group step=%u/%u layers=%u-%u/50 "
                    "seconds=%.6f\n",
                    step + 1u, step_count, layer_start + 1u, layer_end,
                    e2e_now() - group_started);
            fflush(stderr);
        }
        if (pipeline_layer_groups) {
            if (group_count == 0u ||
                h3_wait_committed(group_commands[group_count - 1u], error,
                                  error_capacity) != 0) {
                free(video);
                free(audio);
                goto cleanup;
            }
            for (unsigned group = 0u; group < group_count; ++group) {
                id<MTLCommandBuffer> command = group_commands[group];
                if (command.status == MTLCommandBufferStatusError) {
                    e2e_error(error, error_capacity,
                              "Metal layer-group command failed: %s",
                              command.error.localizedDescription.UTF8String);
                    free(video);
                    free(audio);
                    goto cleanup;
                }
                double gpu_seconds = command.GPUEndTime - command.GPUStartTime;
                fprintf(stderr,
                        "stage=denoise-layer-group step=%u/%u "
                        "layers=%u-%u/50 seconds=%.6f submit=pipelined\n",
                        step + 1u, step_count,
                        group_layer_starts[group] + 1u,
                        group_layer_ends[group], gpu_seconds);
            }
            fflush(stderr);
        }
        if (!validate_layer_groups) {
            size_t first_bad = 0u;
            float largest = 0.0f;
            if (!h3_bfloat_buffer_is_finite(hidden, (size_t)rows * 5376u,
                                            &first_bad, &largest)) {
                free(video);
                free(audio);
                e2e_error(error, error_capacity,
                          "non-finite transformer hidden at step %u "
                          "index %zu (largest finite magnitude %.9g)",
                          step, first_bad, largest);
                goto cleanup;
            }
        }
        {
            id<MTLCommandBuffer> command = [metal->queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (h3_rms_adaln_bound(
                    &weights, metal, encoder, "final_layer.norm.weight",
                    final_modulation, step, 3u, 2u * 5376u, 0u, 5376u,
                    final_indices, hidden, normalized, rows, error,
                    error_capacity) != 0) {
                [encoder endEncoding];
                free(video);
                free(audio);
                goto cleanup;
            }
            [encoder endEncoding];
            if (h3_wait(command, error, error_capacity) != 0) {
                free(video);
                free(audio);
                goto cleanup;
            }
            size_t first_bad = 0u;
            float largest = 0.0f;
            if (!h3_bfloat_buffer_is_finite(normalized,
                                            (size_t)rows * 5376u,
                                            &first_bad, &largest)) {
                free(video);
                free(audio);
                e2e_error(error, error_capacity,
                          "non-finite final norm at step %u index %zu "
                          "(largest finite magnitude %.9g)",
                          step, first_bad, largest);
                goto cleanup;
            }
        }
        memcpy(audio_hidden.contents,
               (uint16_t *)normalized.contents +
                   ((size_t)text_rows + condition_rows) * 5376u,
               (size_t)audio_rows * 5376u * 2u);
        memcpy(video_hidden.contents, (uint16_t *)normalized.contents +
                   (size_t)text_rows * 5376u,
               (size_t)condition_rows * 5376u * 2u);
        memcpy((uint16_t *)video_hidden.contents +
                   (size_t)condition_rows * 5376u,
               (uint16_t *)normalized.contents +
                   ((size_t)text_rows + condition_rows + audio_rows) * 5376u,
               (size_t)target_video_rows * 5376u * 2u);
        {
            id<MTLCommandBuffer> command = [metal->queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (h3_dense_linear_f32_output(
                    &weights, metal, encoder, "final_layer.video_out",
                    video_hidden, video_velocity, video_rows, 5376u, error,
                    error_capacity) != 0 ||
                h3_dense_linear_f32_output(
                    &weights, metal, encoder, "final_layer.audio_out",
                    audio_hidden, audio_velocity, audio_rows, 5376u, error,
                    error_capacity) != 0) {
                [encoder endEncoding];
                free(video);
                free(audio);
                goto cleanup;
            }
            [encoder endEncoding];
            if (h3_wait(command, error, error_capacity) != 0) {
                free(video);
                free(audio);
                goto cleanup;
            }
            size_t first_bad = 0u;
            float largest = 0.0f;
            if (!h3_f32_buffer_is_finite(video_velocity,
                                         (size_t)video_rows * 96u,
                                         &first_bad, &largest)) {
                free(video);
                free(audio);
                e2e_error(error, error_capacity,
                          "non-finite video velocity at step %u index %zu "
                          "(largest finite magnitude %.9g)",
                          step, first_bad, largest);
                goto cleanup;
            }
            if (!h3_f32_buffer_is_finite(audio_velocity,
                                         (size_t)audio_rows * 32u,
                                         &first_bad, &largest)) {
                free(video);
                free(audio);
                e2e_error(error, error_capacity,
                          "non-finite audio velocity at step %u index %zu "
                          "(largest finite magnitude %.9g)",
                          step, first_bad, largest);
                goto cleanup;
            }
        }
        const float *vv = (const float *)video_velocity.contents +
                          (size_t)condition_rows * 96u;
        patch_row = 0u;
        float vd = video_sigma[step] - video_sigma[step + 1u];
        for (uint32_t frame = 0u; frame < video_frames; ++frame)
            for (uint32_t y = 0u; y < latent_height; y += 2u)
                for (uint32_t x = 0u; x < latent_width; x += 2u) {
                    uint32_t column = 0u;
                    for (uint32_t channel = 0u; channel < 24u; ++channel)
                        for (uint32_t py = 0u; py < 2u; ++py)
                            for (uint32_t px = 0u; px < 2u; ++px) {
                                size_t destination = (((size_t)channel * video_frames + frame) *
                                                      latent_height + y + py) * latent_width + x + px;
                                video[destination] += vd * vv[
                                    (size_t)patch_row * 96u + column++];
                            }
                    ++patch_row;
                }
        const float *av = audio_velocity.contents;
        float ad = audio_sigma[step] - audio_sigma[step + 1u];
        for (uint32_t channel = 0u; channel < 2u; ++channel)
            for (uint32_t frame = 0u; frame < audio_frames; ++frame)
                for (uint32_t feature = 0u; feature < 32u; ++feature) {
                    size_t destination = ((size_t)feature * 2u + channel) *
                                         audio_frames + frame;
                    size_t source = ((size_t)channel * audio_frames + frame) *
                                    32u + feature;
                    audio[destination] += ad * av[source];
                }
        for (size_t index = 0u; index < video_values; ++index) {
            if (!isfinite(video[index])) {
                free(video); free(audio);
                e2e_error(error, error_capacity,
                          "non-finite video latent at step %u index %zu",
                          step, index);
                goto cleanup;
            }
        }
        for (size_t index = 0u; index < audio_values; ++index) {
            if (!isfinite(audio[index])) {
                free(video); free(audio);
                e2e_error(error, error_capacity,
                          "non-finite audio latent at step %u index %zu",
                          step, index);
                goto cleanup;
            }
        }
        *peak_footprint = MAX(*peak_footprint, e2e_footprint());
        fprintf(stderr, "stage=denoise step=%u/%u rows=%u footprint=%zu\n",
                step + 1u, step_count, rows, e2e_footprint());
        fflush(stderr);
    }
    *denoise_seconds = e2e_now() - started;
    latents->video = video;
    latents->audio = audio;
    latents->video_latent_frames = video_frames;
    latents->latent_height = latent_height;
    latents->latent_width = latent_width;
    latents->audio_latent_frames = audio_frames;
    status = 0;
cleanup:
    free(condition_video);
    h3_remote_image_close(&adapter);
    h3_remote_image_close(&cache);
    h3_remote_image_close(&weights);
    return status;
}

typedef struct {
    h3_tensor_binding norm1;
    h3_dense_binding qkv;
    h3_dense_binding attention_output;
    h3_tensor_binding scale1;
    h3_tensor_binding norm2;
    h3_dense_binding w1;
    h3_dense_binding w2;
    h3_tensor_binding scale2;
} h3_video_block_binding;

typedef struct {
    h3_dense_binding x_embedder;
    h3_video_block_binding blocks[36];
    h3_tensor_binding norm_out_weight;
    h3_tensor_binding norm_out_bias;
    h3_dense_binding projection_out;
} h3_video_model_binding;

static int h3_video_bind_weights(h3_remote_image *weights,
                                 h3_metal *metal,
                                 h3_video_model_binding *model,
                                 char *error,
                                 size_t error_capacity) {
    char base[160];
    char name[192];
    if (h3_dense_bind(weights, metal, "decoder.x_embedder",
                      &model->x_embedder, error, error_capacity) != 0)
        return 1;
    for (unsigned layer = 0u; layer < 36u; ++layer) {
        h3_video_block_binding *block = &model->blocks[layer];
        snprintf(base, sizeof(base), "decoder.transformer_blocks.%u", layer);
#define H3_VIDEO_BIND_TENSOR(field, suffix)                                  \
        do {                                                                 \
            snprintf(name, sizeof(name), "%s.%s", base, suffix);            \
            if (h3_bind_tensor(weights, metal, name, &block->field, error,   \
                               error_capacity) != 0)                         \
                return 1;                                                    \
        } while (0)
#define H3_VIDEO_BIND_DENSE(field, suffix)                                   \
        do {                                                                 \
            snprintf(name, sizeof(name), "%s.%s", base, suffix);            \
            if (h3_dense_bind(weights, metal, name, &block->field, error,    \
                              error_capacity) != 0)                          \
                return 1;                                                    \
        } while (0)
        H3_VIDEO_BIND_TENSOR(norm1, "norm1.weight");
        H3_VIDEO_BIND_DENSE(qkv, "attn.to_qkv");
        H3_VIDEO_BIND_DENSE(attention_output, "attn.to_out");
        H3_VIDEO_BIND_TENSOR(scale1, "scale1");
        H3_VIDEO_BIND_TENSOR(norm2, "norm2.weight");
        H3_VIDEO_BIND_DENSE(w1, "ff.w1");
        H3_VIDEO_BIND_DENSE(w2, "ff.w2");
        H3_VIDEO_BIND_TENSOR(scale2, "scale2");
#undef H3_VIDEO_BIND_DENSE
#undef H3_VIDEO_BIND_TENSOR
    }
    if (h3_bind_tensor(weights, metal, "decoder.norm_out.weight",
                       &model->norm_out_weight, error, error_capacity) != 0 ||
        h3_bind_tensor(weights, metal, "decoder.norm_out.bias",
                       &model->norm_out_bias, error, error_capacity) != 0 ||
        h3_dense_bind(weights, metal, "decoder.proj_out",
                      &model->projection_out, error, error_capacity) != 0)
        return 1;
    return 0;
}

static void h3_video_rms_bound(const h3_tensor_binding *weight,
                               h3_metal *metal,
                               id<MTLComputeCommandEncoder> encoder,
                               id<MTLBuffer> input,
                               id<MTLBuffer> output,
                               uint32_t rows) {
    h3_norm_parameters parameters = {
        .rows = rows, .columns = 2048u, .epsilon = 1e-5f,
    };
    [encoder setComputePipelineState:metal->rms_f16];
    [encoder setBuffer:input offset:0 atIndex:0];
    h3_set(encoder, *weight, 1);
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
}

static void h3_video_scaled_residual_bound(
    const h3_tensor_binding *scale,
    h3_metal *metal,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> hidden,
    id<MTLBuffer> update,
    uint32_t rows) {
    uint32_t columns = 2048u;
    [encoder setComputePipelineState:metal->scaled_residual_f16];
    [encoder setBuffer:hidden offset:0 atIndex:0];
    [encoder setBuffer:update offset:0 atIndex:1];
    h3_set(encoder, *scale, 2);
    [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
    [encoder setBytes:&columns length:sizeof(columns) atIndex:4];
    [encoder dispatchThreads:MTLSizeMake((size_t)rows * columns, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
}

static int h3_video_block_encode(
    const h3_video_block_binding *block,
    h3_metal *metal,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> hidden,
    id<MTLBuffer> normalized,
    id<MTLBuffer> qkv,
    id<MTLBuffer> query,
    id<MTLBuffer> key,
    id<MTLBuffer> value,
    id<MTLBuffer> attended,
    id<MTLBuffer> attention_output,
    id<MTLBuffer> fc1,
    id<MTLBuffer> gated,
    id<MTLBuffer> mlp_output,
    id<MTLBuffer> rotary,
    uint32_t rows,
    char *error,
    size_t error_capacity) {
    h3_video_rms_bound(&block->norm1, metal, encoder, hidden, normalized,
                       rows);
    if (h3_dense_linear_bound(&block->qkv, metal, encoder, normalized, qkv,
                              rows, 2048u, error, error_capacity) != 0)
        return 1;
    [encoder setComputePipelineState:metal->video_prepare];
    [encoder setBuffer:qkv offset:0 atIndex:0];
    [encoder setBuffer:rotary offset:0 atIndex:1];
    [encoder setBuffer:query offset:0 atIndex:2];
    [encoder setBuffer:key offset:0 atIndex:3];
    [encoder setBuffer:value offset:0 atIndex:4];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 32u, 1u)
                threadsPerThreadgroup:MTLSizeMake(64u, 1u, 1u)];
    [encoder setComputePipelineState:metal->reference_vae_attention
                                         ? metal->video_attention
                                         : metal->video_attention_tiled];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key offset:0 atIndex:1];
    [encoder setBuffer:value offset:0 atIndex:2];
    [encoder setBuffer:attended offset:0 atIndex:3];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    if (metal->reference_vae_attention) {
        [encoder dispatchThreadgroups:MTLSizeMake(rows * 32u, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    } else {
        [encoder dispatchThreadgroups:
            MTLSizeMake(((rows + 7u) / 8u) * 32u, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    }
    if (h3_dense_linear_bound(&block->attention_output, metal, encoder,
                              attended, attention_output, rows, 2048u, error,
                              error_capacity) != 0)
        return 1;
    h3_video_scaled_residual_bound(&block->scale1, metal, encoder, hidden,
                                   attention_output, rows);
    h3_video_rms_bound(&block->norm2, metal, encoder, hidden, normalized,
                       rows);
    if (h3_dense_linear_bound(&block->w1, metal, encoder, normalized, fc1,
                              rows, 2048u, error, error_capacity) != 0)
        return 1;
    uint32_t width = 8192u;
    [encoder setComputePipelineState:metal->silu_split];
    [encoder setBuffer:fc1 offset:0 atIndex:0];
    [encoder setBuffer:gated offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
    [encoder setBytes:&width length:sizeof(width) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake((size_t)rows * width, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    if (h3_dense_linear_bound(&block->w2, metal, encoder, gated, mlp_output,
                              rows, 8192u, error, error_capacity) != 0)
        return 1;
    h3_video_scaled_residual_bound(&block->scale2, metal, encoder, hidden,
                                   mlp_output, rows);
    return 0;
}

static const uint16_t *h3_f16_tensor_pointer(h3_remote_image *image,
                                              const char *name,
                                              minimax_h3_remote_tensor *tensor,
                                              char *error,
                                              size_t error_capacity) {
    if (minimax_h3_remote_safetensors_find(&image->remote, name, tensor, error,
                                            error_capacity) != 0 ||
        strcmp(tensor->dtype, "F16") != 0)
        return NULL;
    return (const uint16_t *)((const uint8_t *)image->mapping +
                              image->file_padding + tensor->data_start);
}

static MPSImage *h3_mps_image(id<MTLDevice> device,
                              uint32_t width,
                              uint32_t height,
                              uint32_t channels) {
    MPSImageDescriptor *descriptor = [MPSImageDescriptor
        imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                    width:width
                                   height:height
                          featureChannels:channels];
    return [[MPSImage alloc] initWithDevice:device imageDescriptor:descriptor];
}

static int h3_mps_wait(id<MTLCommandBuffer> command,
                       char *error,
                       size_t error_capacity) {
    return h3_wait(command, error, error_capacity);
}

static int h3_mps_image_unary(h3_metal *metal,
                              id<MTLComputePipelineState> pipeline,
                              MPSImage *source,
                              MPSImage *destination,
                              char *error,
                              size_t error_capacity) {
    id<MTLCommandBuffer> command = [metal->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:source.texture atIndex:0];
    [encoder setTexture:destination.texture atIndex:1];
    NSUInteger slices = (destination.featureChannels + 3u) / 4u;
    [encoder dispatchThreads:MTLSizeMake(destination.width,
                                         destination.height, slices)
             threadsPerThreadgroup:MTLSizeMake(8u, 8u, 1u)];
    [encoder endEncoding];
    return h3_mps_wait(command, error, error_capacity);
}

static int h3_mps_image_add(h3_metal *metal,
                            MPSImage *left,
                            MPSImage *right,
                            MPSImage *destination,
                            char *error,
                            size_t error_capacity) {
    id<MTLCommandBuffer> command = [metal->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:metal->image_add];
    [encoder setTexture:left.texture atIndex:0];
    [encoder setTexture:right.texture atIndex:1];
    [encoder setTexture:destination.texture atIndex:2];
    NSUInteger slices = (destination.featureChannels + 3u) / 4u;
    [encoder dispatchThreads:MTLSizeMake(destination.width,
                                         destination.height, slices)
             threadsPerThreadgroup:MTLSizeMake(8u, 8u, 1u)];
    [encoder endEncoding];
    return h3_mps_wait(command, error, error_capacity);
}

static int h3_mps_convolution(h3_remote_image *vae,
                              h3_metal *metal,
                              const char *prefix,
                              MPSImage *source,
                              uint32_t stride,
                              MPSImage **destination,
                              char *error,
                              size_t error_capacity) {
    char weight_name[192];
    char bias_name[192];
    snprintf(weight_name, sizeof(weight_name), "%s.weight", prefix);
    snprintf(bias_name, sizeof(bias_name), "%s.bias", prefix);
    minimax_h3_remote_tensor weight_tensor;
    minimax_h3_remote_tensor bias_tensor;
    const uint16_t *source_weights = h3_f16_tensor_pointer(
        vae, weight_name, &weight_tensor, error, error_capacity);
    const uint16_t *source_bias = h3_f16_tensor_pointer(
        vae, bias_name, &bias_tensor, error, error_capacity);
    if (source_weights == NULL || source_bias == NULL ||
        weight_tensor.rank != 5u || bias_tensor.rank != 1u ||
        weight_tensor.shape[0] != bias_tensor.shape[0] ||
        weight_tensor.shape[1] != source.featureChannels ||
        (weight_tensor.shape[2] != 1u && weight_tensor.shape[2] != 3u) ||
        (weight_tensor.shape[3] != 1u && weight_tensor.shape[3] != 3u) ||
        weight_tensor.shape[3] != weight_tensor.shape[4] ||
        (stride != 1u && stride != 2u)) {
        e2e_error(error, error_capacity,
                  "unexpected Video VAE convolution shape: %s", prefix);
        return 1;
    }
    uint32_t output_channels = (uint32_t)weight_tensor.shape[0];
    uint32_t input_channels = (uint32_t)weight_tensor.shape[1];
    uint32_t temporal_kernel = (uint32_t)weight_tensor.shape[2];
    uint32_t kernel = (uint32_t)weight_tensor.shape[3];
    size_t packed_count = (size_t)output_channels * kernel * kernel *
                          input_channels;
    NSMutableData *packed = [NSMutableData
        dataWithLength:packed_count * sizeof(uint16_t)];
    if (packed == nil) {
        e2e_error(error, error_capacity,
                  "cannot allocate Video VAE convolution weights: %s", prefix);
        return 1;
    }
    uint16_t *target_weights = packed.mutableBytes;
    uint32_t temporal = temporal_kernel - 1u;
    for (uint32_t output = 0u; output < output_channels; ++output)
        for (uint32_t y = 0u; y < kernel; ++y)
            for (uint32_t x = 0u; x < kernel; ++x)
                for (uint32_t input = 0u; input < input_channels; ++input) {
                    size_t source_index =
                        ((((size_t)output * input_channels + input) *
                               temporal_kernel + temporal) *
                              kernel + y) *
                             kernel + x;
                    size_t target_index =
                        (((size_t)output * kernel + y) * kernel + x) *
                            input_channels + input;
                    target_weights[target_index] = source_weights[source_index];
                }
    NSMutableData *biases = [NSMutableData
        dataWithLength:(size_t)output_channels * sizeof(float)];
    float *target_bias = biases.mutableBytes;
    for (uint32_t output = 0u; output < output_channels; ++output)
        target_bias[output] = (float)((const __fp16 *)source_bias)[output];

    MPSCNNConvolutionDescriptor *descriptor = [MPSCNNConvolutionDescriptor
        cnnConvolutionDescriptorWithKernelWidth:kernel
                                   kernelHeight:kernel
                           inputFeatureChannels:input_channels
                          outputFeatureChannels:output_channels];
    descriptor.strideInPixelsX = stride;
    descriptor.strideInPixelsY = stride;
    H3MPSConvolutionDataSource *data_source =
        [H3MPSConvolutionDataSource new];
    data_source.convDescriptor = descriptor;
    data_source.weightStorage = packed;
    data_source.biasStorage = biases;
    data_source.sourceLabel = [NSString stringWithUTF8String:prefix];
    MPSCNNConvolution *convolution = [[MPSCNNConvolution alloc]
        initWithDevice:metal->device weights:data_source];
    convolution.edgeMode = MPSImageEdgeModeMirror;
    convolution.offset = (MPSOffset) {
        .x = stride == 2u ? 1 : 0,
        .y = stride == 2u ? 1 : 0,
        .z = 0,
    };
    uint32_t width = stride == 2u ? (uint32_t)((source.width + 1u) / 2u)
                                  : (uint32_t)source.width;
    uint32_t height = stride == 2u ? (uint32_t)((source.height + 1u) / 2u)
                                   : (uint32_t)source.height;
    MPSImage *output = h3_mps_image(metal->device, width, height,
                                    output_channels);
    if (convolution == nil || output == nil) {
        e2e_error(error, error_capacity,
                  "cannot create Video VAE convolution: %s", prefix);
        return 1;
    }
    id<MTLCommandBuffer> command = [metal->queue commandBuffer];
    [convolution encodeToCommandBuffer:command
                           sourceImage:source
                      destinationImage:output];
    if (h3_mps_wait(command, error, error_capacity) != 0) return 1;
    *destination = output;
    return 0;
}

static int h3_mps_norm_silu(h3_remote_image *vae,
                            h3_metal *metal,
                            const char *prefix,
                            MPSImage *source,
                            MPSImage **destination,
                            char *error,
                            size_t error_capacity) {
    char weight_name[192];
    char bias_name[192];
    snprintf(weight_name, sizeof(weight_name), "%s.weight", prefix);
    snprintf(bias_name, sizeof(bias_name), "%s.bias", prefix);
    minimax_h3_remote_tensor weight_tensor;
    minimax_h3_remote_tensor bias_tensor;
    const uint16_t *weights = h3_f16_tensor_pointer(
        vae, weight_name, &weight_tensor, error, error_capacity);
    const uint16_t *bias = h3_f16_tensor_pointer(
        vae, bias_name, &bias_tensor, error, error_capacity);
    if (weights == NULL || bias == NULL || weight_tensor.rank != 1u ||
        bias_tensor.rank != 1u ||
        weight_tensor.shape[0] != source.featureChannels ||
        bias_tensor.shape[0] != source.featureChannels) {
        e2e_error(error, error_capacity,
                  "unexpected Video VAE group norm shape: %s", prefix);
        return 1;
    }
    uint32_t channels = (uint32_t)source.featureChannels;
    H3MPSGroupNormDataSource *data_source = [H3MPSGroupNormDataSource new];
    data_source.gammaStorage = [NSMutableData
        dataWithLength:(size_t)channels * sizeof(float)];
    data_source.betaStorage = [NSMutableData
        dataWithLength:(size_t)channels * sizeof(float)];
    data_source.channelCount = channels;
    data_source.numberOfGroups = 32u;
    data_source.sourceLabel = [NSString stringWithUTF8String:prefix];
    float *gamma = data_source.gammaStorage.mutableBytes;
    float *beta = data_source.betaStorage.mutableBytes;
    for (uint32_t channel = 0u; channel < channels; ++channel) {
        gamma[channel] = (float)((const __fp16 *)weights)[channel];
        beta[channel] = (float)((const __fp16 *)bias)[channel];
    }
    MPSCNNGroupNormalization *normalization =
        [[MPSCNNGroupNormalization alloc] initWithDevice:metal->device
                                              dataSource:data_source];
    normalization.epsilon = 1e-6f;
    MPSImage *normalized = h3_mps_image(
        metal->device, (uint32_t)source.width, (uint32_t)source.height,
        channels);
    MPSImage *activated = h3_mps_image(
        metal->device, (uint32_t)source.width, (uint32_t)source.height,
        channels);
    if (normalization == nil || normalized == nil || activated == nil) {
        e2e_error(error, error_capacity,
                  "cannot create Video VAE group norm: %s", prefix);
        return 1;
    }
    id<MTLCommandBuffer> command = [metal->queue commandBuffer];
    [normalization encodeToCommandBuffer:command
                              sourceImage:source
                         destinationImage:normalized];
    if (h3_mps_wait(command, error, error_capacity) != 0 ||
        h3_mps_image_unary(metal, metal->image_silu, normalized, activated,
                           error, error_capacity) != 0)
        return 1;
    *destination = activated;
    return 0;
}

static int h3_mps_resnet_block(h3_remote_image *vae,
                               h3_metal *metal,
                               uint32_t level,
                               uint32_t block,
                               MPSImage *source,
                               MPSImage **destination,
                               char *error,
                               size_t error_capacity) {
    char prefix[160];
    char name[192];
    snprintf(prefix, sizeof(prefix), "encoder.down.%u.block.%u", level,
             block);
    MPSImage *activated = nil;
    MPSImage *hidden = nil;
    snprintf(name, sizeof(name), "%s.norm1", prefix);
    if (h3_mps_norm_silu(vae, metal, name, source, &activated, error,
                         error_capacity) != 0)
        return 1;
    snprintf(name, sizeof(name), "%s.conv1", prefix);
    if (h3_mps_convolution(vae, metal, name, activated, 1u, &hidden, error,
                           error_capacity) != 0)
        return 1;
    snprintf(name, sizeof(name), "%s.norm2", prefix);
    if (h3_mps_norm_silu(vae, metal, name, hidden, &activated, error,
                         error_capacity) != 0)
        return 1;
    snprintf(name, sizeof(name), "%s.conv2", prefix);
    if (h3_mps_convolution(vae, metal, name, activated, 1u, &hidden, error,
                           error_capacity) != 0)
        return 1;
    MPSImage *residual = source;
    if (source.featureChannels != hidden.featureChannels) {
        snprintf(name, sizeof(name), "%s.nin_shortcut", prefix);
        if (h3_mps_convolution(vae, metal, name, source, 1u, &residual, error,
                               error_capacity) != 0)
            return 1;
    }
    MPSImage *output = h3_mps_image(
        metal->device, (uint32_t)hidden.width, (uint32_t)hidden.height,
        (uint32_t)hidden.featureChannels);
    if (output == nil || h3_mps_image_add(metal, residual, hidden, output,
                                          error, error_capacity) != 0)
        return 1;
    *destination = output;
    return 0;
}

static int h3_load_condition_image(const char *path,
                                   uint32_t width,
                                   uint32_t height,
                                   uint16_t **pixels,
                                   char *error,
                                   size_t error_capacity) {
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    CGImageSourceRef source = CGImageSourceCreateWithURL(
        (__bridge CFURLRef)url, NULL);
    CGImageRef image = source != NULL
                           ? CGImageSourceCreateImageAtIndex(source, 0u, NULL)
                           : NULL;
    if (source != NULL) CFRelease(source);
    if (image == NULL) {
        e2e_error(error, error_capacity,
                  "cannot decode conditioning image: %s", path);
        return 1;
    }
    size_t source_width = CGImageGetWidth(image);
    size_t source_height = CGImageGetHeight(image);
    double source_ratio = (double)source_width / (double)source_height;
    double target_ratio = (double)width / (double)height;
    CGRect crop = CGRectMake(0.0, 0.0, (double)source_width,
                             (double)source_height);
    if (source_ratio > target_ratio) {
        double crop_width = (double)source_height * target_ratio;
        crop.origin.x = ((double)source_width - crop_width) * 0.5;
        crop.size.width = crop_width;
    } else if (source_ratio < target_ratio) {
        double crop_height = (double)source_width / target_ratio;
        crop.origin.y = ((double)source_height - crop_height) * 0.5;
        crop.size.height = crop_height;
    }
    CGImageRef cropped = CGImageCreateWithImageInRect(image, crop);
    size_t rgba_bytes = (size_t)width * height * 4u;
    uint8_t *rgba = calloc(rgba_bytes, 1u);
    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = rgba != NULL
        ? CGBitmapContextCreate(rgba, width, height, 8u, (size_t)width * 4u,
                                color_space,
                                kCGImageAlphaPremultipliedLast |
                                    kCGBitmapByteOrder32Big)
        : NULL;
    CGColorSpaceRelease(color_space);
    if (cropped == NULL || context == NULL) {
        if (cropped != NULL) CGImageRelease(cropped);
        CGImageRelease(image);
        if (context != NULL) CGContextRelease(context);
        free(rgba);
        e2e_error(error, error_capacity,
                  "cannot allocate conditioning image canvas");
        return 1;
    }
    CGContextSetInterpolationQuality(context, kCGInterpolationHigh);
    CGContextTranslateCTM(context, 0.0, height);
    CGContextScaleCTM(context, 1.0, -1.0);
    CGContextDrawImage(context, CGRectMake(0.0, 0.0, width, height), cropped);
    CGContextRelease(context);
    CGImageRelease(cropped);
    CGImageRelease(image);

    size_t values = (size_t)width * height * 3u;
    uint16_t *normalized = malloc(values * sizeof(uint16_t));
    if (normalized == NULL) {
        free(rgba);
        e2e_error(error, error_capacity,
                  "cannot allocate normalized conditioning image");
        return 1;
    }
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float standard_deviation[3] = {0.229f, 0.224f, 0.225f};
    for (size_t pixel = 0u; pixel < (size_t)width * height; ++pixel)
        for (uint32_t channel = 0u; channel < 3u; ++channel) {
            float value = (float)rgba[pixel * 4u + channel] / 255.0f;
            ((__fp16 *)normalized)[pixel * 3u + channel] =
                (__fp16)((value - mean[channel]) /
                         standard_deviation[channel]);
        }
    free(rgba);
    *pixels = normalized;
    return 0;
}

static int h3_video_encode_condition(const char *path,
                                     const minimax_h3_m3_e2e_options *options,
                                     h3_metal *metal,
                                     float **latents,
                                     double *seconds,
                                     size_t *peak_footprint,
                                     char *error,
                                     size_t error_capacity) {
    double started = e2e_now();
    h3_remote_image vae = {0};
    uint16_t *pixels = NULL;
    float *result = NULL;
    MPSImage *hidden = nil;
    MPSImage *next = nil;
    int status = 1;
    if (h3_remote_image_open("video_vae.safetensors", "video-vae", &vae,
                             error, error_capacity) != 0)
        return 1;
    if (h3_load_condition_image(path, options->width, options->height,
                                &pixels, error, error_capacity) != 0)
        goto cleanup;
    hidden = h3_mps_image(metal->device, options->width,
                          options->height, 3u);
    if (hidden == nil) {
        e2e_error(error, error_capacity,
                  "cannot allocate Video VAE input image");
        goto cleanup;
    }
    MTLRegion full = MTLRegionMake3D(0u, 0u, 0u, options->width,
                                     options->height, 1u);
    MPSImageReadWriteParams rw = {
        .featureChannelOffset = 0u,
        .numberOfFeatureChannelsToReadWrite = 3u,
    };
    [hidden writeBytes:pixels
            dataLayout:MPSDataLayoutHeightxWidthxFeatureChannels
           bytesPerRow:(NSUInteger)options->width * 3u * sizeof(uint16_t)
                region:full
    featureChannelInfo:rw
            imageIndex:0u];
    free(pixels);
    pixels = NULL;
    if (h3_mps_convolution(&vae, metal, "encoder.conv_in", hidden, 1u,
                           &next, error, error_capacity) != 0)
        goto cleanup;
    hidden = next;
    for (uint32_t level = 0u; level < 6u; ++level) {
        for (uint32_t block = 0u; block < 2u; ++block) {
            if (h3_mps_resnet_block(&vae, metal, level, block, hidden, &next,
                                    error, error_capacity) != 0)
                goto cleanup;
            hidden = next;
        }
        if (level < 4u) {
            char name[160];
            snprintf(name, sizeof(name), "encoder.down.%u.downsample.conv",
                     level);
            if (h3_mps_convolution(&vae, metal, name, hidden, 2u, &next,
                                   error, error_capacity) != 0)
                goto cleanup;
            hidden = next;
        }
        *peak_footprint = MAX(*peak_footprint, e2e_footprint());
        fprintf(stderr,
                "stage=image-vae path=%s level=%u/6 geometry=%lux%lux%lu "
                "footprint=%zu\n",
                path, level + 1u, (unsigned long)hidden.width,
                (unsigned long)hidden.height,
                (unsigned long)hidden.featureChannels, e2e_footprint());
        fflush(stderr);
    }
    if (h3_mps_norm_silu(&vae, metal, "encoder.norm_out", hidden, &next,
                         error, error_capacity) != 0 ||
        h3_mps_convolution(&vae, metal, "encoder.conv_out", next, 1u,
                           &hidden, error, error_capacity) != 0 ||
        h3_mps_convolution(&vae, metal, "quant_conv", hidden, 1u, &next,
                           error, error_capacity) != 0)
        goto cleanup;
    uint32_t latent_width = options->width / 16u;
    uint32_t latent_height = options->height / 16u;
    if (next.width != latent_width || next.height != latent_height ||
        next.featureChannels != 48u) {
        e2e_error(error, error_capacity,
                  "unexpected Video VAE latent geometry: %lux%lux%lu",
                  (unsigned long)next.width, (unsigned long)next.height,
                  (unsigned long)next.featureChannels);
        goto cleanup;
    }
    size_t moment_values = (size_t)latent_width * latent_height * 48u;
    uint16_t *moments = malloc(moment_values * sizeof(uint16_t));
    result = malloc((size_t)latent_width * latent_height * 24u *
                    sizeof(float));
    if (moments == NULL || result == NULL) {
        free(moments);
        e2e_error(error, error_capacity,
                  "cannot allocate Video VAE posterior");
        goto cleanup;
    }
    full = MTLRegionMake3D(0u, 0u, 0u, latent_width, latent_height, 1u);
    rw.numberOfFeatureChannelsToReadWrite = 48u;
    [next readBytes:moments
          dataLayout:MPSDataLayoutHeightxWidthxFeatureChannels
         bytesPerRow:(NSUInteger)latent_width * 48u * sizeof(uint16_t)
              region:full
  featureChannelInfo:rw
          imageIndex:0u];
    minimax_h3_remote_tensor mean_tensor;
    minimax_h3_remote_tensor std_tensor;
    const uint16_t *latent_mean = h3_f16_tensor_pointer(
        &vae, "latents_mean", &mean_tensor, error, error_capacity);
    const uint16_t *latent_std = h3_f16_tensor_pointer(
        &vae, "latents_std", &std_tensor, error, error_capacity);
    if (latent_mean == NULL || latent_std == NULL ||
        mean_tensor.rank != 1u || std_tensor.rank != 1u ||
        mean_tensor.shape[0] != 24u || std_tensor.shape[0] != 24u) {
        free(moments);
        goto cleanup;
    }
    uint64_t rng = UINT64_C(42);
    size_t spatial = (size_t)latent_width * latent_height;
    for (size_t pixel = 0u; pixel < spatial; ++pixel)
        for (uint32_t channel = 0u; channel < 24u; ++channel) {
            float mean = (float)((const __fp16 *)moments)[pixel * 48u +
                                                               channel];
            float log_variance = (float)((const __fp16 *)moments)[
                pixel * 48u + 24u + channel];
            log_variance = fmaxf(-30.0f, fminf(20.0f, log_variance));
            float sample = mean + expf(0.5f * log_variance) *
                                      h3_rng_normal(&rng);
            sample = (float)(__fp16)sample;
            result[(size_t)channel * spatial + pixel] =
                (sample - (float)((const __fp16 *)latent_mean)[channel]) /
                (float)((const __fp16 *)latent_std)[channel];
        }
    free(moments);
    *latents = result;
    result = NULL;
    *seconds = e2e_now() - started;
    fprintf(stderr,
            "stage=image-vae status=complete path=%s latent=%ux%ux24 "
            "seconds=%.6f\n",
            path, latent_width, latent_height, *seconds);
    fflush(stderr);
    status = 0;
cleanup:
    free(pixels);
    free(result);
    h3_remote_image_close(&vae);
    return status;
}

static int h3_write_ppm_frames(const char *directory,
                               const uint8_t *rgb,
                               uint32_t frames,
                               uint32_t width,
                               uint32_t height,
                               char *error,
                               size_t error_capacity) {
    if (mkdir(directory, 0755) != 0 && errno != EEXIST) {
        e2e_error(error, error_capacity, "cannot create frame directory: %s",
                  strerror(errno));
        return 1;
    }
    for (uint32_t frame = 0u; frame < frames; ++frame) {
        char path[1200];
        if (snprintf(path, sizeof(path), "%s/frame-%04u.ppm", directory,
                     frame) >= (int)sizeof(path)) {
            e2e_error(error, error_capacity, "frame path is too long");
            return 1;
        }
        FILE *file = fopen(path, "wb");
        if (file == NULL) {
            e2e_error(error, error_capacity, "cannot create %s: %s", path,
                      strerror(errno));
            return 1;
        }
        fprintf(file, "P6\n%u %u\n255\n", width, height);
        size_t bytes = (size_t)width * height * 3u;
        if (fwrite(rgb + (size_t)frame * bytes, 1u, bytes, file) != bytes ||
            fclose(file) != 0) {
            e2e_error(error, error_capacity, "cannot write %s", path);
            return 1;
        }
    }
    return 0;
}

static int h3_video_decode_raw_frame(const __fp16 *decoded,
                                     uint32_t latent_height,
                                     uint32_t latent_width,
                                     uint32_t raw_frame,
                                     uint32_t width,
                                     uint32_t height,
                                     float *rgb,
                                     char *error,
                                     size_t error_capacity) {
    const float rgb_mean[3] = {0.485f, 0.456f, 0.406f};
    const float rgb_std[3] = {0.229f, 0.224f, 0.225f};
    uint32_t latent_frame = raw_frame / 4u;
    uint32_t temporal_patch = raw_frame % 4u;
    for (uint32_t y = 0u; y < height; ++y)
        for (uint32_t x = 0u; x < width; ++x) {
            uint32_t latent_y = y / 16u;
            uint32_t latent_x = x / 16u;
            uint32_t py = y % 16u;
            uint32_t px = x % 16u;
            uint32_t token = (latent_frame * latent_height + latent_y) *
                             latent_width + latent_x;
            for (uint32_t channel = 0u; channel < 3u; ++channel) {
                uint32_t column = (((channel * 4u + temporal_patch) * 16u +
                                    py) * 16u + px);
                float value = (float)decoded[(size_t)token * 3072u + column] *
                              rgb_std[channel] + rgb_mean[channel];
                if (!isfinite(value)) {
                    e2e_error(error, error_capacity,
                              "video VAE produced a non-finite pixel");
                    return 1;
                }
                rgb[((size_t)y * width + x) * 3u + channel] = value;
            }
        }
    return 0;
}

static void h3_video_store_rgb(const float *source,
                               uint8_t *destination,
                               size_t elements) {
    for (size_t index = 0u; index < elements; ++index) {
        float value = fminf(1.0f, fmaxf(0.0f, source[index]));
        destination[index] = (uint8_t)lrintf(value * 255.0f);
    }
}

enum { H3_VIDEO_TILE_LIMIT = 64 };

typedef struct {
    uint32_t count;
    uint32_t starts[H3_VIDEO_TILE_LIMIT];
    uint32_t lengths[H3_VIDEO_TILE_LIMIT];
    uint32_t overlaps[H3_VIDEO_TILE_LIMIT - 1];
} h3_video_tile_plan;

static int h3_video_make_tile_plan(uint32_t length,
                                   uint32_t tile_size,
                                   h3_video_tile_plan *plan,
                                   char *error,
                                   size_t error_capacity) {
    const uint32_t minimum_overlap = 64u;
    memset(plan, 0, sizeof(*plan));
    if (length == 0u || length % 16u != 0u || tile_size < 128u ||
        tile_size % 16u != 0u) {
        e2e_error(error, error_capacity,
                  "VAE axis and tile size must be valid multiples of 16");
        return 1;
    }
    if (length <= tile_size) {
        plan->count = 1u;
        plan->lengths[0] = length;
        return 0;
    }
    uint32_t count = (length + tile_size - 1u) / tile_size;
    while ((uint64_t)tile_size * count -
               (uint64_t)minimum_overlap * (count - 1u) < length)
        ++count;
    if (count > H3_VIDEO_TILE_LIMIT) {
        e2e_error(error, error_capacity, "VAE tile count exceeds limit");
        return 1;
    }
    plan->count = count;
    uint32_t overlap_sum = 0u;
    for (uint32_t index = 0u; index + 1u < count; ++index) {
        plan->lengths[index] = tile_size;
        plan->overlaps[index] = minimum_overlap;
        overlap_sum += minimum_overlap;
    }
    plan->lengths[count - 1u] = tile_size;
    uint32_t remaining = tile_size * count - overlap_sum - length;
    for (uint32_t index = 0u; index < remaining / 16u; ++index)
        plan->overlaps[index % (count - 1u)] += 16u;
    for (uint32_t index = 1u; index < count; ++index)
        plan->starts[index] = plan->starts[index - 1u] + tile_size -
                              plan->overlaps[index - 1u];
    if (plan->starts[count - 1u] + plan->lengths[count - 1u] != length) {
        e2e_error(error, error_capacity, "VAE tile plan does not cover axis");
        return 1;
    }
    return 0;
}

static int h3_video_decode(const h3_latents *latents,
                           const minimax_h3_m3_e2e_options *options,
                           h3_metal *metal,
                           char *frame_directory,
                           size_t frame_directory_capacity,
                           double *download_seconds,
                           double *precompute_seconds,
                           double *seconds,
                           size_t *peak_footprint,
                           char *error,
                           size_t error_capacity) {
    h3_remote_image vae = {0};
    if (h3_remote_image_open("video_vae.safetensors", "video-vae", &vae,
                             error, error_capacity) != 0)
        return 1;
    *download_seconds = vae.download_seconds;
    int status = 1;
    uint8_t *rgb = NULL;
    float *frame_rgb = NULL;
    float *overlap_rgb = NULL;
    float *clip_rgb = NULL;
    float *tile_storage_a = NULL;
    float *tile_storage_b = NULL;
    h3_video_model_binding model = {0};
    double started = e2e_now();
    /* The released VAE is temporally decoded in overlapping clips and uses
     * 256-pixel spatial tiles with at least 64 pixels of overlap.  With
     * clip_length=17, ratio_t=4 and token_drop=3, every task consumes seven
     * latent frames, advances by five and returns 17 primary frames plus a
     * five-frame overlap. */
    const uint32_t chunk_latent_frames = 7u;
    const uint32_t chunk_stride = 5u;
    const uint32_t primary_frames = 17u;
    const uint32_t overlap_frames = 5u;
    uint32_t pseudo_tokens = latents->video_latent_frames + 3u;
    if (pseudo_tokens % chunk_stride != 0u ||
        pseudo_tokens / chunk_stride < 2u) {
        e2e_error(error, error_capacity,
                  "invalid VAE temporal length %u",
                  latents->video_latent_frames);
        h3_remote_image_close(&vae);
        return 1;
    }
    uint32_t chunk_count = pseudo_tokens / chunk_stride - 1u;
    /* At the 640x352 "360p" profile, a 256-pixel-high tile forces two
     * tiles to overlap by 160 pixels.  Three 160-pixel tiles retain the
     * required 64-pixel overlap while reducing both redundant projection
     * work and the quadratic attention footprint. */
    uint32_t tile_height = options->height == 352u ? 160u : 256u;
    uint32_t tile_width = 256u;
    const char *tile_height_text = getenv("MINIMAX_H3_VAE_TILE_HEIGHT");
    const char *tile_width_text = getenv("MINIMAX_H3_VAE_TILE_WIDTH");
    if (tile_height_text != NULL)
        tile_height = (uint32_t)strtoul(tile_height_text, NULL, 10);
    if (tile_width_text != NULL)
        tile_width = (uint32_t)strtoul(tile_width_text, NULL, 10);
    h3_video_tile_plan y_plan;
    h3_video_tile_plan x_plan;
    if (h3_video_make_tile_plan(options->height, tile_height, &y_plan, error,
                                error_capacity) != 0 ||
        h3_video_make_tile_plan(options->width, tile_width, &x_plan, error,
                                error_capacity) != 0) {
        h3_remote_image_close(&vae);
        return 1;
    }
    fprintf(stderr,
            "stage=video-vae-tiling tile=%ux%u grid=%ux%u chunks=%u\n",
            tile_width, tile_height, x_plan.count, y_plan.count, chunk_count);
    uint32_t maximum_latent_height = y_plan.lengths[0] / 16u;
    uint32_t maximum_latent_width = x_plan.lengths[0] / 16u;
    uint32_t maximum_tokens = chunk_latent_frames * maximum_latent_height *
                              maximum_latent_width;
    uint32_t maximum_rows = maximum_tokens + 5u;
    uint32_t maximum_padded = (maximum_rows + 63u) & ~63u;
    double precompute_started = e2e_now();
    size_t hidden_bytes = (size_t)maximum_padded * 2048u * 2u;
    id<MTLBuffer> latent_input = [metal->device
        newBufferWithLength:(size_t)maximum_tokens * 24u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> hidden = [metal->device newBufferWithLength:hidden_bytes
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> normalized = [metal->device newBufferWithLength:hidden_bytes
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> qkv = [metal->device
        newBufferWithLength:(size_t)maximum_padded * 6144u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> query = [metal->device newBufferWithLength:hidden_bytes
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> key = [metal->device newBufferWithLength:hidden_bytes
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> value = [metal->device newBufferWithLength:hidden_bytes
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> attended = [metal->device newBufferWithLength:hidden_bytes
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> attention_output = [metal->device newBufferWithLength:hidden_bytes
                                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> fc1 = [metal->device
        newBufferWithLength:(size_t)maximum_padded * 16384u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> gated = [metal->device
        newBufferWithLength:(size_t)maximum_padded * 8192u * 2u options:MTLResourceStorageModeShared];
    id<MTLBuffer> mlp_output = [metal->device newBufferWithLength:hidden_bytes
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> positions = [metal->device
        newBufferWithLength:(size_t)maximum_rows * 3u * 4u options:MTLResourceStorageModeShared];
    id<MTLBuffer> rotary = [metal->device
        newBufferWithLength:(size_t)maximum_rows * 24u * 2u * sizeof(float)
                    options:MTLResourceStorageModePrivate];
    id<MTLBuffer> pixels = [metal->device
        newBufferWithLength:(size_t)maximum_tokens * 3072u * 2u options:MTLResourceStorageModeShared];
    if (latent_input == nil || hidden == nil || normalized == nil || qkv == nil ||
        query == nil || key == nil || value == nil || attended == nil ||
        attention_output == nil || fc1 == nil || gated == nil ||
        mlp_output == nil || positions == nil || rotary == nil ||
        pixels == nil) {
        e2e_error(error, error_capacity, "video VAE activation allocation failed");
        goto cleanup;
    }
    if (h3_video_bind_weights(&vae, metal, &model, error,
                              error_capacity) != 0)
        goto cleanup;
    {
        float *pos = positions.contents;
        for (uint32_t frame = 0u; frame < chunk_latent_frames; ++frame)
            for (uint32_t y = 0u; y < maximum_latent_height; ++y)
                for (uint32_t x = 0u; x < maximum_latent_width; ++x) {
                    uint32_t token =
                        (frame * maximum_latent_height + y) *
                            maximum_latent_width + x;
                    pos[token * 3u] = (float)(
                        (2.0 * ((frame + 0.5) / chunk_latent_frames) - 1.0) *
                        6.283185307179586);
                    pos[token * 3u + 1u] = (float)(
                        (2.0 * ((y + 0.5) / maximum_latent_height) - 1.0) *
                        6.283185307179586);
                    pos[token * 3u + 2u] = (float)(
                        (2.0 * ((x + 0.5) / maximum_latent_width) - 1.0) *
                        6.283185307179586);
                }
        memset(pos + (size_t)maximum_tokens * 3u, 0,
               5u * 3u * sizeof(float));
        id<MTLCommandBuffer> command = [metal->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        [encoder setComputePipelineState:metal->video_rope];
        [encoder setBuffer:positions offset:0 atIndex:0];
        [encoder setBuffer:rotary offset:0 atIndex:1];
        [encoder setBytes:&maximum_rows length:sizeof(maximum_rows) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake((size_t)maximum_rows * 24u, 1u,
                                             1u)
                 threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
        if (h3_wait(command, error, error_capacity) != 0) goto cleanup;
    }
    *precompute_seconds = e2e_now() - precompute_started;
    started = e2e_now();
    minimax_h3_remote_tensor mean_tensor, std_tensor, pqw_tensor, pqb_tensor;
    const uint16_t *mean = h3_f16_tensor_pointer(&vae, "latents_mean",
        &mean_tensor, error, error_capacity);
    const uint16_t *std = h3_f16_tensor_pointer(&vae, "latents_std",
        &std_tensor, error, error_capacity);
    const uint16_t *pqw = h3_f16_tensor_pointer(&vae, "post_quant_conv.weight",
        &pqw_tensor, error, error_capacity);
    const uint16_t *pqb = h3_f16_tensor_pointer(&vae, "post_quant_conv.bias",
        &pqb_tensor, error, error_capacity);
    if (mean == NULL || std == NULL || pqw == NULL || pqb == NULL)
        goto cleanup;
    minimax_h3_remote_tensor registers_tensor;
    const uint16_t *registers = h3_f16_tensor_pointer(
        &vae, "decoder.register_tokens", &registers_tensor, error,
        error_capacity);
    if (registers == NULL) goto cleanup;
    const uint32_t raw_frames = chunk_latent_frames * 4u;
    size_t rgb_bytes = (size_t)options->frames * options->width *
                       options->height * 3u;
    size_t frame_elements = (size_t)options->width * options->height * 3u;
    size_t maximum_tile_frame_elements =
        (size_t)y_plan.lengths[0] * x_plan.lengths[0] * 3u;
    size_t maximum_tile_values =
        (size_t)raw_frames * maximum_tile_frame_elements;
    rgb = malloc(rgb_bytes);
    frame_rgb = malloc(frame_elements * sizeof(float));
    overlap_rgb = malloc((size_t)overlap_frames * frame_elements *
                         sizeof(float));
    clip_rgb = malloc((size_t)raw_frames * frame_elements * sizeof(float));
    tile_storage_a = malloc((size_t)x_plan.count * maximum_tile_values *
                            sizeof(float));
    tile_storage_b = malloc((size_t)x_plan.count * maximum_tile_values *
                            sizeof(float));
    if (rgb == NULL || frame_rgb == NULL || overlap_rgb == NULL ||
        clip_rgb == NULL || tile_storage_a == NULL || tile_storage_b == NULL) {
        e2e_error(error, error_capacity, "RGB tile allocation failed");
        goto cleanup;
    }
    float *previous_tiles = tile_storage_a;
    float *current_tiles = tile_storage_b;
    uint32_t output_frame = 0u;
    for (uint32_t chunk = 0u; chunk < chunk_count; ++chunk) {
        uint32_t source_frame0 = chunk * chunk_stride;
        if (source_frame0 + chunk_latent_frames >
            latents->video_latent_frames) {
            e2e_error(error, error_capacity,
                      "VAE chunk %u exceeds latent timeline", chunk);
            goto cleanup;
        }
        memset(clip_rgb, 0,
               (size_t)raw_frames * frame_elements * sizeof(float));
        for (uint32_t tile_y = 0u; tile_y < y_plan.count; ++tile_y) {
            uint32_t tile_height = y_plan.lengths[tile_y];
            uint32_t tile_latent_height = tile_height / 16u;
            uint32_t source_y0 = y_plan.starts[tile_y] / 16u;
            for (uint32_t tile_x = 0u; tile_x < x_plan.count; ++tile_x) {
                uint32_t tile_width = x_plan.lengths[tile_x];
                uint32_t tile_latent_width = tile_width / 16u;
                uint32_t source_x0 = x_plan.starts[tile_x] / 16u;
                uint32_t tokens = chunk_latent_frames * tile_latent_height *
                                  tile_latent_width;
                uint32_t rows = tokens + 5u;
                __fp16 *input_values = latent_input.contents;
                for (uint32_t frame = 0u; frame < chunk_latent_frames;
                     ++frame)
                    for (uint32_t y = 0u; y < tile_latent_height; ++y)
                        for (uint32_t x = 0u; x < tile_latent_width; ++x) {
                            uint32_t token =
                                (frame * tile_latent_height + y) *
                                    tile_latent_width + x;
                            float source_values[24];
                            for (uint32_t channel = 0u; channel < 24u;
                                 ++channel) {
                                size_t source =
                                    (((size_t)channel *
                                          latents->video_latent_frames +
                                      source_frame0 + frame) *
                                         latents->latent_height +
                                     source_y0 + y) *
                                        latents->latent_width +
                                    source_x0 + x;
                                source_values[channel] =
                                    latents->video[source] *
                                        (float)((const __fp16 *)std)[channel] +
                                    (float)((const __fp16 *)mean)[channel];
                            }
                            for (uint32_t output = 0u; output < 24u;
                                 ++output) {
                                float sum =
                                    (float)((const __fp16 *)pqb)[output];
                                for (uint32_t input = 0u; input < 24u;
                                     ++input)
                                    sum += source_values[input] *
                                        (float)((const __fp16 *)pqw)[
                                            output * 24u + input];
                                input_values[(size_t)token * 24u + output] =
                                    (__fp16)sum;
                            }
                        }
                memset(hidden.contents, 0, hidden_bytes);
                {
                    id<MTLCommandBuffer> command =
                        [metal->queue commandBuffer];
                    id<MTLComputeCommandEncoder> encoder =
                        [command computeCommandEncoder];
                    if (h3_dense_linear_bound(
                            &model.x_embedder, metal, encoder, latent_input,
                            hidden, tokens, 24u, error,
                            error_capacity) != 0) {
                        [encoder endEncoding];
                        goto cleanup;
                    }
                    [encoder endEncoding];
                    if (h3_wait(command, error, error_capacity) != 0)
                        goto cleanup;
                }
                memcpy((uint16_t *)hidden.contents +
                           (size_t)tokens * 2048u,
                       registers, 4u * 2048u * 2u);
                if (rows != maximum_rows) {
                    e2e_error(error, error_capacity,
                              "video VAE tile geometry changed within a run");
                    goto cleanup;
                }
                /* Four ViT3D blocks per command buffer keeps each submission
                 * below a long-running GPU watchdog interval while removing
                 * 27 of the former 36 CPU waits per tile. */
                for (unsigned layer0 = 0u; layer0 < 36u; layer0 += 4u) {
                    id<MTLCommandBuffer> command =
                        [metal->queue commandBuffer];
                    id<MTLComputeCommandEncoder> encoder =
                        [command computeCommandEncoder];
                    unsigned layer_end = layer0 + 4u < 36u
                                             ? layer0 + 4u
                                             : 36u;
                    for (unsigned layer = layer0; layer < layer_end; ++layer)
                        if (h3_video_block_encode(
                                &model.blocks[layer], metal, encoder, hidden,
                                normalized, qkv, query, key, value, attended,
                                attention_output, fc1, gated, mlp_output,
                                rotary, rows, error, error_capacity) != 0) {
                            [encoder endEncoding];
                            goto cleanup;
                        }
                    [encoder endEncoding];
                    if (h3_wait(command, error, error_capacity) != 0)
                        goto cleanup;
                    fprintf(stderr,
                            "stage=video-decode chunk=%u/%u tile=%u,%u/%u,%u "
                            "layer=%u/36 footprint=%zu\n",
                            chunk + 1u, chunk_count, tile_y + 1u, tile_x + 1u,
                            y_plan.count, x_plan.count, layer_end,
                            e2e_footprint());
                }
                {
                    h3_norm_parameters parameters = {
                        .rows = rows,
                        .columns = 2048u,
                        .epsilon = 1e-5f,
                    };
                    id<MTLCommandBuffer> command =
                        [metal->queue commandBuffer];
                    id<MTLComputeCommandEncoder> encoder =
                        [command computeCommandEncoder];
                    [encoder setComputePipelineState:metal->layernorm_f16];
                    [encoder setBuffer:hidden offset:0 atIndex:0];
                    h3_set(encoder, model.norm_out_weight, 1);
                    h3_set(encoder, model.norm_out_bias, 2);
                    [encoder setBuffer:normalized offset:0 atIndex:3];
                    [encoder setBytes:&parameters
                               length:sizeof(parameters)
                              atIndex:4];
                    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1u, 1u)
                                threadsPerThreadgroup:MTLSizeMake(256u, 1u,
                                                                 1u)];
                    if (h3_dense_linear_bound(
                            &model.projection_out, metal, encoder, normalized,
                            pixels, tokens, 2048u, error,
                            error_capacity) != 0) {
                        [encoder endEncoding];
                        goto cleanup;
                    }
                    [encoder endEncoding];
                    if (h3_wait(command, error, error_capacity) != 0)
                        goto cleanup;
                }
                float *current = current_tiles +
                    (size_t)tile_x * maximum_tile_values;
                size_t tile_frame_elements =
                    (size_t)tile_height * tile_width * 3u;
                for (uint32_t raw_frame = 0u; raw_frame < raw_frames;
                     ++raw_frame)
                    if (h3_video_decode_raw_frame(
                            pixels.contents, tile_latent_height,
                            tile_latent_width, raw_frame, tile_width,
                            tile_height,
                            current +
                                (size_t)raw_frame * tile_frame_elements,
                            error, error_capacity) != 0)
                        goto cleanup;

                uint32_t kept_height = tile_height -
                    (tile_y + 1u < y_plan.count
                         ? y_plan.overlaps[tile_y]
                         : 0u);
                uint32_t kept_width = tile_width -
                    (tile_x + 1u < x_plan.count
                         ? x_plan.overlaps[tile_x]
                         : 0u);
                for (uint32_t raw_frame = 0u; raw_frame < raw_frames;
                     ++raw_frame)
                    for (uint32_t y = 0u; y < kept_height; ++y)
                        for (uint32_t x = 0u; x < kept_width; ++x)
                            for (uint32_t channel = 0u; channel < 3u;
                                 ++channel) {
                                size_t local =
                                    ((size_t)raw_frame * tile_height *
                                         tile_width +
                                     (size_t)y * tile_width + x) *
                                        3u +
                                    channel;
                                float value = current[local];
                                if (tile_y != 0u &&
                                    y < y_plan.overlaps[tile_y - 1u]) {
                                    const float *above = previous_tiles +
                                        (size_t)tile_x * maximum_tile_values;
                                    uint32_t source_y =
                                        y_plan.lengths[tile_y - 1u] -
                                        y_plan.overlaps[tile_y - 1u] + y;
                                    size_t above_index =
                                        ((size_t)raw_frame *
                                             y_plan.lengths[tile_y - 1u] *
                                             tile_width +
                                         (size_t)source_y * tile_width + x) *
                                            3u +
                                        channel;
                                    float weight_b =
                                        (float)y /
                                        (float)y_plan.overlaps[tile_y - 1u];
                                    value = above[above_index] *
                                                (1.0f - weight_b) +
                                            value * weight_b;
                                }
                                if (tile_x != 0u &&
                                    x < x_plan.overlaps[tile_x - 1u]) {
                                    const float *left = current_tiles +
                                        (size_t)(tile_x - 1u) *
                                            maximum_tile_values;
                                    uint32_t left_width =
                                        x_plan.lengths[tile_x - 1u];
                                    uint32_t source_x = left_width -
                                        x_plan.overlaps[tile_x - 1u] + x;
                                    size_t left_index =
                                        ((size_t)raw_frame * tile_height *
                                             left_width +
                                         (size_t)y * left_width + source_x) *
                                            3u +
                                        channel;
                                    float weight_b =
                                        (float)x /
                                        (float)x_plan.overlaps[tile_x - 1u];
                                    value = left[left_index] *
                                                (1.0f - weight_b) +
                                            value * weight_b;
                                }
                                size_t destination =
                                    ((size_t)raw_frame * options->height *
                                         options->width +
                                     (size_t)(y_plan.starts[tile_y] + y) *
                                         options->width +
                                     x_plan.starts[tile_x] + x) *
                                        3u +
                                    channel;
                                clip_rgb[destination] = value;
                            }
            }
            float *swap = previous_tiles;
            previous_tiles = current_tiles;
            current_tiles = swap;
        }
        for (uint32_t local = 0u; local < primary_frames; ++local) {
            memcpy(frame_rgb,
                   clip_rgb + (size_t)(3u + local) * frame_elements,
                   frame_elements * sizeof(float));
            if (chunk != 0u && local < overlap_frames) {
                float weight_b = (float)local / (float)overlap_frames;
                float weight_a = 1.0f - weight_b;
                const float *previous = overlap_rgb +
                    (size_t)local * frame_elements;
                for (size_t element = 0u; element < frame_elements; ++element)
                    frame_rgb[element] = previous[element] * weight_a +
                                         frame_rgb[element] * weight_b;
            }
            h3_video_store_rgb(frame_rgb,
                rgb + (size_t)output_frame * frame_elements, frame_elements);
            ++output_frame;
        }
        for (uint32_t local = 0u; local < overlap_frames; ++local)
            memcpy(overlap_rgb + (size_t)local * frame_elements,
                   clip_rgb + (size_t)(23u + local) * frame_elements,
                   frame_elements * sizeof(float));
    }
    for (uint32_t local = 0u; local < overlap_frames; ++local) {
        h3_video_store_rgb(overlap_rgb + (size_t)local * frame_elements,
            rgb + (size_t)output_frame * frame_elements, frame_elements);
        ++output_frame;
    }
    if (output_frame != options->frames ||
        snprintf(frame_directory, frame_directory_capacity, "%s/frames",
                 options->output_directory) >= (int)frame_directory_capacity ||
        h3_write_ppm_frames(frame_directory, rgb, output_frame,
                            options->width, options->height, error,
                            error_capacity) != 0) {
        if (output_frame != options->frames)
            e2e_error(error, error_capacity, "video VAE emitted %u/%u frames",
                      output_frame, options->frames);
        goto cleanup;
    }
    *seconds = e2e_now() - started;
    *peak_footprint = MAX(*peak_footprint, e2e_footprint());
    status = 0;
cleanup:
    free(tile_storage_b);
    free(tile_storage_a);
    free(clip_rgb);
    free(overlap_rgb);
    free(frame_rgb);
    free(rgb);
    h3_remote_image_close(&vae);
    return status;
}

static int h3_audio_conv(h3_remote_image *weights,
                         h3_metal *metal,
                         const char *prefix,
                         id<MTLBuffer> input,
                         uint32_t input_length,
                         uint32_t input_channels,
                         uint32_t stride,
                         uint32_t padding,
                         uint32_t dilation,
                         int transposed,
                         int has_bias,
                         id<MTLBuffer> __strong *output,
                         uint32_t *output_length,
                         uint32_t *output_channels,
                         char *error,
                         size_t error_capacity) {
    char name[192];
    h3_tensor_binding weight = { {0}, nil, 0u };
    h3_tensor_binding bias = { {0}, nil, 0u };
    snprintf(name, sizeof(name), "%s.weight", prefix);
    if (h3_bind_tensor(weights, metal, name, &weight, error,
                       error_capacity) != 0)
        return 1;
    if (strcmp(weight.tensor.dtype, "F32") != 0 ||
        weight.tensor.rank != 3u) {
        e2e_error(error, error_capacity, "invalid audio convolution: %s",
                  prefix);
        return 1;
    }
    uint32_t in_channels = (uint32_t)weight.tensor.shape[transposed ? 0u : 1u];
    uint32_t out_channels = (uint32_t)weight.tensor.shape[transposed ? 1u : 0u];
    uint32_t kernel = (uint32_t)weight.tensor.shape[2];
    if (in_channels != input_channels || stride == 0u || dilation == 0u) {
        e2e_error(error, error_capacity, "audio convolution shape mismatch: %s",
                  prefix);
        return 1;
    }
    uint32_t length;
    if (transposed) {
        length = (input_length - 1u) * stride - 2u * padding + kernel;
    } else {
        uint64_t receptive = (uint64_t)dilation * (kernel - 1u) + 1u;
        if ((uint64_t)input_length + 2u * padding < receptive) {
            e2e_error(error, error_capacity, "audio convolution is empty: %s",
                      prefix);
            return 1;
        }
        length = (uint32_t)(((uint64_t)input_length + 2u * padding -
                             receptive) / stride + 1u);
    }
    if (has_bias) {
        snprintf(name, sizeof(name), "%s.bias", prefix);
        if (h3_bind_tensor(weights, metal, name, &bias, error,
                           error_capacity) != 0)
            return 1;
    }
    id<MTLBuffer> result = [metal->device
        newBufferWithLength:(size_t)2u * length * out_channels * sizeof(float)
                    options:MTLResourceStorageModeShared];
    if (result == nil) {
        e2e_error(error, error_capacity, "audio activation allocation failed");
        return 1;
    }
    h3_audio_conv_parameters parameters = {
        .batch = 2u,
        .input_length = input_length,
        .output_length = length,
        .input_channels = input_channels,
        .output_channels = out_channels,
        .kernel = kernel,
        .stride = stride,
        .padding = padding,
        .dilation = dilation,
    };
    id<MTLCommandBuffer> command = [metal->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:transposed ? metal->audio_conv_transpose
                                                  : metal->audio_conv];
    [encoder setBuffer:input offset:0 atIndex:0];
    h3_set(encoder, weight, 1);
    if (has_bias) h3_set(encoder, bias, 2);
    else [encoder setBuffer:result offset:0 atIndex:2];
    [encoder setBuffer:result offset:0 atIndex:3];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
    uint32_t bias_flag = has_bias ? 1u : 0u;
    if (!transposed)
        [encoder setBytes:&bias_flag length:sizeof(bias_flag) atIndex:5];
    uint64_t count = UINT64_C(2) * length * out_channels;
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)count, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder endEncoding];
    if (h3_wait(command, error, error_capacity) != 0) return 1;
    *output = result;
    *output_length = length;
    *output_channels = out_channels;
    return 0;
}

static int h3_audio_alias(h3_remote_image *weights,
                          h3_metal *metal,
                          const char *prefix,
                          id<MTLBuffer> input,
                          uint32_t length,
                          uint32_t channels,
                          id<MTLBuffer> __strong *output,
                          char *error,
                          size_t error_capacity) {
    char name[192];
    h3_tensor_binding alpha = { {0}, nil, 0u };
    h3_tensor_binding beta = { {0}, nil, 0u };
    snprintf(name, sizeof(name), "%s.act.alpha", prefix);
    if (h3_bind_tensor(weights, metal, name, &alpha, error,
                       error_capacity) != 0)
        return 1;
    snprintf(name, sizeof(name), "%s.act.beta", prefix);
    if (h3_bind_tensor(weights, metal, name, &beta, error,
                       error_capacity) != 0)
        return 1;
    id<MTLBuffer> result = [metal->device
        newBufferWithLength:(size_t)2u * length * channels * sizeof(float)
                    options:MTLResourceStorageModeShared];
    if (result == nil) {
        e2e_error(error, error_capacity, "audio alias allocation failed");
        return 1;
    }
    id<MTLCommandBuffer> command = [metal->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:metal->audio_alias];
    [encoder setBuffer:input offset:0 atIndex:0];
    h3_set(encoder, alpha, 1);
    h3_set(encoder, beta, 2);
    [encoder setBuffer:result offset:0 atIndex:3];
    uint32_t batch = 2u;
    [encoder setBytes:&batch length:sizeof(batch) atIndex:4];
    [encoder setBytes:&length length:sizeof(length) atIndex:5];
    [encoder setBytes:&channels length:sizeof(channels) atIndex:6];
    [encoder dispatchThreads:MTLSizeMake((size_t)2u * length * channels, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder endEncoding];
    if (h3_wait(command, error, error_capacity) != 0) return 1;
    *output = result;
    return 0;
}

static int h3_audio_residual_block(h3_remote_image *weights,
                                   h3_metal *metal,
                                   unsigned block,
                                   unsigned kernel,
                                   id<MTLBuffer> input,
                                   uint32_t length,
                                   uint32_t channels,
                                   id<MTLBuffer> __strong *output,
                                   char *error,
                                   size_t error_capacity) {
    size_t bytes = (size_t)2u * length * channels * sizeof(float);
    id<MTLBuffer> hidden = [metal->device newBufferWithLength:bytes
                                                     options:MTLResourceStorageModeShared];
    if (hidden == nil) {
        e2e_error(error, error_capacity, "audio residual allocation failed");
        return 1;
    }
    memcpy(hidden.contents, input.contents, bytes);
    const uint32_t dilations[3] = {1u, 3u, 5u};
    for (unsigned layer = 0u; layer < 3u; ++layer) {
        char prefix[192];
        id<MTLBuffer> activated = nil;
        id<MTLBuffer> update = nil;
        uint32_t update_length = 0u;
        uint32_t update_channels = 0u;
        snprintf(prefix, sizeof(prefix),
                 "decoder.resblocks.%u.activations.%u", block, layer * 2u);
        if (h3_audio_alias(weights, metal, prefix, hidden, length, channels,
                           &activated, error, error_capacity) != 0)
            return 1;
        snprintf(prefix, sizeof(prefix), "decoder.resblocks.%u.convs1.%u",
                 block, layer);
        uint32_t padding = (kernel * dilations[layer] - dilations[layer]) / 2u;
        if (h3_audio_conv(weights, metal, prefix, activated, length, channels,
                          1u, padding, dilations[layer], 0, 1, &update,
                          &update_length, &update_channels, error,
                          error_capacity) != 0)
            return 1;
        snprintf(prefix, sizeof(prefix),
                 "decoder.resblocks.%u.activations.%u", block,
                 layer * 2u + 1u);
        if (h3_audio_alias(weights, metal, prefix, update, update_length,
                           update_channels, &activated, error,
                           error_capacity) != 0)
            return 1;
        snprintf(prefix, sizeof(prefix), "decoder.resblocks.%u.convs2.%u",
                 block, layer);
        if (h3_audio_conv(weights, metal, prefix, activated, update_length,
                          update_channels, 1u, (kernel - 1u) / 2u, 1u, 0, 1,
                          &update, &update_length, &update_channels, error,
                          error_capacity) != 0)
            return 1;
        if (update_length != length || update_channels != channels) {
            e2e_error(error, error_capacity,
                      "audio residual changed geometry at block %u", block);
            return 1;
        }
        id<MTLCommandBuffer> command = [metal->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        uint32_t count = 2u * length * channels;
        [encoder setComputePipelineState:metal->audio_residual];
        [encoder setBuffer:hidden offset:0 atIndex:0];
        [encoder setBuffer:update offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                 threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
        if (h3_wait(command, error, error_capacity) != 0) return 1;
    }
    *output = hidden;
    return 0;
}

static int h3_write_wav(const char *path,
                        const float *samples,
                        uint32_t source_length,
                        uint32_t requested_length,
                        char *error,
                        size_t error_capacity) {
    uint32_t length = requested_length < source_length ? requested_length
                                                        : source_length;
    uint32_t data_bytes = length * 2u * 2u;
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        e2e_error(error, error_capacity, "cannot create WAV: %s",
                  strerror(errno));
        return 1;
    }
    uint32_t riff_size = 36u + data_bytes;
    uint32_t fmt_size = 16u;
    uint16_t format = 1u, channels = 2u, bits = 16u;
    uint32_t rate = 32000u, byte_rate = rate * channels * bits / 8u;
    uint16_t align = channels * bits / 8u;
    fwrite("RIFF", 1u, 4u, file); fwrite(&riff_size, 4u, 1u, file);
    fwrite("WAVEfmt ", 1u, 8u, file); fwrite(&fmt_size, 4u, 1u, file);
    fwrite(&format, 2u, 1u, file); fwrite(&channels, 2u, 1u, file);
    fwrite(&rate, 4u, 1u, file); fwrite(&byte_rate, 4u, 1u, file);
    fwrite(&align, 2u, 1u, file); fwrite(&bits, 2u, 1u, file);
    fwrite("data", 1u, 4u, file); fwrite(&data_bytes, 4u, 1u, file);
    for (uint32_t time = 0u; time < length; ++time)
        for (uint32_t channel = 0u; channel < 2u; ++channel) {
            float value = fminf(1.0f, fmaxf(-1.0f,
                samples[((size_t)channel * source_length + time)]));
            int16_t pcm = (int16_t)lrintf(value * 32767.0f);
            fwrite(&pcm, sizeof(pcm), 1u, file);
        }
    if (fclose(file) != 0) {
        e2e_error(error, error_capacity, "cannot finalize WAV");
        return 1;
    }
    return 0;
}

static int h3_audio_decode(const h3_latents *latents,
                           const minimax_h3_m3_e2e_options *options,
                           h3_metal *metal,
                           char *wav_path,
                           size_t wav_path_capacity,
                           double *download_seconds,
                           double *seconds,
                           size_t *peak_footprint,
                           char *error,
                           size_t error_capacity) {
    h3_remote_image vae = {0};
    if (h3_remote_image_open("audio_vae.safetensors", "audio-vae", &vae,
                             error, error_capacity) != 0)
        return 1;
    *download_seconds = vae.download_seconds;
    int status = 1;
    double started = e2e_now();
    uint32_t length = latents->audio_latent_frames;
    uint32_t channels = 32u;
    id<MTLBuffer> hidden = [metal->device
        newBufferWithLength:(size_t)2u * length * channels * sizeof(float)
                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> activated = nil;
    if (hidden == nil) {
        e2e_error(error, error_capacity, "audio input allocation failed");
        goto cleanup;
    }
    minimax_h3_remote_tensor mean_tensor, std_tensor;
    if (minimax_h3_remote_safetensors_find(&vae.remote, "latents_mean",
                                            &mean_tensor, error,
                                            error_capacity) != 0 ||
        minimax_h3_remote_safetensors_find(&vae.remote, "latents_std",
                                            &std_tensor, error,
                                            error_capacity) != 0)
        goto cleanup;
    const float *mean = (const float *)((const uint8_t *)vae.mapping +
        vae.file_padding + mean_tensor.data_start);
    const float *std = (const float *)((const uint8_t *)vae.mapping +
        vae.file_padding + std_tensor.data_start);
    float *input = hidden.contents;
    for (uint32_t batch = 0u; batch < 2u; ++batch)
        for (uint32_t time = 0u; time < length; ++time)
            for (uint32_t feature = 0u; feature < 32u; ++feature) {
                size_t source = ((size_t)feature * 2u + batch) * length + time;
                input[((size_t)batch * length + time) * 32u + feature] =
                    latents->audio[source] * std[feature] + mean[feature];
            }
    uint32_t next_length, next_channels;
    if (h3_audio_conv(&vae, metal, "dec_in_proj", hidden, length, channels,
                      1u, 0u, 1u, 0, 1, &hidden, &next_length, &next_channels,
                      error, error_capacity) != 0)
        goto cleanup;
    length = next_length; channels = next_channels;
    if (h3_audio_conv(&vae, metal, "decoder.conv_pre", hidden, length,
                      channels, 1u, 3u, 1u, 0, 1, &hidden, &next_length,
                      &next_channels, error, error_capacity) != 0)
        goto cleanup;
    length = next_length; channels = next_channels;
    const uint32_t rates[7] = {5u, 5u, 2u, 2u, 2u, 2u, 2u};
    const uint32_t kernels[7] = {9u, 9u, 4u, 4u, 4u, 4u, 4u};
    const uint32_t residual_kernels[3] = {3u, 7u, 11u};
    for (unsigned stage = 0u; stage < 7u; ++stage) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "decoder.ups.%u.0", stage);
        if (h3_audio_conv(&vae, metal, prefix, hidden, length, channels,
                          rates[stage], (kernels[stage] - rates[stage]) / 2u,
                          1u, 1, 1, &hidden, &next_length, &next_channels,
                          error, error_capacity) != 0)
            goto cleanup;
        length = next_length; channels = next_channels;
        id<MTLBuffer> branch[3] = {nil, nil, nil};
        for (unsigned index = 0u; index < 3u; ++index) {
            if (h3_audio_residual_block(&vae, metal, stage * 3u + index,
                                        residual_kernels[index], hidden,
                                        length, channels, &branch[index],
                                        error, error_capacity) != 0)
                goto cleanup;
        }
        id<MTLBuffer> averaged = [metal->device
            newBufferWithLength:(size_t)2u * length * channels * sizeof(float)
                        options:MTLResourceStorageModeShared];
        if (averaged == nil) {
            e2e_error(error, error_capacity, "audio average allocation failed");
            goto cleanup;
        }
        id<MTLCommandBuffer> command = [metal->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        uint32_t count = 2u * length * channels;
        [encoder setComputePipelineState:metal->audio_average3];
        [encoder setBuffer:branch[0] offset:0 atIndex:0];
        [encoder setBuffer:branch[1] offset:0 atIndex:1];
        [encoder setBuffer:branch[2] offset:0 atIndex:2];
        [encoder setBuffer:averaged offset:0 atIndex:3];
        [encoder setBytes:&count length:sizeof(count) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                 threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
        if (h3_wait(command, error, error_capacity) != 0) goto cleanup;
        hidden = averaged;
        *peak_footprint = MAX(*peak_footprint, e2e_footprint());
        fprintf(stderr, "stage=audio-decode upsample=%u/7 length=%u channels=%u footprint=%zu\n",
                stage + 1u, length, channels, e2e_footprint());
    }
    if (h3_audio_alias(&vae, metal, "decoder.activation_post", hidden, length,
                       channels, &activated, error, error_capacity) != 0)
        goto cleanup;
    if (h3_audio_conv(&vae, metal, "decoder.conv_post", activated, length,
                      channels, 1u, 3u, 1u, 0, 0, &hidden, &next_length,
                      &next_channels, error, error_capacity) != 0)
        goto cleanup;
    length = next_length; channels = next_channels;
    const float *waveform = hidden.contents;
    for (size_t index = 0u; index < (size_t)2u * length; ++index) {
        if (!isfinite(waveform[index])) {
            e2e_error(error, error_capacity,
                      "audio VAE produced a non-finite sample");
            goto cleanup;
        }
    }
    if (channels != 1u ||
        snprintf(wav_path, wav_path_capacity, "%s/audio.wav",
                 options->output_directory) >= (int)wav_path_capacity ||
        h3_write_wav(wav_path, hidden.contents, length,
                     (uint32_t)llround((double)options->frames * 32000.0 / 24.0),
                     error, error_capacity) != 0)
        goto cleanup;
    *seconds = e2e_now() - started;
    *peak_footprint = MAX(*peak_footprint, e2e_footprint());
    status = 0;
cleanup:
    h3_remote_image_close(&vae);
    return status;
}

static int h3_mux(const char *frame_directory,
                  const char *wav_path,
                  const char *output_directory,
                  char *video_path,
                  size_t video_path_capacity,
                  char *output_path,
                  size_t output_path_capacity,
                  double *seconds,
                  char *error,
                  size_t error_capacity) {
    char pattern[1200];
    char silent[1200];
    char final[1200];
    if (snprintf(pattern, sizeof(pattern), "%s/frame-%%04d.ppm",
                 frame_directory) >= (int)sizeof(pattern) ||
        snprintf(silent, sizeof(silent), "%s/video.mp4",
                 output_directory) >= (int)sizeof(silent) ||
        snprintf(final, sizeof(final), "%s/minimax-h3.mp4",
                 output_directory) >= (int)sizeof(final)) {
        e2e_error(error, error_capacity, "media path is too long");
        return 1;
    }
    double started = e2e_now();
    /* ffmpeg's default CRF 23 rings visibly on dark anime edges -
     * ghost lines parallel to every high-contrast contour. CRF 14 is
     * visually transparent for this content and the file stays small
     * next to the frames it encodes. */
    char *video_arguments[] = {
        "/opt/homebrew/bin/ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-framerate", "24", "-i", pattern, "-c:v", "libx264",
        "-crf", "14", "-preset", "slow",
        "-pix_fmt", "yuv420p", silent, NULL,
    };
    pid_t child;
    int spawn_status = posix_spawn(&child, video_arguments[0], NULL, NULL,
                                   video_arguments, environ);
    int wait_status = 0;
    if (spawn_status != 0 || waitpid(child, &wait_status, 0) < 0 ||
        !WIFEXITED(wait_status) || WEXITSTATUS(wait_status) != 0) {
        e2e_error(error, error_capacity, "ffmpeg video encode failed");
        return 1;
    }
    char *mux_arguments[] = {
        "/opt/homebrew/bin/ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", silent, "-i", (char *)wav_path, "-c:v", "copy",
        "-c:a", "aac", final, NULL,
    };
    spawn_status = posix_spawn(&child, mux_arguments[0], NULL, NULL,
                               mux_arguments, environ);
    wait_status = 0;
    if (spawn_status != 0 || waitpid(child, &wait_status, 0) < 0 ||
        !WIFEXITED(wait_status) || WEXITSTATUS(wait_status) != 0) {
        e2e_error(error, error_capacity, "ffmpeg A/V mux failed");
        return 1;
    }
    struct stat info;
    if (stat(final, &info) != 0 || info.st_size <= 0) {
        e2e_error(error, error_capacity, "mux output is missing or empty");
        return 1;
    }
    snprintf(video_path, video_path_capacity, "%s", silent);
    snprintf(output_path, output_path_capacity, "%s", final);
    *seconds = e2e_now() - started;
    return 0;
}

int minimax_h3_m3_run_downstream_smoke(
    const minimax_h3_m3_e2e_options *options,
    minimax_h3_m3_e2e_result *result,
    char *error,
    size_t error_capacity) {
    if (options == NULL || result == NULL || options->metallib_path == NULL ||
        options->output_directory == NULL) {
        e2e_error(error, error_capacity, "invalid downstream smoke options");
        return 2;
    }
    @autoreleasepool {
        h3_metal metal;
        minimax_h3_m3_e2e_result measured;
        memset(&measured, 0, sizeof(measured));
        h3_latents generated = {0};
        struct rusage usage_start;
        struct rusage usage_end;
        getrusage(RUSAGE_SELF, &usage_start);
        uint16_t fixed_state[5120];
        for (size_t index = 0u; index < 5120u; ++index)
            fixed_state[index] = h3_float_to_half(
                (float)((int)(index % 17u) - 8) * 0.001f);
        double started = e2e_now();
        if (h3_metal_open(options->metallib_path, &metal, error,
                          error_capacity) != 0)
            return 1;
        measured.metal_setup_seconds = metal.setup_seconds;
        measured.pipeline_archive_hit = metal.pipeline_archive_hit;
        if (h3_transformer_run(fixed_state, 1u, options, NULL, NULL, &metal,
                               &generated,
                               &measured.transformer_download_seconds,
                               &measured.turbo_compile_seconds,
                               &measured.rope_precompute_seconds,
                               &measured.denoise_seconds,
                               &measured.peak_footprint_bytes, error,
                               error_capacity) != 0)
            return 1;
        measured.turbo_adapter_enabled =
            getenv("MINIMAX_H3_TURBO_ADAPTER") != NULL;
        measured.sampling_steps = measured.turbo_adapter_enabled ? 4u : 30u;
        if (mkdir(options->output_directory, 0755) != 0 && errno != EEXIST) {
            free(generated.video); free(generated.audio);
            e2e_error(error, error_capacity, "cannot create smoke directory");
            return 1;
        }
        char frames[1024];
        if (h3_video_decode(&generated, options, &metal, frames,
                            sizeof(frames), &measured.video_download_seconds,
                            &measured.video_precompute_seconds,
                            &measured.video_decode_seconds,
                            &measured.peak_footprint_bytes, error,
                            error_capacity) != 0) {
            free(generated.video); free(generated.audio);
            return 1;
        }
        free(generated.video);
        if (h3_audio_decode(&generated, options, &metal, measured.audio_path,
                            sizeof(measured.audio_path),
                            &measured.audio_download_seconds,
                            &measured.audio_decode_seconds,
                            &measured.peak_footprint_bytes, error,
                            error_capacity) != 0) {
            free(generated.audio);
            return 1;
        }
        free(generated.audio);
        if (h3_mux(frames, measured.audio_path, options->output_directory,
                   measured.video_path, sizeof(measured.video_path),
                   measured.output_path, sizeof(measured.output_path),
                   &measured.mux_seconds, error, error_capacity) != 0)
            return 1;
        measured.prompt_tokens = 1u;
        measured.sequence_rows = 1u + generated.audio_latent_frames * 2u +
            generated.video_latent_frames * (generated.latent_height / 2u) *
            (generated.latent_width / 2u);
        measured.total_seconds = e2e_now() - started;
        getrusage(RUSAGE_SELF, &usage_end);
        measured.process_user_seconds = e2e_timeval(usage_end.ru_utime) -
                                        e2e_timeval(usage_start.ru_utime);
        measured.process_system_seconds = e2e_timeval(usage_end.ru_stime) -
                                          e2e_timeval(usage_start.ru_stime);
        *result = measured;
        return 0;
    }
}

int minimax_h3_m3_run_video_vae_smoke(
    const minimax_h3_m3_e2e_options *options,
    minimax_h3_m3_e2e_result *result,
    char *error,
    size_t error_capacity) {
    if (options == NULL || result == NULL || options->metallib_path == NULL ||
        options->output_directory == NULL || options->width == 0u ||
        options->height == 0u || options->width % 16u != 0u ||
        options->height % 16u != 0u || options->frames < 22u ||
        (options->frames - 5u) % 17u != 0u) {
        e2e_error(error, error_capacity, "invalid Video VAE smoke options");
        return 2;
    }
    @autoreleasepool {
        h3_metal metal;
        minimax_h3_m3_e2e_result measured;
        memset(&measured, 0, sizeof(measured));
        struct rusage usage_start;
        struct rusage usage_end;
        getrusage(RUSAGE_SELF, &usage_start);
        double total_started = e2e_now();
        if (h3_metal_open(options->metallib_path, &metal, error,
                          error_capacity) != 0)
            return 1;
        measured.metal_setup_seconds = metal.setup_seconds;
        measured.pipeline_archive_hit = metal.pipeline_archive_hit;
        h3_latents latents = {
            .video_latent_frames =
                ((options->frames - 5u) / 17u) * 5u + 2u,
            .latent_height = options->height / 16u,
            .latent_width = options->width / 16u,
        };
        size_t video_values = (size_t)24u * latents.video_latent_frames *
                              latents.latent_height * latents.latent_width;
        latents.video = malloc(video_values * sizeof(float));
        if (latents.video == NULL) {
            e2e_error(error, error_capacity,
                      "cannot allocate deterministic Video VAE input");
            return 1;
        }
        uint64_t state = options->seed != 0u
                             ? options->seed
                             : UINT64_C(0x9e3779b97f4a7c15);
        for (size_t index = 0u; index < video_values; ++index) {
            state ^= state >> 12u;
            state ^= state << 25u;
            state ^= state >> 27u;
            uint32_t sample =
                (uint32_t)((state * UINT64_C(2685821657736338717)) >> 40u);
            latents.video[index] =
                ((float)sample / 16777215.0f - 0.5f) * 0.5f;
        }
        if (mkdir(options->output_directory, 0755) != 0 && errno != EEXIST) {
            free(latents.video);
            e2e_error(error, error_capacity,
                      "cannot create Video VAE smoke directory");
            return 1;
        }
        char frames[1024];
        if (h3_video_decode(&latents, options, &metal, frames,
                            sizeof(frames), &measured.video_download_seconds,
                            &measured.video_precompute_seconds,
                            &measured.video_decode_seconds,
                            &measured.peak_footprint_bytes, error,
                            error_capacity) != 0) {
            free(latents.video);
            return 1;
        }
        free(latents.video);
        measured.sequence_rows =
            (size_t)latents.video_latent_frames * latents.latent_height *
            latents.latent_width;
        snprintf(measured.video_path, sizeof(measured.video_path), "%s",
                 frames);
        snprintf(measured.output_path, sizeof(measured.output_path), "%s",
                 frames);
        measured.total_seconds = e2e_now() - total_started;
        getrusage(RUSAGE_SELF, &usage_end);
        measured.process_user_seconds = e2e_timeval(usage_end.ru_utime) -
                                        e2e_timeval(usage_start.ru_utime);
        measured.process_system_seconds = e2e_timeval(usage_end.ru_stime) -
                                          e2e_timeval(usage_start.ru_stime);
        *result = measured;
        return 0;
    }
}

int minimax_h3_m3_run_image_vae_smoke(
    const minimax_h3_m3_e2e_options *options,
    minimax_h3_m3_e2e_result *result,
    char *error,
    size_t error_capacity) {
    if (options == NULL || result == NULL ||
        options->first_image_path == NULL || options->metallib_path == NULL ||
        options->output_directory == NULL || options->width == 0u ||
        options->height == 0u || options->width % 16u != 0u ||
        options->height % 16u != 0u) {
        e2e_error(error, error_capacity, "invalid image VAE smoke options");
        return 2;
    }
    @autoreleasepool {
        h3_metal metal;
        minimax_h3_m3_e2e_result measured;
        memset(&measured, 0, sizeof(measured));
        float *condition = NULL;
        if (h3_metal_open(options->metallib_path, &metal, error,
                          error_capacity) != 0)
            return 1;
        measured.metal_setup_seconds = metal.setup_seconds;
        measured.pipeline_archive_hit = metal.pipeline_archive_hit;
        if (h3_video_encode_condition(
                options->first_image_path, options, &metal, &condition,
                &measured.image_encode_seconds,
                &measured.peak_footprint_bytes, error,
                error_capacity) != 0)
            return 1;
        if (mkdir(options->output_directory, 0755) != 0 && errno != EEXIST) {
            free(condition);
            e2e_error(error, error_capacity,
                      "cannot create image VAE smoke directory");
            return 1;
        }
        if (snprintf(measured.output_path, sizeof(measured.output_path),
                     "%s/condition-latent.f32", options->output_directory) >=
            (int)sizeof(measured.output_path)) {
            free(condition);
            e2e_error(error, error_capacity,
                      "image VAE smoke output path is too long");
            return 1;
        }
        FILE *file = fopen(measured.output_path, "wb");
        size_t values = (size_t)24u * (options->height / 16u) *
                        (options->width / 16u);
        size_t written = 0u;
        int close_status = 0;
        if (file != NULL) {
            written = fwrite(condition, sizeof(float), values, file);
            close_status = fclose(file);
        }
        if (file == NULL || written != values || close_status != 0) {
            free(condition);
            e2e_error(error, error_capacity,
                      "cannot write image VAE smoke latent");
            return 1;
        }
        free(condition);
        measured.condition_images = 1u;
        measured.sequence_rows = (size_t)(options->height / 32u) *
                                 (options->width / 32u);
        *result = measured;
        return 0;
    }
}

int minimax_h3_m3_run_e2e(const minimax_h3_m3_e2e_options *options,
                          minimax_h3_m3_e2e_result *result,
                          char *error,
                          size_t error_capacity) {
    if (options == NULL || result == NULL || options->prompt == NULL ||
        options->tokenizer_image_path == NULL || options->metallib_path == NULL ||
        options->output_directory == NULL ||
        options->width == 0u || options->height == 0u || options->frames == 0u) {
        e2e_error(error, error_capacity, "invalid MiniMax-H3 E2E options");
        return 2;
    }
    @autoreleasepool {
        double total_started = e2e_now();
        struct rusage usage_start;
        struct rusage usage_end;
        getrusage(RUSAGE_SELF, &usage_start);
        minimax_h3_m3_e2e_result measured;
        memset(&measured, 0, sizeof(measured));
        h3_metal metal;
        if (h3_metal_open(options->metallib_path, &metal, error,
                          error_capacity) != 0)
            return 1;
        measured.metal_setup_seconds = metal.setup_seconds;
        measured.pipeline_archive_hit = metal.pipeline_archive_hit;
        double tokenizer_started = e2e_now();
        qwen38_tokenizer *tokenizer = qwen38_tokenizer_open(
            options->tokenizer_image_path, error, error_capacity);
        if (tokenizer == NULL) return 1;
        size_t capacity = strlen(options->prompt) * 2u + 32u;
        uint32_t *token_ids = malloc(capacity * sizeof(*token_ids));
        size_t token_count = 0u;
        if (token_ids == NULL || qwen38_tokenizer_encode(
                tokenizer, options->prompt, token_ids, capacity, &token_count,
                error, error_capacity) != 0 || token_count == 0u) {
            free(token_ids);
            qwen38_tokenizer_close(tokenizer);
            return 1;
        }
        qwen38_tokenizer_close(tokenizer);
        measured.tokenizer_seconds = e2e_now() - tokenizer_started;
        measured.prompt_tokens = token_count;
        uint16_t *text_states = NULL;
        if (h3_text_encode(token_ids, token_count, &metal, &text_states,
                           &measured.text_download_seconds,
                           &measured.text_encode_seconds,
                           &measured.peak_footprint_bytes, error,
                           error_capacity) != 0) {
            free(token_ids);
            return 1;
        }
        free(token_ids);
        float *first_condition = NULL;
        float *last_condition = NULL;
        if (options->first_image_path != NULL &&
            h3_video_encode_condition(
                options->first_image_path, options, &metal, &first_condition,
                &measured.image_encode_seconds,
                &measured.peak_footprint_bytes, error, error_capacity) != 0) {
            free(text_states);
            return 1;
        }
        if (options->last_image_path != NULL) {
            double last_seconds = 0.0;
            if (h3_video_encode_condition(
                    options->last_image_path, options, &metal,
                    &last_condition, &last_seconds,
                    &measured.peak_footprint_bytes, error,
                    error_capacity) != 0) {
                free(first_condition);
                free(text_states);
                return 1;
            }
            measured.image_encode_seconds += last_seconds;
        }
        measured.condition_images =
            (first_condition != NULL ? 1u : 0u) +
            (last_condition != NULL ? 1u : 0u);
        if (measured.condition_images != 0u) {
            fprintf(stderr,
                    "stage=fl2va conditions=%u conditioner=vae-anchor "
                    "vision_tokens=text-only\n",
                    measured.condition_images);
            fflush(stderr);
        }
        h3_latents generated = {0};
        measured.peak_footprint_bytes = e2e_footprint();
        if (h3_transformer_run(text_states, token_count, options,
                               first_condition, last_condition, &metal,
                               &generated,
                               &measured.transformer_download_seconds,
                               &measured.turbo_compile_seconds,
                               &measured.rope_precompute_seconds,
                               &measured.denoise_seconds,
                               &measured.peak_footprint_bytes, error,
                               error_capacity) != 0) {
            free(first_condition);
            free(last_condition);
            free(text_states);
            return 1;
        }
        free(first_condition);
        free(last_condition);
        measured.turbo_adapter_enabled =
            getenv("MINIMAX_H3_TURBO_ADAPTER") != NULL;
        measured.sampling_steps = measured.turbo_adapter_enabled ? 4u : 30u;
        free(text_states);
        const char *denoise_only_text =
            getenv("MINIMAX_H3_DENOISE_ONLY");
        if (denoise_only_text != NULL &&
            strcmp(denoise_only_text, "1") == 0) {
            measured.sequence_rows = token_count +
                (size_t)generated.audio_latent_frames * 2u +
                ((size_t)generated.video_latent_frames +
                 measured.condition_images) *
                    (generated.latent_height / 2u) *
                    (generated.latent_width / 2u);
            free(generated.video);
            free(generated.audio);
            measured.total_seconds = e2e_now() - total_started;
            getrusage(RUSAGE_SELF, &usage_end);
            measured.process_user_seconds =
                e2e_timeval(usage_end.ru_utime) -
                e2e_timeval(usage_start.ru_utime);
            measured.process_system_seconds =
                e2e_timeval(usage_end.ru_stime) -
                e2e_timeval(usage_start.ru_stime);
            measured.peak_footprint_bytes =
                MAX(measured.peak_footprint_bytes, e2e_footprint());
            *result = measured;
            if (error != NULL && error_capacity != 0u) error[0] = '\0';
            return 0;
        }
        if (mkdir(options->output_directory, 0755) != 0 && errno != EEXIST) {
            free(generated.video);
            free(generated.audio);
            e2e_error(error, error_capacity, "cannot create output directory: %s",
                      strerror(errno));
            return 1;
        }
        char frame_directory[1024];
        if (h3_video_decode(&generated, options, &metal, frame_directory,
                            sizeof(frame_directory),
                            &measured.video_download_seconds,
                            &measured.video_precompute_seconds,
                            &measured.video_decode_seconds,
                            &measured.peak_footprint_bytes, error,
                            error_capacity) != 0) {
            free(generated.video);
            free(generated.audio);
            return 1;
        }
        free(generated.video);
        generated.video = NULL;
        if (h3_audio_decode(&generated, options, &metal,
                            measured.audio_path,
                            sizeof(measured.audio_path),
                            &measured.audio_download_seconds,
                            &measured.audio_decode_seconds,
                            &measured.peak_footprint_bytes, error,
                            error_capacity) != 0) {
            free(generated.audio);
            return 1;
        }
        free(generated.audio);
        generated.audio = NULL;
        if (h3_mux(frame_directory, measured.audio_path,
                   options->output_directory, measured.video_path,
                   sizeof(measured.video_path), measured.output_path,
                   sizeof(measured.output_path), &measured.mux_seconds, error,
                   error_capacity) != 0)
            return 1;
        measured.sequence_rows = token_count +
            (size_t)generated.audio_latent_frames * 2u +
            ((size_t)generated.video_latent_frames +
             measured.condition_images) *
                (generated.latent_height / 2u) *
                (generated.latent_width / 2u);
        measured.total_seconds = e2e_now() - total_started;
        getrusage(RUSAGE_SELF, &usage_end);
        measured.process_user_seconds = e2e_timeval(usage_end.ru_utime) -
                                        e2e_timeval(usage_start.ru_utime);
        measured.process_system_seconds = e2e_timeval(usage_end.ru_stime) -
                                          e2e_timeval(usage_start.ru_stime);
        measured.peak_footprint_bytes =
            MAX(measured.peak_footprint_bytes, e2e_footprint());
        *result = measured;
        if (error != NULL && error_capacity != 0u) error[0] = '\0';
        return 0;
    }
}
