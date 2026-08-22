#ifndef LLM_IN_C_MINIMINDO_MIMI_H
#define LLM_IN_C_MINIMINDO_MIMI_H

#include <stddef.h>
#include <stdint.h>

enum { MINIMINDO_MIMI_CODEBOOKS = 8 };

typedef struct minimindo_mimi minimindo_mimi;
typedef struct minimindo_mimi_stream minimindo_mimi_stream;

minimindo_mimi *minimindo_mimi_open(const char *image_path,
                                    uint32_t max_frames,
                                    char *error, size_t error_capacity);
void minimindo_mimi_close(minimindo_mimi *model);

uint32_t minimindo_mimi_sample_rate(const minimindo_mimi *model);
size_t minimindo_mimi_samples_for_frames(const minimindo_mimi *model,
                                         size_t frames);

/* codes is codebook-major [8][frames], exactly as MimiModel.decode expects. */
int minimindo_mimi_decode(minimindo_mimi *model,
                          const uint32_t *codes, size_t frames,
                          float *audio, size_t audio_capacity,
                          size_t *audio_samples,
                          char *error, size_t error_capacity);

/*
 * Stateful causal decoder. Each call pushes exactly one new codec frame, so
 * codes is codebook-major [8][1]. The stream retains the Mimi transformer KV
 * cache, the semantic upsampler tail, and every causal convolution/deconvolution
 * boundary. One-frame pushes are intentional: this is the production SPSC
 * boundary between the Talker and Mimi, not a selectable batch mode.
 */
minimindo_mimi_stream *minimindo_mimi_stream_open(
    minimindo_mimi *model, char *error, size_t error_capacity);
void minimindo_mimi_stream_reset(minimindo_mimi_stream *stream);
void minimindo_mimi_stream_close(minimindo_mimi_stream *stream);
int minimindo_mimi_stream_decode(minimindo_mimi_stream *stream,
                                 const uint32_t *codes, size_t frames,
                                 float *audio, size_t audio_capacity,
                                 size_t *audio_samples,
                                 char *error, size_t error_capacity);

#endif
