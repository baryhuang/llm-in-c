#include "minimindo_mimi.h"
#include "minimindo_parallel.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void little_u16(FILE *stream, uint16_t value)
{
    unsigned char b[2] = {(unsigned char)value, (unsigned char)(value >> 8)};
    fwrite(b, 1, sizeof(b), stream);
}

static void little_u32(FILE *stream, uint32_t value)
{
    unsigned char b[4] = {(unsigned char)value, (unsigned char)(value >> 8),
                          (unsigned char)(value >> 16), (unsigned char)(value >> 24)};
    fwrite(b, 1, sizeof(b), stream);
}

static int write_wav(const char *path, const float *audio, size_t samples,
                     uint32_t sample_rate)
{
    if (samples > UINT32_MAX / 2U) return -1;
    FILE *stream = fopen(path, "wb");
    if (stream == NULL) return -1;
    const uint32_t bytes = (uint32_t)samples * 2U;
    fwrite("RIFF", 1, 4, stream); little_u32(stream, 36U + bytes);
    fwrite("WAVEfmt ", 1, 8, stream); little_u32(stream, 16);
    little_u16(stream, 1); little_u16(stream, 1); little_u32(stream, sample_rate);
    little_u32(stream, sample_rate * 2); little_u16(stream, 2); little_u16(stream, 16);
    fwrite("data", 1, 4, stream); little_u32(stream, bytes);
    for (size_t i = 0; i < samples; ++i) {
        float value = audio[i];
        if (value > 1) value = 1; else if (value < -1) value = -1;
        little_u16(stream, (uint16_t)(int16_t)lrintf(value * 32767.0f));
    }
    const int failed = ferror(stream); fclose(stream); return failed ? -1 : 0;
}

int main(int argc, char **argv)
{
    if (argc != 4 && argc != 6) {
        fprintf(stderr,
                "usage: %s MIMI.mmo CODES.txt OUTPUT.wav [--stream 1]\n",
                argv[0]);
        return 2;
    }
    size_t stream_frames = 0;
    if (argc == 6) {
        if (strcmp(argv[4], "--stream") != 0) return 2;
        stream_frames = strtoul(argv[5], NULL, 10);
        if (stream_frames != 1U) {
            fprintf(stderr, "the stateful streaming decoder requires --stream 1\n");
            return 2;
        }
    }
    FILE *input = fopen(argv[2], "r");
    if (input == NULL) { perror(argv[2]); return 3; }
    size_t frames = 0, capacity = 64;
    uint32_t *frame_major = malloc(capacity * MINIMINDO_MIMI_CODEBOOKS * sizeof(uint32_t));
    while (1) {
        uint32_t row[MINIMINDO_MIMI_CODEBOOKS];
        int count = 0;
        for (; count < MINIMINDO_MIMI_CODEBOOKS; ++count)
            if (fscanf(input, "%u", &row[count]) != 1) break;
        if (count == 0) break;
        if (count != MINIMINDO_MIMI_CODEBOOKS) { fprintf(stderr, "incomplete code frame\n"); return 3; }
        if (frames == capacity) {
            capacity *= 2;
            uint32_t *grown = realloc(frame_major, capacity * MINIMINDO_MIMI_CODEBOOKS * sizeof(uint32_t));
            if (grown == NULL) return 3;
            frame_major = grown;
        }
        for (int i = 0; i < MINIMINDO_MIMI_CODEBOOKS; ++i)
            frame_major[frames * MINIMINDO_MIMI_CODEBOOKS + i] = row[i];
        ++frames;
    }
    fclose(input);
    if (frames == 0) { fprintf(stderr, "no Mimi frames\n"); return 3; }
    uint32_t *codes = malloc(frames * MINIMINDO_MIMI_CODEBOOKS * sizeof(uint32_t));
    for (size_t t = 0; t < frames; ++t)
        for (size_t c = 0; c < MINIMINDO_MIMI_CODEBOOKS; ++c)
            codes[c * frames + t] = frame_major[t * MINIMINDO_MIMI_CODEBOOKS + c];
    free(frame_major);
    char error[256] = {0};
    unsigned threads = 4U;
    const char *thread_text = getenv("MINIMINDO_THREADS");
    if (thread_text != NULL) {
        const unsigned parsed = (unsigned)strtoul(thread_text, NULL, 10);
        if (parsed < 1U || parsed > 4U) {
            fprintf(stderr, "MINIMINDO_THREADS must be in [1,4]\n");
            return 2;
        }
        threads = parsed;
    }
    minimindo_mimi *model = minimindo_mimi_open(argv[1], (uint32_t)frames, error, sizeof(error));
    if (model == NULL) { fprintf(stderr, "%s\n", error); return 4; }
    const size_t capacity_samples = minimindo_mimi_samples_for_frames(model, frames);
    float *audio = malloc(capacity_samples * sizeof(float)); size_t samples = 0;
    int decode_result = 0;
    (void)minimindo_parallel_pin_current(0U);
    (void)minimindo_parallel_session_begin(threads);
    if (audio == NULL) {
        decode_result = -1;
    } else if (stream_frames == 0) {
        decode_result = minimindo_mimi_decode(model, codes, frames, audio,
                                               capacity_samples, &samples,
                                               error, sizeof(error));
    } else {
        minimindo_mimi_stream *stream = minimindo_mimi_stream_open(
            model, error, sizeof(error));
        uint32_t *chunk_codes = malloc(
            stream_frames * MINIMINDO_MIMI_CODEBOOKS * sizeof(uint32_t));
        if (stream == NULL || chunk_codes == NULL) {
            decode_result = -1;
        } else {
            for (size_t offset = 0; offset < frames && decode_result == 0;) {
                size_t count = frames - offset;
                if (count > stream_frames) count = stream_frames;
                for (size_t c = 0; c < MINIMINDO_MIMI_CODEBOOKS; ++c)
                    for (size_t t = 0; t < count; ++t)
                        chunk_codes[c * count + t] = codes[c * frames + offset + t];
                size_t produced = 0;
                decode_result = minimindo_mimi_stream_decode(
                    stream, chunk_codes, count, audio + samples,
                    capacity_samples - samples, &produced, error,
                    sizeof(error));
                samples += produced;
                offset += count;
            }
        }
        free(chunk_codes);
        minimindo_mimi_stream_close(stream);
    }
    minimindo_parallel_session_end();
    if (decode_result != 0) {
        fprintf(stderr, "%s\n", error); return 5;
    }
    double sum = 0, peak = 0;
    for (size_t i = 0; i < samples; ++i) {
        sum += (double)audio[i] * audio[i];
        if (fabs(audio[i]) > peak) peak = fabs(audio[i]);
    }
    if (write_wav(argv[3], audio, samples, minimindo_mimi_sample_rate(model)) != 0) {
        perror(argv[3]); return 6;
    }
    printf("{\"frames\":%zu,\"samples\":%zu,\"sample_rate\":%u,"
           "\"stream_frames\":%zu,\"rms\":%.9g,\"peak\":%.9g}\n",
           frames, samples, minimindo_mimi_sample_rate(model), stream_frames,
           sqrt(sum / samples), peak);
    free(audio); free(codes); minimindo_mimi_close(model); return 0;
}
