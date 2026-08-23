/* Resident interactive chat for Qwen3.8-27B on Apple M3 Pro.
 *
 * Opens the tokenizer and model once - the one-time weight wiring happens
 * at startup - then serves prompts in a loop from standard input, so every
 * prompt after the first runs at the ready-state time to first token
 * instead of paying a full cold start. Layer state is reset between
 * prompts; each prompt is an independent single-turn request, the same
 * contract as the one-shot generator. */

#include "qwen38_m3_decode.h"
#include "qwen38_sampler.h"
#include "qwen38_tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <unicode/uchar.h>
#include <unicode/utf8.h>

enum { QWEN38_END_OF_TEXT = 248044, QWEN38_IM_END = 248046 };

static double seconds_now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static uint32_t parse_u32(const char *text, uint32_t fallback) {
    if (text == NULL) return fallback;
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    return end != text && *end == '\0' ? (uint32_t)value : fallback;
}

static char *trim_prompt(const char *prompt) {
    const uint8_t *bytes = (const uint8_t *)prompt;
    int32_t length = (int32_t)strlen(prompt);
    int32_t start = 0;
    while (start < length) {
        int32_t next = start;
        UChar32 codepoint;
        U8_NEXT(bytes, next, length, codepoint);
        if (codepoint < 0 || !u_isUWhiteSpace(codepoint)) break;
        start = next;
    }
    int32_t end = length;
    while (end > start) {
        int32_t previous = end;
        UChar32 codepoint;
        U8_PREV(bytes, 0, previous, codepoint);
        if (codepoint < 0 || !u_isUWhiteSpace(codepoint)) break;
        end = previous;
    }
    size_t trimmed_length = (size_t)(end - start);
    char *trimmed = malloc(trimmed_length + 1);
    if (trimmed == NULL) return NULL;
    memcpy(trimmed, prompt + start, trimmed_length);
    trimmed[trimmed_length] = '\0';
    return trimmed;
}

/* Prefill the half-open token span [from, to) at its own positions. */
static int prefill_span(qwen38_m3_model *model, const uint32_t *ids,
                        uint32_t from, uint32_t to, char *error,
                        size_t error_capacity) {
    if (to <= from) return 0;
    qwen38_m3_prefill_result prefill;
    return qwen38_m3_model_prefill(model, ids + from, to - from, from,
                                   &prefill, error, error_capacity);
}

static char *render_chat(const char *prompt) {
    static const char prefix[] = "<|im_start|>user\n";
    static const char suffix[] =
        "<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n";
    size_t length = sizeof(prefix) - 1 + strlen(prompt) + sizeof(suffix);
    char *chat = malloc(length);
    if (chat == NULL) return NULL;
    snprintf(chat, length, "%s%s%s", prefix, prompt, suffix);
    return chat;
}

static void put_json_string(const char *bytes, size_t length) {
    putchar('"');
    for (size_t index = 0; index < length; ++index) {
        unsigned char byte = (unsigned char)bytes[index];
        switch (byte) {
        case '"': fputs("\\\"", stdout); break;
        case '\\': fputs("\\\\", stdout); break;
        case '\n': fputs("\\n", stdout); break;
        case '\r': fputs("\\r", stdout); break;
        case '\t': fputs("\\t", stdout); break;
        default:
            if (byte < 0x20) printf("\\u%04x", byte);
            else putchar(byte);
        }
    }
    putchar('"');
}

static void append_utf8(char **cursor, uint32_t codepoint) {
    char *out = *cursor;
    if (codepoint < 0x80) {
        *out++ = (char)codepoint;
    } else if (codepoint < 0x800) {
        *out++ = (char)(0xc0 | (codepoint >> 6));
        *out++ = (char)(0x80 | (codepoint & 0x3f));
    } else if (codepoint < 0x10000) {
        *out++ = (char)(0xe0 | (codepoint >> 12));
        *out++ = (char)(0x80 | ((codepoint >> 6) & 0x3f));
        *out++ = (char)(0x80 | (codepoint & 0x3f));
    } else {
        *out++ = (char)(0xf0 | (codepoint >> 18));
        *out++ = (char)(0x80 | ((codepoint >> 12) & 0x3f));
        *out++ = (char)(0x80 | ((codepoint >> 6) & 0x3f));
        *out++ = (char)(0x80 | (codepoint & 0x3f));
    }
    *cursor = out;
}

static int hex_value(char character) {
    if (character >= '0' && character <= '9') return character - '0';
    if (character >= 'a' && character <= 'f') return character - 'a' + 10;
    if (character >= 'A' && character <= 'F') return character - 'A' + 10;
    return -1;
}

/* Decode one JSON string literal starting at *cursor (which must point
 * at the opening quote) to UTF-8; advances *cursor past the closing
 * quote. */
static char *decode_json_string_span(const char **cursor) {
    const char *line = *cursor;
    if (*line != '"') return NULL;
    ++line;
    char *output = malloc(strlen(line) * 4 + 1);
    if (output == NULL) return NULL;
    char *out_cursor = output;
    while (*line != '\0' && *line != '"') {
        if (*line != '\\') {
            *out_cursor++ = *line++;
            continue;
        }
        ++line;
        switch (*line) {
        case '"': *out_cursor++ = '"'; ++line; break;
        case '\\': *out_cursor++ = '\\'; ++line; break;
        case '/': *out_cursor++ = '/'; ++line; break;
        case 'b': *out_cursor++ = '\b'; ++line; break;
        case 'f': *out_cursor++ = '\f'; ++line; break;
        case 'n': *out_cursor++ = '\n'; ++line; break;
        case 'r': *out_cursor++ = '\r'; ++line; break;
        case 't': *out_cursor++ = '\t'; ++line; break;
        case 'u': {
            uint32_t codepoint = 0;
            ++line;
            for (int digit = 0; digit < 4; ++digit) {
                int value = hex_value(*line);
                if (value < 0) { free(output); return NULL; }
                codepoint = codepoint << 4 | (uint32_t)value;
                ++line;
            }
            if (codepoint >= 0xd800 && codepoint <= 0xdbff &&
                line[0] == '\\' && line[1] == 'u') {
                uint32_t low = 0;
                line += 2;
                for (int digit = 0; digit < 4; ++digit) {
                    int value = hex_value(*line);
                    if (value < 0) { free(output); return NULL; }
                    low = low << 4 | (uint32_t)value;
                    ++line;
                }
                codepoint = 0x10000 +
                    ((codepoint - 0xd800) << 10) + (low - 0xdc00);
            }
            append_utf8(&out_cursor, codepoint);
            break;
        }
        default:
            free(output);
            return NULL;
        }
    }
    if (*line != '"') { free(output); return NULL; }
    *out_cursor = '\0';
    *cursor = line + 1;
    return output;
}

static char *decode_json_string(const char *line) {
    while (*line == ' ' || *line == '\t') ++line;
    return decode_json_string_span(&line);
}

/* One machine-mode request. A plain JSON string line is a greedy
 * request with the session defaults; a JSON object line carries
 * per-request sampling and budget:
 *   {"prompt": "...", "temperature": 0.7, "top_k": 20, "top_p": 0.8,
 *    "min_p": 0.0, "presence_penalty": 1.5, "max_new": 512}
 * "prefix" optionally names a leading span of "prompt" that the caller
 * expects to repeat across conversations - an agent front end's system
 * turn, say - so the engine can keep a checkpoint at that boundary and
 * skip re-prefilling it when a new conversation starts.
 * Unknown keys are ignored. */
typedef struct {
    char *chat;
    char *prefix;
    float temperature;
    uint32_t top_k;
    float top_p;
    float min_p;
    float presence_penalty;
    uint32_t max_new;
} chat_request;

static int parse_request_object(const char *line, chat_request *request) {
    while (*line == ' ' || *line == '\t') ++line;
    if (*line != '{') return -1;
    ++line;
    for (;;) {
        while (*line == ' ' || *line == '\t' || *line == ',') ++line;
        if (*line == '}') return request->chat != NULL ? 0 : -1;
        if (*line != '"') return -1;
        char *key = decode_json_string_span(&line);
        if (key == NULL) return -1;
        while (*line == ' ' || *line == '\t') ++line;
        if (*line != ':') { free(key); return -1; }
        ++line;
        while (*line == ' ' || *line == '\t') ++line;
        if (*line == '"') {
            char *value = decode_json_string_span(&line);
            if (value == NULL) { free(key); return -1; }
            if (strcmp(key, "prompt") == 0) {
                free(request->chat);
                request->chat = value;
            } else if (strcmp(key, "prefix") == 0) {
                free(request->prefix);
                request->prefix = value;
            } else {
                free(value);
            }
        } else {
            char *end = NULL;
            double value = strtod(line, &end);
            if (end == line) { free(key); return -1; }
            line = end;
            if (strcmp(key, "temperature") == 0)
                request->temperature = (float)value;
            else if (strcmp(key, "top_k") == 0)
                request->top_k = value > 0 ? (uint32_t)value : 0;
            else if (strcmp(key, "top_p") == 0)
                request->top_p = (float)value;
            else if (strcmp(key, "min_p") == 0)
                request->min_p = (float)value;
            else if (strcmp(key, "presence_penalty") == 0)
                request->presence_penalty = (float)value;
            else if (strcmp(key, "max_new") == 0)
                request->max_new = value > 0 ? (uint32_t)value : 0;
        }
        free(key);
    }
}

/* Decode the visible token prefix and print only the new complete UTF-8
 * suffix, so text appears as soon as each token completes. In machine
 * mode each delta goes out as a 'D "..."' protocol line. */
/* Incremental stream decoder: each call decodes only the tokens added
 * since the last one and appends their bytes, so a T-token generation
 * costs O(T) total instead of re-decoding the visible prefix each step.
 * Only the last, possibly incomplete UTF-8 character is held back. */
typedef struct {
    char *bytes;
    size_t byte_count;
    size_t byte_capacity;
    size_t token_count;
    size_t emitted;
} stream_text;

static void stream_text_reset(stream_text *stream) {
    stream->byte_count = 0;
    stream->token_count = 0;
    stream->emitted = 0;
}

static size_t stream_complete_bytes(const stream_text *stream) {
    const unsigned char *bytes = (const unsigned char *)stream->bytes;
    size_t count = stream->byte_count;
    size_t back = 0;
    while (back < count && back < 3 &&
           (bytes[count - 1 - back] & 0xc0) == 0x80)
        ++back;
    if (back >= count) return count;
    unsigned char lead = bytes[count - 1 - back];
    size_t need = (lead & 0x80) == 0 ? 1 :
                  (lead & 0xe0) == 0xc0 ? 2 :
                  (lead & 0xf0) == 0xe0 ? 3 :
                  (lead & 0xf8) == 0xf0 ? 4 : 1;
    if (back + 1 >= need) return count;
    return count - back - 1;
}

static int emit_new_text(const qwen38_tokenizer *tokenizer,
                         const uint32_t *tokens, size_t token_count,
                         stream_text *stream, int machine, char *error,
                         size_t error_capacity) {
    if (token_count > stream->token_count) {
        const uint32_t *fresh = tokens + stream->token_count;
        size_t fresh_count = token_count - stream->token_count;
        size_t fresh_length = 0;
        (void)qwen38_tokenizer_decode(tokenizer, fresh, fresh_count, NULL,
                                      0, &fresh_length, error,
                                      error_capacity);
        if (stream->byte_count + fresh_length + 1 >
            stream->byte_capacity) {
            size_t capacity =
                (stream->byte_count + fresh_length + 1) * 2 + 256;
            char *grown = realloc(stream->bytes, capacity);
            if (grown == NULL) return -1;
            stream->bytes = grown;
            stream->byte_capacity = capacity;
        }
        size_t written = 0;
        if (qwen38_tokenizer_decode(
                tokenizer, fresh, fresh_count,
                stream->bytes + stream->byte_count,
                stream->byte_capacity - stream->byte_count, &written,
                error, error_capacity) != 0)
            return -1;
        stream->byte_count += written;
        stream->token_count = token_count;
    }
    size_t complete = stream_complete_bytes(stream);
    if (complete > stream->emitted) {
        if (machine) {
            fputs("D ", stdout);
            put_json_string(stream->bytes + stream->emitted,
                            complete - stream->emitted);
            putchar('\n');
        } else {
            fwrite(stream->bytes + stream->emitted, 1,
                   complete - stream->emitted, stdout);
        }
        fflush(stdout);
        stream->emitted = complete;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 9) {
        fprintf(stderr,
            "usage: %s MODEL_DIR METALLIB TOKENIZER.q38tok "
            "[CONTEXT] [MAX_NEW] [TEMPERATURE] [TOP_K] [SEED]\n",
            argv[0]);
        return 2;
    }
    uint32_t capacity = parse_u32(argc > 4 ? argv[4] : NULL, 4096);
    uint32_t maximum_new = parse_u32(argc > 5 ? argv[5] : NULL, 3072);
    float temperature = argc > 6 ? strtof(argv[6], NULL) : 0.0f;
    uint32_t top_k = parse_u32(argc > 7 ? argv[7] : NULL, 1);
    uint64_t seed = argc > 8 ? strtoull(argv[8], NULL, 10) : 42;

    const char *machine_env = getenv("QWEN38_MACHINE");
    int machine = machine_env != NULL && strcmp(machine_env, "0") != 0;

    char error[512];
    qwen38_tokenizer *tokenizer =
        qwen38_tokenizer_open(argv[3], error, sizeof(error));
    if (tokenizer == NULL) {
        fprintf(stderr, "tokenizer open failed: %s\n", error);
        return 3;
    }
    fprintf(stderr, "Loading Qwen3.8-27B "
                    "(one-time weight wiring at startup)...\n");
    double start = seconds_now();
    qwen38_m3_model *model = qwen38_m3_model_open(
        argv[1], argv[2], capacity, error, sizeof(error));
    if (model == NULL) {
        fprintf(stderr, "model open failed: %s\n", error);
        qwen38_tokenizer_close(tokenizer);
        return 5;
    }
    fprintf(stderr, "Model resident in %.1f s. Enter /quit to exit.\n",
            seconds_now() - start);
    int mtp = 0;
    int mtp_depth = 0; /* 0 = adaptive (max 7) */
    const char *mtp_depth_env = getenv("QWEN38_MTP_DEPTH");
    if (mtp_depth_env != NULL) {
        mtp_depth = atoi(mtp_depth_env);
        if (mtp_depth < 0) mtp_depth = 0;
        if (mtp_depth > 7) mtp_depth = 7;
    }

    const char *mtp_env = getenv("QWEN38_MTP");
    if (mtp_env == NULL || strcmp(mtp_env, "0") != 0) {
        char layer_path[1024];
        char extras_path[1024];
        snprintf(layer_path, sizeof(layer_path), "%s/mtp-layer.q38att",
                 argv[1]);
        snprintf(extras_path, sizeof(extras_path), "%s/mtp.q38mtp",
                 argv[1]);
        if (qwen38_m3_model_mtp_open(model, layer_path, extras_path,
                                     error, sizeof(error)) == 0) {
            mtp = 1;
            fprintf(stderr, "MTP speculative decoding active "
                            "(greedy, output-lossless).\n");
        } else if (mtp_env != NULL) {
            fprintf(stderr, "MTP requested but unavailable: %s\n", error);
        }
    }
    if (machine) {
        printf("R {\"ready\": true, \"context\": %u, "
               "\"max_new\": %u}\n", capacity, maximum_new);
        fflush(stdout);
    }

    uint32_t *prompt_ids = malloc((size_t)capacity * sizeof(*prompt_ids));
    uint32_t *generated = malloc((size_t)maximum_new * sizeof(*generated));
    /* Conversation continuation: history holds the exact token sequence
     * the resident layer state corresponds to. A request whose tokens
     * extend it prefills only the new suffix, so multi-turn
     * time-to-first-token depends on the new turn, not the whole
     * conversation. QWEN38_CONTINUE=0 disables. */
    uint32_t *history = malloc((size_t)capacity * sizeof(*history));
    uint32_t history_count = 0;
    /* Prompt-boundary checkpoint tokens: generation can tokenize
     * non-canonically (its tokens need not survive a decode/encode
     * round trip), so when the live token history does not match a
     * follow-up's rendering, the engine rewinds the GDN state to the
     * last prompt boundary — whose tokens are canonical by
     * construction — and prefills the reply plus the new turn. */
    uint32_t *snapshot_tokens = malloc((size_t)capacity *
                                       sizeof(*snapshot_tokens));
    uint32_t snapshot_count = 0;
    /* System-prefix checkpoint: the span a caller declares as stable
     * across conversations. The turn checkpoint above only ever holds
     * the conversation in flight, so a new session shares nothing with
     * it past the system turn and pays a full prefill; this slot is
     * what makes that first turn cheap. Valid only while no prefill has
     * started below system_count, which would overwrite the attention
     * KV the restore depends on. */
    uint32_t *system_tokens = malloc((size_t)capacity *
                                     sizeof(*system_tokens));
    uint32_t *prefix_ids = malloc((size_t)capacity *
                                  sizeof(*prefix_ids));
    uint32_t system_count = 0;
    int system_valid = 0;
    const char *continue_env = getenv("QWEN38_CONTINUE");
    int continuation = continue_env == NULL ||
                       strcmp(continue_env, "0") != 0;
    stream_text stream = {NULL, 0, 0, 0, 0};
    char *line = NULL;
    size_t line_capacity = 0;
    if (prompt_ids == NULL || generated == NULL || history == NULL ||
        snapshot_tokens == NULL || system_tokens == NULL ||
        prefix_ids == NULL) {
        fprintf(stderr, "cannot allocate buffers\n");
        return 6;
    }

    for (;;) {
        if (!machine) {
            printf("\nYou> ");
            fflush(stdout);
        }
        ssize_t read_length = getline(&line, &line_capacity, stdin);
        if (read_length < 0) {
            if (!machine) printf("\n");
            break;
        }
        char *chat = NULL;
        chat_request request = {
            .chat = NULL, .prefix = NULL,
            .temperature = temperature, .top_k = top_k,
            .top_p = 0.0f, .min_p = 0.0f, .presence_penalty = 0.0f,
            .max_new = 0
        };
        if (machine) {
            /* One request per line: a JSON string holding the complete
             * rendered chat template, or a JSON object that also
             * carries per-request sampling and budget. */
            const char *scan = line;
            while (*scan == ' ' || *scan == '\t') ++scan;
            if (*scan == '{') {
                if (parse_request_object(line, &request) != 0) {
                    free(request.chat);
                    free(request.prefix);
                    printf("X \"invalid request object\"\n");
                    fflush(stdout);
                    continue;
                }
                chat = request.chat;
            } else {
                chat = decode_json_string(line);
            }
            if (chat == NULL) {
                if (line[0] != '\n') {
                    printf("X \"request line is not a JSON string\"\n");
                    fflush(stdout);
                }
                continue;
            }
        } else {
            char *trimmed = trim_prompt(line);
            if (trimmed == NULL) continue;
            if (trimmed[0] == '\0') {
                free(trimmed);
                continue;
            }
            if (strcmp(trimmed, "/quit") == 0 ||
                strcmp(trimmed, "/exit") == 0) {
                free(trimmed);
                break;
            }
            chat = render_chat(trimmed);
            free(trimmed);
        }
        size_t prompt_count = 0;
        if (chat == NULL || qwen38_tokenizer_encode(
                tokenizer, chat, prompt_ids, capacity, &prompt_count,
                error, sizeof(error)) != 0) {
            const char *reason = chat == NULL ? "allocation" : error;
            if (machine) {
                fputs("X ", stdout);
                put_json_string(reason, strlen(reason));
                putchar('\n');
                fflush(stdout);
            } else {
                fprintf(stderr, "prompt encode failed: %s\n", reason);
            }
            free(chat);
            free(request.prefix);
            continue;
        }
        free(chat);
        /* Tokenize the declared prefix and require it to be a token
         * prefix of the prompt. BPE need not split a substring the same
         * way, and a mismatched span would restore GDN state that does
         * not belong to these tokens, so a failed check simply drops
         * back to the ordinary paths. */
        uint32_t prefix_count = 0;
        if (request.prefix != NULL && continuation) {
            size_t encoded = 0;
            if (qwen38_tokenizer_encode(tokenizer, request.prefix,
                                        prefix_ids, capacity, &encoded,
                                        error, sizeof(error)) == 0 &&
                encoded != 0 && encoded < prompt_count &&
                memcmp(prompt_ids, prefix_ids,
                       encoded * sizeof(uint32_t)) == 0) {
                prefix_count = (uint32_t)encoded;
            } else if (getenv("QWEN38_CONTINUE_DEBUG") != NULL) {
                fprintf(stderr, "[continue] declared prefix is not a "
                        "token prefix of the prompt; ignored\n");
            }
        }
        free(request.prefix);
        request.prefix = NULL;
        if (prompt_count + 1 > capacity) {
            if (machine) {
                printf("X \"prompt (%zu tokens) exceeds context %u\"\n",
                       prompt_count, capacity);
                fflush(stdout);
            } else {
                fprintf(stderr, "prompt (%zu tokens) exceeds context "
                        "%u\n", prompt_count, capacity);
            }
            continue;
        }
        /* A long prompt shrinks this reply's budget instead of failing. */
        uint32_t budget = maximum_new;
        if (request.max_new != 0 && request.max_new < budget)
            budget = request.max_new;
        if (prompt_count + budget > capacity)
            budget = capacity - (uint32_t)prompt_count;

        if (!machine) {
            printf("\nModel>\n");
            fflush(stdout);
        }
        double prompt_start = seconds_now();
        /* Phase timers: with a warm checkpoint the visible cost of a
         * turn is no longer the prefill, so the breakdown has to be
         * measurable rather than guessed at. */
        double stage_restore = 0.0;
        double stage_prefill = 0.0;
        double stage_forward = 0.0;
        double stage_save = 0.0;
        double stage_mark = prompt_start;
        uint32_t start_position = 0;
        if (continuation && history_count != 0 &&
            prompt_count > history_count &&
            memcmp(prompt_ids, history,
                   (size_t)history_count * sizeof(uint32_t)) == 0) {
            /* The request extends the live state exactly. */
            start_position = history_count;
        } else if (continuation && snapshot_count != 0 &&
                   prompt_count > snapshot_count &&
                   memcmp(prompt_ids, snapshot_tokens,
                          (size_t)snapshot_count *
                              sizeof(uint32_t)) == 0 &&
                   qwen38_m3_model_prefix_restore(
                       model, error, sizeof(error)) == 0) {
            /* The request extends the last prompt boundary: rewind the
             * GDN state and prefill the reply plus the new turn. */
            start_position = snapshot_count;
            history_count = 0;
        } else if (continuation && system_valid && system_count != 0 &&
                   prompt_count > system_count &&
                   memcmp(prompt_ids, system_tokens,
                          (size_t)system_count * sizeof(uint32_t)) == 0 &&
                   qwen38_m3_model_prefix_restore_slot(
                       model, QWEN38_M3_PREFIX_SYSTEM, error,
                       sizeof(error)) == 0) {
            /* A new conversation that opens with the same system turn:
             * rewind to that boundary instead of prefilling it again. */
            start_position = system_count;
            history_count = 0;
        } else {
            if (getenv("QWEN38_CONTINUE_DEBUG") != NULL &&
                history_count != 0)
                fprintf(stderr, "[continue] no match: history %u "
                        "snapshot %u system %u prompt %zu\n",
                        history_count, snapshot_count, system_count,
                        prompt_count);
            /* Reset zeroes the attention KV, so the system checkpoint's
             * keys and values go with it. */
            qwen38_m3_model_reset(model);
            history_count = 0;
            system_valid = 0;
        }
        stage_restore = seconds_now() - stage_mark;
        stage_mark = seconds_now();
        if (getenv("QWEN38_CONTINUE_DEBUG") != NULL)
            fprintf(stderr, "[continue] prefill from %u of %zu\n",
                    start_position, prompt_count);
        const float *logits = NULL;
        size_t logit_count = 0;
        qwen38_m3_decode_result result = {0};
        int failed = 0;
        uint32_t prefill_end = (uint32_t)(prompt_count - 1);
        if (prefill_end > start_position) {
            /* Prefilling from zero with a declared prefix stops at that
             * boundary to take the checkpoint, then continues. The only
             * cost is one partly filled prompt chunk at the seam. */
            uint32_t boundary = start_position;
            if (start_position == 0 && prefix_count != 0 &&
                prefix_count < prefill_end)
                boundary = prefix_count;
            if (prefill_span(model, prompt_ids, start_position, boundary,
                             error, sizeof(error)) != 0) {
                fprintf(stderr, "prefill failed: %s\n", error);
                failed = 1;
            }
            if (!failed && boundary != start_position) {
                stage_prefill += seconds_now() - stage_mark;
                stage_mark = seconds_now();
                if (qwen38_m3_model_prefix_save_slot(
                        model, QWEN38_M3_PREFIX_SYSTEM, error,
                        sizeof(error)) == 0) {
                    memcpy(system_tokens, prompt_ids,
                           (size_t)boundary * sizeof(uint32_t));
                    system_count = boundary;
                    system_valid = 1;
                } else {
                    system_valid = 0;
                }
                stage_save += seconds_now() - stage_mark;
                stage_mark = seconds_now();
            }
            if (!failed && prefill_span(model, prompt_ids, boundary,
                                        prefill_end, error,
                                        sizeof(error)) != 0) {
                fprintf(stderr, "prefill failed: %s\n", error);
                failed = 1;
            }
        }
        stage_prefill += seconds_now() - stage_mark;
        stage_mark = seconds_now();
        if (!failed && qwen38_m3_model_forward(
                model, prompt_ids[prompt_count - 1],
                (uint32_t)(prompt_count - 1), &result, &logits,
                &logit_count, error, sizeof(error)) != 0) {
            fprintf(stderr, "prompt forward failed: %s\n", error);
            failed = 1;
        }
        if (failed) {
            qwen38_m3_model_reset(model);
            history_count = 0;
            system_valid = 0;
            continue;
        }
        stage_forward = seconds_now() - stage_mark;
        stage_mark = seconds_now();
        memcpy(history, prompt_ids,
               (size_t)prompt_count * sizeof(uint32_t));
        history_count = (uint32_t)prompt_count;
        if (continuation) {
            if (qwen38_m3_model_prefix_save(model, error,
                                            sizeof(error)) == 0) {
                memcpy(snapshot_tokens, prompt_ids,
                       (size_t)prompt_count * sizeof(uint32_t));
                snapshot_count = (uint32_t)prompt_count;
            } else {
                snapshot_count = 0;
            }
        }
        stage_save += seconds_now() - stage_mark;

        qwen38_sampler sampler = {
            .temperature = request.temperature, .top_k = request.top_k,
            .top_p = request.top_p, .min_p = request.min_p,
            .presence_penalty = request.presence_penalty, .state = seed
        };
        qwen38_sampler_begin(&sampler, QWEN38_TOKENIZER_VOCAB);
        int greedy_request = request.temperature <= 0.0f ||
                             request.top_k == 1;
        int use_mtp = mtp && greedy_request;
        uint32_t generated_count = 0;
        size_t visible_count = 0;
        stream_text_reset(&stream);
        double first_token_seconds = -1.0;
        size_t mtp_steps = 0;
        size_t mtp_accepts = 0;

        if (use_mtp) {
            uint32_t pending;
            size_t sample_count = logit_count;
            if (sample_count > QWEN38_TOKENIZER_VOCAB)
                sample_count = QWEN38_TOKENIZER_VOCAB;
            if (qwen38_sample_logits(&sampler, logits, sample_count,
                                     &pending) != 0) {
                fprintf(stderr, "sampling failed\n");
                failed = 1;
            }
            uint32_t mtp_position = (uint32_t)prompt_count;
            int done = failed;
            while (!done) {
                if (pending == QWEN38_END_OF_TEXT ||
                    pending == QWEN38_IM_END ||
                    generated_count + 1 >= budget ||
                    (uint64_t)mtp_position + 2 > capacity) {
                    generated[generated_count++] = pending;
                    break;
                }
                uint32_t step_emitted[8];
                uint32_t step_count = 0;
                int step_accepted = 0;
                qwen38_m3_model_mtp_context(model, history,
                                            history_count);
                if (qwen38_m3_model_mtp_step(
                        model, &pending, &mtp_position, step_emitted,
                        &step_count, &step_accepted, error,
                        sizeof(error)) != 0) {
                    fprintf(stderr, "MTP step failed: %s\n", error);
                    failed = 1;
                    break;
                }
                ++mtp_steps;
                mtp_accepts += (size_t)step_accepted;
                for (uint32_t i = 0; i < step_count; ++i)
                    if (history_count < capacity)
                        history[history_count++] = step_emitted[i];
                for (uint32_t i = 0; i < step_count && !done; ++i) {
                    generated[generated_count++] = step_emitted[i];
                    if (step_emitted[i] == QWEN38_END_OF_TEXT ||
                        step_emitted[i] == QWEN38_IM_END ||
                        generated_count >= budget)
                        done = 1;
                }
                visible_count = generated_count;
                if (generated_count != 0 &&
                    (generated[generated_count - 1] ==
                         QWEN38_END_OF_TEXT ||
                     generated[generated_count - 1] == QWEN38_IM_END))
                    visible_count = generated_count - 1;
                if (first_token_seconds < 0.0 && visible_count != 0)
                    first_token_seconds = seconds_now() - prompt_start;
                emit_new_text(tokenizer, generated, visible_count,
                              &stream, machine, error,
                              sizeof(error));
            }
            visible_count = generated_count;
            if (generated_count != 0 &&
                (generated[generated_count - 1] == QWEN38_END_OF_TEXT ||
                 generated[generated_count - 1] == QWEN38_IM_END))
                visible_count = generated_count - 1;
            if (first_token_seconds < 0.0 && visible_count != 0)
                first_token_seconds = seconds_now() - prompt_start;
            emit_new_text(tokenizer, generated, visible_count,
                          &stream, machine, error, sizeof(error));
        } else
        for (uint32_t index = 0; index < budget; ++index) {
            uint32_t token;
            size_t sample_count = logit_count;
            if (sample_count > QWEN38_TOKENIZER_VOCAB)
                sample_count = QWEN38_TOKENIZER_VOCAB;
            if (qwen38_sample_logits(&sampler, logits, sample_count,
                                     &token) != 0) {
                fprintf(stderr, "sampling failed\n");
                failed = 1;
                break;
            }
            generated[generated_count++] = token;
            if (token == QWEN38_END_OF_TEXT || token == QWEN38_IM_END)
                break;
            visible_count = generated_count;
            if (first_token_seconds < 0.0)
                first_token_seconds = seconds_now() - prompt_start;
            if (index + 1 == budget) {
                emit_new_text(tokenizer, generated, visible_count,
                              &stream, machine, error,
                              sizeof(error));
                break;
            }
            qwen38_sampler_note(&sampler, token);
            uint32_t position = (uint32_t)prompt_count + index;
            if (qwen38_m3_model_forward_submit(
                    model, token, position, error, sizeof(error)) != 0) {
                fprintf(stderr, "generation submit failed: %s\n", error);
                failed = 1;
                break;
            }
            if (history_count < capacity)
                history[history_count++] = token;
            emit_new_text(tokenizer, generated, visible_count,
                          &stream, machine, error, sizeof(error));
            if (qwen38_m3_model_forward_wait(
                    model, &result, &logits, &logit_count,
                    error, sizeof(error)) != 0) {
                fprintf(stderr, "generation wait failed: %s\n", error);
                failed = 1;
                break;
            }
        }
        double total_seconds = seconds_now() - prompt_start;
        if (machine) {
            if (failed) {
                fputs("X ", stdout);
                put_json_string(error, strlen(error));
                putchar('\n');
            } else {
                int stopped = generated_count != 0 &&
                    (generated[generated_count - 1] ==
                         QWEN38_END_OF_TEXT ||
                     generated[generated_count - 1] == QWEN38_IM_END);
                printf("E {\"tokens\": %zu, \"prompt_tokens\": %zu, "
                       "\"first_token_s\": %.3f, \"total_s\": %.3f, "
                       "\"stop\": \"%s\", \"mtp_steps\": %zu, "
                       "\"mtp_accepted\": %zu, \"mtp_depth\": %d, "
                       "\"prefilled_from\": %u, \"restore_s\": %.3f, "
                       "\"prefill_s\": %.3f, \"forward_s\": %.3f, "
                       "\"save_s\": %.3f}\n",
                       visible_count, prompt_count,
                       first_token_seconds >= 0.0 ?
                           first_token_seconds : 0.0,
                       total_seconds,
                       stopped ? "stop" : "length",
                       mtp_steps, mtp_accepts, mtp_depth,
                       start_position, stage_restore, stage_prefill,
                       stage_forward, stage_save);
            }
            fflush(stdout);
        } else {
            printf("\n");
            fflush(stdout);
            if (!failed && visible_count > 1 &&
                total_seconds > first_token_seconds) {
                if (mtp_steps != 0)
                    fprintf(stderr, "[first token %.2f s, %zu tokens, "
                            "%.1f tok/s, drafts accepted %zu over "
                            "%zu steps]\n",
                            first_token_seconds, visible_count,
                            (double)(visible_count - 1) /
                            (total_seconds - first_token_seconds),
                            mtp_accepts, mtp_steps);
                else
                    fprintf(stderr, "[first token %.2f s, %zu tokens, "
                            "%.1f tok/s]\n", first_token_seconds,
                            visible_count,
                            (double)(visible_count - 1) /
                            (total_seconds - first_token_seconds));
            } else if (!failed && first_token_seconds >= 0.0) {
                fprintf(stderr, "[first token %.2f s]\n",
                        first_token_seconds);
            }
        }
        qwen38_sampler_release(&sampler);
        if (failed) {
            qwen38_m3_model_reset(model);
            history_count = 0;
        }
    }

    free(line);
    free(prompt_ids);
    free(generated);
    free(history);
    free(snapshot_tokens);
    qwen38_m3_model_close(model);
    qwen38_tokenizer_close(tokenizer);
    return 0;
}
