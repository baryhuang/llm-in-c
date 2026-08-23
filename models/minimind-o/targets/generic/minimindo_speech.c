#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "minimindo_audio_encoder.h"
#include "minimindo_mimi.h"
#include "minimindo_parallel.h"
#include "minimindo_talker.h"
#include "minimindo_thinker.h"
#include "minimindo_tokenizer.h"
#include "minimindo_volume.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <spawn.h>
#if defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#endif
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

typedef struct { float score; uint32_t id; } candidate;

static minimindo_thinker *resident_thinker;
static minimindo_talker *resident_talker;
static minimindo_tokenizer *resident_tokenizer;
static minimindo_mimi *resident_mimi;
static minimindo_audio_encoder *resident_audio_encoder;
static const uint32_t resident_context = 512;

typedef enum {
    HUB_LED_NONE = 0,
    HUB_LED_LISTENING,
    HUB_LED_THINKING,
    HUB_LED_PLAYBACK,
    HUB_LED_ERROR,
    HUB_LED_STOP_CLEAR,
    HUB_LED_STOP_ERROR
} hub_led_state;

typedef struct {
    pthread_t thread;
    int wake_read;
    int wake_write;
    atomic_int desired;
    int started;
} hub_led_worker;

static hub_led_worker resident_led;

static const char *hub_led_name(hub_led_state state)
{
    switch (state) {
    case HUB_LED_LISTENING: return "listening-green-very-slow";
    case HUB_LED_THINKING: return "thinking-yellow-slow";
    case HUB_LED_PLAYBACK: return "playback-deep-blue-slow";
    case HUB_LED_ERROR:
    case HUB_LED_STOP_ERROR: return "error-red-slow";
    case HUB_LED_STOP_CLEAR: return "supervisor-default";
    default: return NULL;
    }
}

#define HUB_LED_SUPERVISOR "/usr/local/bin/supervisor"

static int hub_led_execute_command(const char *command)
{
    char *const arguments[] = {
        (char *)HUB_LED_SUPERVISOR, (char *)"led",
        (char *)command, NULL
    };
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) return -1;
    (void)posix_spawn_file_actions_addopen(
        &actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    (void)posix_spawn_file_actions_addopen(
        &actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    pid_t child = -1;
    const int spawn_result = posix_spawn(
        &child, arguments[0], &actions, NULL, arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    if (spawn_result != 0) return -1;
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) return -1;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : -1;
}

static int hub_led_execute(hub_led_state state)
{
    int result = 0;
    /* The resident ThirdReality supervisor owns the RGB GPIO lines and owns
     * the blink timer.  These named states produce hardware-timed patterns:
     * firmware_updating = green 1 s on/1 s off,
     * offline = yellow 1 s on/1 s off, and
     * wifi_config_pending = pure blue 0.5 s on/0.5 s off.
     * Clear only the higher tiers that could mask the requested pattern. */
    switch (state) {
    case HUB_LED_LISTENING:
        return hub_led_execute_command("firmware_updating");
    case HUB_LED_THINKING:
        if (hub_led_execute_command("sys_event_off") != 0) result = -1;
        if (hub_led_execute_command("wifi_config_stopped") != 0) result = -1;
        if (hub_led_execute_command("offline") != 0) result = -1;
        return result;
    case HUB_LED_PLAYBACK:
        if (hub_led_execute_command("sys_event_off") != 0) result = -1;
        if (hub_led_execute_command("wifi_config_pending") != 0) result = -1;
        return result;
    case HUB_LED_ERROR:
    case HUB_LED_STOP_ERROR:
        if (hub_led_execute_command("sys_event_off") != 0) result = -1;
        if (hub_led_execute_command("wifi_config_stopped") != 0) result = -1;
        if (hub_led_execute_command("error") != 0) result = -1;
        return result;
    case HUB_LED_STOP_CLEAR:
        if (hub_led_execute_command("sys_event_off") != 0) result = -1;
        if (hub_led_execute_command("wifi_config_stopped") != 0) result = -1;
        if (hub_led_execute_command("clear") != 0) result = -1;
        return result;
    default:
        return -1;
    }
}

static void *hub_led_thread(void *opaque)
{
    hub_led_worker *worker = opaque;
    for (;;) {
        unsigned char wake[32];
        const ssize_t count = read(worker->wake_read, wake, sizeof(wake));
        if (count < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (count == 0) break;
        const hub_led_state state = (hub_led_state)atomic_exchange_explicit(
            &worker->desired, HUB_LED_NONE, memory_order_acq_rel);
        if (state == HUB_LED_NONE) continue;
        const int result = hub_led_execute(state);
        printf("LED state=%s result=%d\n", hub_led_name(state), result);
        fflush(stdout);
        if (state == HUB_LED_STOP_CLEAR || state == HUB_LED_STOP_ERROR) break;
    }
    return NULL;
}

static int hub_led_start(void)
{
    /* Only the ThirdReality hub ships the supervisor that owns the RGB GPIO
     * lines. Without it every state change would spawn a doomed process, so
     * boards such as RK3588 run the voice path with no LED worker at all.
     * MINIMINDO_HUB_LED=0 disables it explicitly. */
    const char *configured = getenv("MINIMINDO_HUB_LED");
    if (configured != NULL && configured[0] == '0') return -1;
    if (access(HUB_LED_SUPERVISOR, X_OK) != 0) return -1;

    hub_led_worker *worker = &resident_led;
    memset(worker, 0, sizeof(*worker));
    worker->wake_read = -1;
    worker->wake_write = -1;
    atomic_init(&worker->desired, HUB_LED_NONE);
    int wake[2];
    if (pipe(wake) != 0) return -1;
    worker->wake_read = wake[0];
    worker->wake_write = wake[1];
    const int flags = fcntl(worker->wake_write, F_GETFL, 0);
    if (flags < 0 ||
        fcntl(worker->wake_write, F_SETFL, flags | O_NONBLOCK) != 0 ||
        fcntl(worker->wake_read, F_SETFD, FD_CLOEXEC) != 0 ||
        fcntl(worker->wake_write, F_SETFD, FD_CLOEXEC) != 0 ||
        pthread_create(&worker->thread, NULL, hub_led_thread, worker) != 0) {
        close(worker->wake_read);
        close(worker->wake_write);
        worker->wake_read = -1;
        worker->wake_write = -1;
        return -1;
    }
    worker->started = 1;
    return 0;
}

static void hub_led_publish(hub_led_state state)
{
    hub_led_worker *worker = &resident_led;
    if (!worker->started) return;
    atomic_store_explicit(&worker->desired, state, memory_order_release);
    const unsigned char wake = 1U;
    const ssize_t ignored = write(worker->wake_write, &wake, sizeof(wake));
    (void)ignored;
}

static void hub_led_stop(hub_led_state final_state)
{
    hub_led_worker *worker = &resident_led;
    if (!worker->started) return;
    hub_led_publish(final_state);
    pthread_join(worker->thread, NULL);
    close(worker->wake_read);
    close(worker->wake_write);
    worker->wake_read = -1;
    worker->wake_write = -1;
    worker->started = 0;
}

static double monotonic_seconds(void);
static double process_cpu_seconds(void);

static int ensure_resident(const char *thinker_path,const char *talker_path,
                           const char *tokenizer_path,const char *mimi_path,
                           const char *audio_encoder_path,char *error,size_t error_capacity)
{
    if(!resident_thinker)resident_thinker=minimindo_thinker_open(thinker_path,resident_context,error,error_capacity);
    if(!resident_talker)resident_talker=minimindo_talker_open(talker_path,resident_context,error,error_capacity);
    if(!resident_tokenizer)resident_tokenizer=minimindo_tokenizer_open(tokenizer_path,error,error_capacity);
    if(!resident_mimi)resident_mimi=minimindo_mimi_open(mimi_path,512,error,error_capacity);
    if(audio_encoder_path&&!resident_audio_encoder)resident_audio_encoder=minimindo_audio_encoder_open(audio_encoder_path,error,error_capacity);
    return resident_thinker&&resident_talker&&resident_tokenizer&&resident_mimi&&
           (!audio_encoder_path||resident_audio_encoder)?0:-1;
}

static uint64_t random_state;
static uint64_t random_u64(void)
{
    uint64_t x = random_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    random_state = x;
    return x * UINT64_C(2685821657736338717);
}
static double random_unit(void) { return (random_u64() >> 11) * 0x1.0p-53; }

static int descending(const void *left, const void *right)
{
    const candidate *a = left, *b = right;
    return a->score < b->score ? 1 : a->score > b->score ? -1 : 0;
}

static uint32_t sample_top_p(const float *logits, uint32_t count,
                             float temperature, float top_p,
                             const uint32_t *history, size_t history_count,
                             float repetition_penalty, candidate *work)
{
    float maximum = -FLT_MAX;
    for (uint32_t i = 0; i < count; ++i) {
        float score = logits[i];
        if (repetition_penalty != 1.0f)
            for (size_t j = 0; j < history_count; ++j)
                if (history[j] == i) { score = score > 0 ? score / repetition_penalty : score * repetition_penalty; break; }
        work[i] = (candidate){score / temperature, i};
        if (work[i].score > maximum) maximum = work[i].score;
    }
    qsort(work, count, sizeof(*work), descending);
    double denominator = 0;
    for (uint32_t i = 0; i < count; ++i) denominator += exp(work[i].score - maximum);
    double cumulative = 0; uint32_t kept = count;
    for (uint32_t i = 0; i < count; ++i) {
        cumulative += exp(work[i].score - maximum) / denominator;
        if (cumulative >= top_p) { kept = i + 1; break; }
    }
    double selected_total = 0;
    for (uint32_t i = 0; i < kept; ++i) selected_total += exp(work[i].score - maximum);
    const double target = random_unit() * selected_total;
    double running = 0;
    for (uint32_t i = 0; i < kept; ++i) {
        running += exp(work[i].score - maximum);
        if (running >= target) return work[i].id;
    }
    return work[kept - 1].id;
}

static uint32_t sample_top_k(const float *logits, uint32_t count, uint32_t top_k,
                             float temperature, const uint32_t *history,
                             size_t history_count, float repetition_penalty,
                             candidate *work)
{
    if (top_k > count) top_k = count;
    for (uint32_t i = 0; i < top_k; ++i) {
        float score = logits[i];
        for (size_t j = 0; j < history_count; ++j)
            if (history[j] == i) { score = score > 0 ? score / repetition_penalty : score * repetition_penalty; break; }
        work[i] = (candidate){score / temperature, i};
    }
    /* Keep only the best k in a min-heap instead of sorting all 2,112 logits. */
    for (uint32_t parent = top_k / 2U; parent != 0U; ) {
        --parent;
        uint32_t root = parent;
        while (root * 2U + 1U < top_k) {
            uint32_t child = root * 2U + 1U;
            if (child + 1U < top_k &&
                work[child + 1U].score < work[child].score) ++child;
            if (work[root].score <= work[child].score) break;
            candidate swap = work[root]; work[root] = work[child];
            work[child] = swap; root = child;
        }
    }
    for (uint32_t i = top_k; i < count; ++i) {
        float score = logits[i];
        for (size_t j = 0; j < history_count; ++j)
            if (history[j] == i) { score = score > 0 ? score / repetition_penalty : score * repetition_penalty; break; }
        candidate value = {score / temperature, i};
        if (value.score <= work[0].score) continue;
        work[0] = value;
        uint32_t root = 0;
        while (root * 2U + 1U < top_k) {
            uint32_t child = root * 2U + 1U;
            if (child + 1U < top_k &&
                work[child + 1U].score < work[child].score) ++child;
            if (work[root].score <= work[child].score) break;
            candidate swap = work[root]; work[root] = work[child];
            work[child] = swap; root = child;
        }
    }
    qsort(work, top_k, sizeof(*work), descending);
    const float maximum = work[0].score;
    double total = 0;
    for (uint32_t i = 0; i < top_k; ++i) total += exp(work[i].score - maximum);
    const double target = random_unit() * total;
    double running = 0;
    for (uint32_t i = 0; i < top_k; ++i) {
        running += exp(work[i].score - maximum);
        if (running >= target) return work[i].id;
    }
    return work[top_k - 1].id;
}

static const char prompt_prefix[] = "<|im_start|>user\n";
static const char prompt_suffix[] =
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

static char *format_prompt(const char *user)
{
    const size_t bytes =
        sizeof(prompt_prefix) + strlen(user) + sizeof(prompt_suffix);
    char *text = malloc(bytes);
    if (text != NULL)
        snprintf(text,bytes,"%s%s%s",prompt_prefix,user,prompt_suffix);
    return text;
}

static int encode(const minimindo_tokenizer *tokenizer, const char *text,
                  uint32_t **ids, size_t *count, char *error, size_t error_capacity)
{
    size_t required = 0;
    (void)minimindo_tokenizer_encode(tokenizer, text, NULL, 0, &required, error, error_capacity);
    *ids = malloc(required * sizeof(**ids));
    if (*ids == NULL) return -1;
    if (minimindo_tokenizer_encode(tokenizer, text, *ids, required, count,
                                   error, error_capacity) != 0) { free(*ids); *ids = NULL; return -1; }
    return 0;
}

static char *decode_text(const minimindo_tokenizer *tokenizer,
                         const uint32_t *ids, size_t count,
                         char *error, size_t error_capacity)
{
    size_t required = 0;
    (void)minimindo_tokenizer_decode(tokenizer, ids, count, NULL, 0, &required, error, error_capacity);
    char *text = malloc(required + 1);
    if (text == NULL) return NULL;
    if (minimindo_tokenizer_decode(tokenizer, ids, count, text, required + 1,
                                   &required, error, error_capacity) != 0) { free(text); return NULL; }
    return text;
}

static int generated_sentence_complete(const minimindo_tokenizer *tokenizer,
                                       const uint32_t *ids,size_t count,
                                       char *error,size_t error_capacity)
{
    char *text = decode_text(tokenizer,ids,count,error,error_capacity);
    if (text == NULL) return 0;
    size_t bytes=strlen(text);
    while(bytes>0U&&(text[bytes-1U]==' '||text[bytes-1U]=='\t'||
                     text[bytes-1U]=='\r'||text[bytes-1U]=='\n'))--bytes;
    int complete=0;
    if(bytes>0U&&(text[bytes-1U]=='.'||text[bytes-1U]=='!'||
                  text[bytes-1U]=='?'))complete=1;
    if(bytes>=3U&&(!memcmp(text+bytes-3U,"。",3U)||
                   !memcmp(text+bytes-3U,"！",3U)||
                   !memcmp(text+bytes-3U,"？",3U)))complete=1;
    free(text);
    return complete;
}

static void u16(FILE *stream, uint16_t value)
{ unsigned char b[2] = {(unsigned char)value,(unsigned char)(value>>8)}; fwrite(b,1,2,stream); }
static void u32(FILE *stream, uint32_t value)
{ unsigned char b[4] = {(unsigned char)value,(unsigned char)(value>>8),(unsigned char)(value>>16),(unsigned char)(value>>24)}; fwrite(b,1,4,stream); }
static int write_wav(const char *path, const float *audio, size_t samples, uint32_t rate)
{
    FILE *f = fopen(path, "wb"); if (!f || samples > UINT32_MAX / 2) return -1;
    uint32_t bytes = (uint32_t)samples * 2;
    fwrite("RIFF",1,4,f); u32(f,36+bytes); fwrite("WAVEfmt ",1,8,f); u32(f,16);
    u16(f,1); u16(f,1); u32(f,rate); u32(f,rate*2); u16(f,2); u16(f,16);
    fwrite("data",1,4,f); u32(f,bytes);
    for (size_t i=0;i<samples;++i) { float x=audio[i]; if(x>1)x=1;if(x< -1)x=-1;u16(f,(uint16_t)(int16_t)lrintf(x*32767)); }
    int failed=ferror(f); fclose(f); return failed ? -1 : 0;
}

typedef struct {
    minimindo_mimi *model;
    minimindo_mimi_stream *stream;
    pthread_t thread;
    /* This mutex protects only the condition-variable sleep boundary.  The
     * SPSC rings below are published with release/acquire atomics; model,
     * codes and PCM each have exactly one writer. */
    pthread_mutex_t wait_mutex;
    pthread_cond_t ready;
    uint32_t *codes;
    float *audio;
    size_t capacity_frames;
    atomic_size_t queued_frames;
    atomic_size_t decoded_frames;
    size_t decode_overlap_frames;
    size_t samples;
    int drain_threads;
    atomic_int producer_done;
    atomic_int failed;
    unsigned turn;
    double inference_start;
    double decode_end;
    double decode_cpu_end;
    char error[256];
} mimi_decode_worker;

static inline void speech_spin_pause(void)
{
#if defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#elif defined(__x86_64__)
    __asm__ volatile("pause" ::: "memory");
#else
    atomic_signal_fence(memory_order_seq_cst);
#endif
}

static void log_thread_placement(const char *stage, unsigned turn,
                                 int requested_threads)
{
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    const int affinity_result = sched_getaffinity(0, sizeof(set), &set);
    char cpus[64];
    size_t used = 0;
    cpus[0] = '\0';
    if (affinity_result == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &set)) continue;
            const int written = snprintf(cpus + used, sizeof(cpus) - used,
                                         "%s%d", used == 0 ? "" : ",", cpu);
            if (written < 0 || (size_t)written >= sizeof(cpus) - used) break;
            used += (size_t)written;
        }
    }
    printf("THREAD stage=%s turn=%u tid=%ld cpu=%d affinity=%s "
           "threads=%d\n", stage, turn, (long)syscall(SYS_gettid),
           sched_getcpu(), affinity_result == 0 ? cpus : "error",
           requested_threads);
#else
    printf("THREAD stage=%s turn=%u threads=%d\n",
           stage, turn, requested_threads);
#endif
    fflush(stdout);
}

static void *mimi_decode_thread(void *opaque)
{
    mimi_decode_worker *worker = opaque;
    const size_t samples_per_frame =
        minimindo_mimi_samples_for_frames(worker->model, 1);
    int overlap_session = 0;
    int drain_session = 0;
    (void)minimindo_parallel_pin_current(3U);
    /* During Talker generation CPU0 is the producer, CPU3 is this decoder,
     * and the two dispatchers share the persistent CPU1/2 workers at matrix
     * boundaries.  No compute lock is held; only pool enqueue/dequeue uses a
     * tiny spin lock. */
    if (minimindo_parallel_session_begin(3U) != 0) {
        snprintf(worker->error,sizeof(worker->error),
                 "Mimi overlap compute session failed");
        atomic_store_explicit(&worker->failed,1,memory_order_release);
        return NULL;
    }
    overlap_session = 1;
    log_thread_placement("mimi_start", worker->turn, 3);
    while (1) {
        uint32_t frame[MINIMINDO_MIMI_CODEBOOKS];
        const size_t decoded = atomic_load_explicit(
            &worker->decoded_frames, memory_order_relaxed);
        const size_t queued = atomic_load_explicit(
            &worker->queued_frames, memory_order_acquire);
        const int producer_done = atomic_load_explicit(
            &worker->producer_done, memory_order_acquire);
        const int failed = atomic_load_explicit(
            &worker->failed, memory_order_acquire);
        if (failed || (producer_done && decoded == queued)) {
            if (!failed) {
                worker->decode_end = monotonic_seconds();
                worker->decode_cpu_end = process_cpu_seconds();
            }
            break;
        }
        if (decoded == queued) {
            /* This thread owns CPU 3 while Talker is active.  Spinning here
             * avoids a futex round trip for every 80 ms codec frame. */
            speech_spin_pause();
            continue;
        }
        const size_t index = decoded;
        for (uint32_t codebook = 0;
             codebook < MINIMINDO_MIMI_CODEBOOKS; ++codebook)
            frame[codebook] =
                worker->codes[index*MINIMINDO_MIMI_CODEBOOKS+codebook];
        if (producer_done && !drain_session) {
            minimindo_parallel_session_end();
            overlap_session = 0;
            (void)minimindo_parallel_pin_current(0U);
            if (minimindo_parallel_session_begin(
                    (unsigned)worker->drain_threads) != 0) {
                snprintf(worker->error,sizeof(worker->error),
                         "Mimi drain compute session failed");
                atomic_store_explicit(&worker->failed,1,
                                      memory_order_release);
                break;
            }
            drain_session = 1;
            log_thread_placement("mimi_drain",worker->turn,
                                 worker->drain_threads);
        }
        size_t produced = 0;
        const double frame_start = monotonic_seconds();
        const double frame_cpu_start = process_cpu_seconds();
        const int result = minimindo_mimi_stream_decode(
            worker->stream, frame, 1,
            worker->audio + index * samples_per_frame,
            samples_per_frame,
            &produced, worker->error, sizeof(worker->error));
        if (result != 0 || produced != samples_per_frame) {
            atomic_store_explicit(&worker->failed, 1, memory_order_release);
        } else {
            worker->samples += produced;
            atomic_store_explicit(&worker->decoded_frames, index + 1U,
                                  memory_order_release);
        }
        const size_t decoded_snapshot = atomic_load_explicit(
            &worker->decoded_frames, memory_order_relaxed);
        const size_t queued_snapshot = atomic_load_explicit(
            &worker->queued_frames, memory_order_acquire);
        const int done_snapshot = atomic_load_explicit(
            &worker->producer_done, memory_order_acquire);
        pthread_mutex_lock(&worker->wait_mutex);
        pthread_cond_broadcast(&worker->ready);
        pthread_mutex_unlock(&worker->wait_mutex);
        const double frame_ms =
            (monotonic_seconds() - frame_start) * 1000.0;
        const double frame_cpu_ms =
            (process_cpu_seconds() - frame_cpu_start) * 1000.0;
        const double audio_ms = produced * 1000.0 /
            minimindo_mimi_sample_rate(worker->model);
        printf("STREAM stage=mimi_decode turn=%u frame=%zu queued=%zu "
               "producer_done=%d frame_ms=%.1f frame_cpu_ms=%.1f "
               "rtf=%.3f elapsed_ms=%.0f\n",
               worker->turn,decoded_snapshot,queued_snapshot,done_snapshot,
               frame_ms, frame_cpu_ms,
               audio_ms > 0.0 ? frame_ms / audio_ms : 0.0,
               (monotonic_seconds()-worker->inference_start)*1000.0);
        fflush(stdout);
    }
    if (drain_session) minimindo_parallel_session_end();
    if (overlap_session) minimindo_parallel_session_end();
    return NULL;
}

static int mimi_decode_worker_start(mimi_decode_worker *worker,
                                    minimindo_mimi *model,
                                    size_t capacity_frames,unsigned turn,
                                    double inference_start)
{
    memset(worker, 0, sizeof(*worker));
    worker->model = model;
    worker->capacity_frames = capacity_frames;
    worker->turn = turn;
    worker->inference_start = inference_start;
    worker->drain_threads = 4;
    atomic_init(&worker->queued_frames, 0U);
    atomic_init(&worker->decoded_frames, 0U);
    atomic_init(&worker->producer_done, 0);
    atomic_init(&worker->failed, 0);
    const size_t capacity_samples =
        minimindo_mimi_samples_for_frames(model, capacity_frames);
    worker->codes = calloc(capacity_frames * MINIMINDO_MIMI_CODEBOOKS,
                           sizeof(*worker->codes));
    worker->audio = malloc(capacity_samples * sizeof(*worker->audio));
    worker->stream = minimindo_mimi_stream_open(
        model, worker->error, sizeof(worker->error));
    if (worker->codes == NULL || worker->audio == NULL ||
        worker->stream == NULL) goto fail;
    if (pthread_mutex_init(&worker->wait_mutex, NULL) != 0) goto fail;
    if (pthread_cond_init(&worker->ready, NULL) != 0) {
        pthread_mutex_destroy(&worker->wait_mutex);
        goto fail;
    }
    if (pthread_create(&worker->thread, NULL, mimi_decode_thread, worker) != 0) {
        pthread_cond_destroy(&worker->ready);
        pthread_mutex_destroy(&worker->wait_mutex);
        goto fail;
    }
    return 0;
fail:
    minimindo_mimi_stream_close(worker->stream);
    worker->stream = NULL;
    free(worker->codes);
    worker->codes = NULL;
    free(worker->audio);
    worker->audio = NULL;
    return -1;
}

static int mimi_decode_worker_push(mimi_decode_worker *worker,
                                   const uint32_t frame[MINIMINDO_MIMI_CODEBOOKS])
{
    if (atomic_load_explicit(&worker->failed, memory_order_acquire))
        return -1;
    const size_t queued = atomic_load_explicit(
        &worker->queued_frames, memory_order_relaxed);
    if (queued >= worker->capacity_frames) return -1;
    for (uint32_t codebook = 0;
         codebook < MINIMINDO_MIMI_CODEBOOKS; ++codebook)
        worker->codes[queued * MINIMINDO_MIMI_CODEBOOKS +
                      codebook] = frame[codebook];
    const size_t queued_snapshot = queued + 1U;
    atomic_store_explicit(&worker->queued_frames, queued_snapshot,
                          memory_order_release);
    printf("STREAM stage=talker_produce turn=%u frame=%zu elapsed_ms=%.0f\n",
           worker->turn,queued_snapshot,
           (monotonic_seconds()-worker->inference_start)*1000.0);
    fflush(stdout);
    return 0;
}

static void mimi_decode_worker_signal_done(mimi_decode_worker *worker)
{
    const int was_done = atomic_exchange_explicit(
        &worker->producer_done, 1, memory_order_acq_rel);
    if (!was_done)
        worker->decode_overlap_frames = atomic_load_explicit(
            &worker->decoded_frames, memory_order_acquire);
    const size_t queued_snapshot = atomic_load_explicit(
        &worker->queued_frames, memory_order_acquire);
    const size_t decoded_snapshot = atomic_load_explicit(
        &worker->decoded_frames, memory_order_acquire);
    pthread_mutex_lock(&worker->wait_mutex);
    pthread_cond_broadcast(&worker->ready);
    pthread_mutex_unlock(&worker->wait_mutex);
    if (!was_done) {
        printf("STREAM stage=talker_done turn=%u frames=%zu decoded=%zu "
               "elapsed_ms=%.0f\n",worker->turn,queued_snapshot,
               decoded_snapshot,
               (monotonic_seconds()-worker->inference_start)*1000.0);
        fflush(stdout);
    }
}

static int mimi_decode_worker_finish(mimi_decode_worker *worker)
{
    mimi_decode_worker_signal_done(worker);
    pthread_join(worker->thread, NULL);
    pthread_cond_destroy(&worker->ready);
    pthread_mutex_destroy(&worker->wait_mutex);
    minimindo_mimi_stream_close(worker->stream);
    worker->stream = NULL;
    free(worker->codes);
    worker->codes = NULL;
    return atomic_load_explicit(&worker->failed, memory_order_acquire) ? -1 : 0;
}

static int16_t pcm16(float sample)
{
    if (sample > 1.0f) sample = 1.0f;
    if (sample < -1.0f) sample = -1.0f;
    return (int16_t)lrintf(sample * 32767.0f);
}

static size_t playback_start_frames(size_t total_frames)
{
    (void)total_frames;
    /* Production invariant: publish the first causal 80 ms PCM frame to ALSA
     * immediately. Continuity must come from codec throughput, never a hidden
     * response buffer or producer-EOS gate. */
    return 1U;
}

static int stream_worker_to_alsa(mimi_decode_worker *worker,
                                 const char *device, unsigned turn,
                                 double inference_start,
                                 double *first_audio_ms,
                                 size_t *queue_waits,
                                 double *queue_wait_ms)
{
    const size_t samples_per_frame =
        minimindo_mimi_samples_for_frames(worker->model, 1);
    /* End-to-end output streaming is mandatory: PCM playback starts on the
     * first decoded frame while Talker and Mimi continue producing. */
    pthread_mutex_lock(&worker->wait_mutex);
    while (!atomic_load_explicit(&worker->failed, memory_order_acquire)) {
        const size_t queued = atomic_load_explicit(
            &worker->queued_frames, memory_order_acquire);
        const size_t decoded = atomic_load_explicit(
            &worker->decoded_frames, memory_order_acquire);
        const int done = atomic_load_explicit(
            &worker->producer_done, memory_order_acquire);
        const size_t target = playback_start_frames(queued);
        if (decoded >= target || (done && decoded != 0U)) break;
        pthread_cond_wait(&worker->ready, &worker->wait_mutex);
    }
    const size_t buffered_frames = atomic_load_explicit(
        &worker->decoded_frames, memory_order_acquire);
    const size_t queued_at_start = atomic_load_explicit(
        &worker->queued_frames, memory_order_acquire);
    const size_t target_frames =
        playback_start_frames(queued_at_start);
    const int failed = atomic_load_explicit(
        &worker->failed, memory_order_acquire);
    const int empty = atomic_load_explicit(
        &worker->producer_done, memory_order_acquire) && queued_at_start == 0U;
    pthread_mutex_unlock(&worker->wait_mutex);
    if (failed || empty) return -1;

    char command[256];
    snprintf(command, sizeof(command),
             "aplay -q -D %s -t raw -f S16_LE -c 1 -r %u",
             device, minimindo_mimi_sample_rate(worker->model));
    FILE *player = popen(command, "w");
    int16_t *pcm = malloc(samples_per_frame * sizeof(*pcm));
    if (player == NULL || pcm == NULL) {
        if (player != NULL) (void)pclose(player);
        free(pcm);
        return -1;
    }
    (void)setvbuf(player, NULL, _IONBF, 0);

    int result = 0;
    for (size_t frame = 0; ; ++frame) {
        const double wait_start = monotonic_seconds();
        pthread_mutex_lock(&worker->wait_mutex);
        const int had_to_wait = atomic_load_explicit(
            &worker->decoded_frames, memory_order_acquire) <= frame;
        while (atomic_load_explicit(&worker->decoded_frames,
                                    memory_order_acquire) <= frame &&
               !atomic_load_explicit(&worker->failed,
                                     memory_order_acquire) &&
               !(atomic_load_explicit(&worker->producer_done,
                                      memory_order_acquire) &&
                 atomic_load_explicit(&worker->queued_frames,
                                      memory_order_acquire) <= frame))
            pthread_cond_wait(&worker->ready, &worker->wait_mutex);
        const int decode_failed = atomic_load_explicit(
            &worker->failed, memory_order_acquire);
        const int finished = atomic_load_explicit(
            &worker->producer_done, memory_order_acquire) &&
            atomic_load_explicit(&worker->queued_frames,
                                 memory_order_acquire) <= frame;
        pthread_mutex_unlock(&worker->wait_mutex);
        if (decode_failed) { result = -1; break; }
        if (finished) break;
        if (had_to_wait) {
            ++*queue_waits;
            *queue_wait_ms += (monotonic_seconds() - wait_start) * 1000.0;
        }
        const float *source = worker->audio + frame * samples_per_frame;
        for (size_t sample = 0; sample < samples_per_frame; ++sample)
            pcm[sample] = pcm16(source[sample]);
        if (fwrite(pcm, sizeof(*pcm), samples_per_frame, player) !=
            samples_per_frame) {
            result = -1;
            break;
        }
        if (frame == 0U) hub_led_publish(HUB_LED_PLAYBACK);
        const size_t decoded_snapshot = atomic_load_explicit(
            &worker->decoded_frames, memory_order_acquire);
        const size_t queued_snapshot = atomic_load_explicit(
            &worker->queued_frames, memory_order_acquire);
        const int producer_done = atomic_load_explicit(
            &worker->producer_done, memory_order_acquire);
        printf("STREAM stage=alsa_write turn=%u frame=%zu decoded=%zu "
               "queued=%zu producer_done=%d elapsed_ms=%.0f\n",
               turn,frame+1U,decoded_snapshot,queued_snapshot,producer_done,
               (monotonic_seconds()-inference_start)*1000.0);
        fflush(stdout);
        if (frame == 0U) {
            *first_audio_ms =
                (monotonic_seconds() - inference_start) * 1000.0;
            printf("EVENT first_audio turn=%u elapsed_ms=%.0f "
                   "buffered_frames=%zu target_frames=%zu buffered_ms=%zu\n",
                   turn, *first_audio_ms, buffered_frames,
                   target_frames,
                   buffered_frames * samples_per_frame * 1000U /
                       minimindo_mimi_sample_rate(worker->model));
            fflush(stdout);
        }
    }
    free(pcm);
    if (pclose(player) != 0) result = -1;
    return result;
}

typedef struct {
    mimi_decode_worker *decoder;
    const char *device;
    unsigned turn;
    double inference_start;
    double first_audio_ms;
    double queue_wait_ms;
    size_t queue_waits;
    int result;
} alsa_playback_worker;

static void *alsa_playback_thread(void *opaque)
{
    alsa_playback_worker *worker = opaque;
    worker->result = stream_worker_to_alsa(
        worker->decoder, worker->device, worker->turn,
        worker->inference_start, &worker->first_audio_ms,
        &worker->queue_waits, &worker->queue_wait_ms);
    return NULL;
}

static uint16_t read_u16(const unsigned char *p){return(uint16_t)(p[0]|p[1]<<8);}
static uint32_t read_u32(const unsigned char *p){return(uint32_t)(p[0]|p[1]<<8|p[2]<<16|p[3]<<24);}
static int load_wav(const char *path,int16_t **samples,size_t *count)
{
    FILE *f=fopen(path,"rb");if(!f)return -1;unsigned char riff[12];
    if(fread(riff,1,12,f)!=12||memcmp(riff,"RIFF",4)||memcmp(riff+8,"WAVE",4)){fclose(f);return -1;}
    uint16_t format=0,channels=0,bits=0;uint32_t rate=0,bytes=0;unsigned char *data=NULL;
    while(!feof(f)){unsigned char h[8];if(fread(h,1,8,f)!=8)break;uint32_t size=read_u32(h+4);
        if(!memcmp(h,"fmt ",4)){unsigned char v[40];if(size>sizeof(v)||fread(v,1,size,f)!=size)break;format=read_u16(v);channels=read_u16(v+2);rate=read_u32(v+4);bits=read_u16(v+14);}
        else if(!memcmp(h,"data",4)){data=malloc(size);if(!data||fread(data,1,size,f)!=size)break;bytes=size;}
        else fseek(f,size,SEEK_CUR);
        if(size&1)fseek(f,1,SEEK_CUR);}
    fclose(f);if(format!=1||channels!=1||rate!=16000||bits!=16||!data){free(data);return -1;}
    *samples=(int16_t *)data;*count=bytes/2;return 0;
}

static char *audio_prompt(size_t frames)
{
    static const char marker[]="<|audio_pad|>";size_t width=sizeof(marker)-1;
    char *text=malloc(frames*width+1);if(!text)return NULL;
    for(size_t i=0;i<frames;++i)
        memcpy(text+i*width,marker,width);
    text[frames*width]='\0';
    return text;
}

typedef struct {
    float *text_logits;
    size_t prompt_count;
    size_t audio_frames;
    minimindo_audio_encoder_profile audio_profile;
    double run_start;
    double audio_encode_seconds;
    double audio_encode_cpu_seconds;
    double prefill_thinker_seconds;
    double prefill_talker_seconds;
    double prefill_cpu_seconds;
} prefilled_audio_input;

static int run(const char *thinker_path, const char *talker_path,
               const char *tokenizer_path, const char *mimi_path,
               const char *prompt_text, const char *wav_path,
               const char *audio_encoder_path,const char *audio_path,
               const int16_t *provided_pcm,size_t provided_pcm_count,
               uint32_t max_tokens, uint64_t seed,
               const char *playback_device, unsigned live_turn,
               const prefilled_audio_input *prefilled)
{
    /* Text EOS and spoken EOS are not close in time: a 10-token Chinese
     * answer required another 46 Talker steps before all Mimi codebooks
     * stopped.  Keep a separate 15.36-second audio-tail budget; normal turns
     * still exit immediately when all eight codebooks report EOS. */
    enum { AUDIO_DRAIN_STEPS = 192 };
    const size_t generation_capacity =
        (size_t)max_tokens + AUDIO_DRAIN_STEPS;
    char error[256] = {0};
    const double run_start = prefilled?prefilled->run_start:monotonic_seconds();
    if(ensure_resident(thinker_path,talker_path,tokenizer_path,mimi_path,audio_encoder_path,error,sizeof(error))){fprintf(stderr,"%s\n",error);return -1;}
    minimindo_thinker *thinker=resident_thinker;minimindo_talker *talker=resident_talker;
    minimindo_tokenizer *tokenizer=resident_tokenizer;minimindo_mimi *mimi=resident_mimi;
    if(!prefilled){minimindo_thinker_reset(thinker);minimindo_talker_reset(talker);}
    const uint32_t tv = minimindo_thinker_vocab_size(thinker), av = minimindo_talker_vocab_size(talker);
    const uint32_t hidden = minimindo_thinker_hidden_size(thinker), pad = minimindo_talker_pad_token(talker);
    float *text_logits = malloc((size_t)tv*sizeof(float));
    float *audio_logits = malloc((size_t)8*av*sizeof(float)); float *bridge=NULL;
    candidate *work=malloc((tv>av?tv:av)*sizeof(candidate));
    uint32_t *generated=calloc(generation_capacity,sizeof(uint32_t));
    uint32_t *all_codes=malloc((size_t)8*generation_capacity*sizeof(uint32_t));
    uint32_t *frames=malloc((size_t)8*generation_capacity*sizeof(uint32_t));
    minimindo_audio_encoder *audio_encoder=resident_audio_encoder;minimindo_audio_encoder_profile audio_profile={0};float *audio_embeddings=NULL;size_t audio_frames=0;char *audio_user=NULL;int16_t *owned_pcm=NULL;
    char *formatted=NULL;uint32_t *prompt=NULL;size_t prompt_count=0;
    float *replacement_embeddings=NULL;uint8_t *replacement_mask=NULL;
    double audio_encode_seconds=0.0,audio_encode_cpu_seconds=0.0;
    double prefill_thinker_seconds=0.0,prefill_talker_seconds=0.0;
    double prefill_seconds=0.0,prefill_cpu_seconds=0.0;
    double prefill_end=0.0,prefill_cpu_end=0.0;
    uint32_t current_audio[8]; for(int i=0;i<8;++i) current_audio[i]=pad;
    if(!text_logits||!audio_logits||!work||!generated||!all_codes||!frames){
        fprintf(stderr,"inference allocation failed\n");return -1;
    }
    if(prefilled){
        if(!prefilled->text_logits||
           minimindo_thinker_position(thinker)!=prefilled->prompt_count||
           minimindo_talker_position(talker)!=prefilled->prompt_count){
            fprintf(stderr,"streaming input state mismatch: thinker=%u talker=%u prompt=%zu\n",
                    minimindo_thinker_position(thinker),
                    minimindo_talker_position(talker),prefilled->prompt_count);
            return -1;
        }
        memcpy(text_logits,prefilled->text_logits,(size_t)tv*sizeof(float));
        prompt_count=prefilled->prompt_count;audio_frames=prefilled->audio_frames;
        audio_profile=prefilled->audio_profile;
        audio_encode_seconds=prefilled->audio_encode_seconds;
        audio_encode_cpu_seconds=prefilled->audio_encode_cpu_seconds;
        prefill_thinker_seconds=prefilled->prefill_thinker_seconds;
        prefill_talker_seconds=prefilled->prefill_talker_seconds;
        prefill_seconds=prefill_thinker_seconds+prefill_talker_seconds;
        prefill_cpu_seconds=prefilled->prefill_cpu_seconds;
        bridge=malloc((size_t)hidden*sizeof(float));
        prefill_end=monotonic_seconds();prefill_cpu_end=process_cpu_seconds();
        printf("EVENT input_ready turn=%u input_frames=%zu prompt_tokens=%zu "
               "speech_start_to_ready_ms=%.0f\n",live_turn,audio_frames,
               prompt_count,(prefill_end-run_start)*1000.0);fflush(stdout);
    }else{
        const double audio_encode_start=monotonic_seconds();
        const double audio_encode_cpu_start=process_cpu_seconds();
        if(audio_path||provided_pcm){const int16_t *pcm=provided_pcm;size_t pcm_count=provided_pcm_count;if(audio_path&&load_wav(audio_path,&owned_pcm,&pcm_count)){fprintf(stderr,"audio input must be 16 kHz mono PCM WAV\n");return -1;}if(owned_pcm)pcm=owned_pcm;
            size_t frame_capacity=minimindo_audio_encoder_frames(pcm_count);audio_embeddings=malloc(frame_capacity*768*sizeof(float));
            if(!audio_encoder||!audio_embeddings||minimindo_audio_encoder_encode_pcm16(audio_encoder,pcm,pcm_count,audio_embeddings,frame_capacity*768,&audio_frames,error,sizeof(error))){fprintf(stderr,"audio encoder: %s\n",error);free(owned_pcm);return -1;}minimindo_audio_encoder_last_profile(audio_encoder,&audio_profile);free(owned_pcm);owned_pcm=NULL;audio_user=audio_prompt(audio_frames);prompt_text=audio_user;}
        const double audio_encode_end=monotonic_seconds();
        const double audio_encode_cpu_end=process_cpu_seconds();
        audio_encode_seconds=audio_encode_end-audio_encode_start;
        audio_encode_cpu_seconds=audio_encode_cpu_end-audio_encode_cpu_start;
        formatted=format_prompt(prompt_text);
        if(!formatted||encode(tokenizer,formatted,&prompt,&prompt_count,error,sizeof(error))!=0) { fprintf(stderr,"allocation/tokenizer: %s\n",error); return -1; }
        if(prompt_count+generation_capacity>resident_context){
            fprintf(stderr,"prompt plus text/audio drain exceeds context: %zu + %zu > %u\n",
                    prompt_count,generation_capacity,resident_context);return -1;}
        bridge=malloc(prompt_count*(size_t)hidden*sizeof(float));
        replacement_embeddings=calloc(prompt_count*(size_t)hidden,sizeof(float));
        replacement_mask=calloc(prompt_count,sizeof(uint8_t));
        if(!bridge||!replacement_embeddings||!replacement_mask){fprintf(stderr,"prefill allocation failed\n");return -1;}
        size_t audio_cursor=0;
        for(size_t p=0;p<prompt_count;++p)if(prompt[p]==16&&audio_cursor<audio_frames){
            memcpy(replacement_embeddings+p*(size_t)hidden,
                   audio_embeddings+audio_cursor++*(size_t)hidden,
                   hidden*sizeof(float));replacement_mask[p]=1;}
        if(audio_cursor!=audio_frames){fprintf(stderr,"audio token/embedding count mismatch: %zu/%zu\n",audio_cursor,audio_frames);return -1;}
        const double prefill_start=monotonic_seconds();
        const double prefill_cpu_start=process_cpu_seconds();
#if defined(MINIMINDO_SEQUENTIAL_PREFILL)
        for(size_t p=0;p<prompt_count;++p) {
        const int need_text_logits=p+1U==prompt_count;
        double stage_start=monotonic_seconds();
        int thinker_result;
        if(replacement_mask[p]){
            const float *embedding=replacement_embeddings+p*(size_t)hidden;
            thinker_result=need_text_logits?minimindo_thinker_forward_embedding(
                thinker,embedding,hidden,text_logits,tv,bridge,hidden,error,sizeof(error)):
                minimindo_thinker_prefill_embedding(thinker,embedding,hidden,
                                                     bridge,hidden,error,sizeof(error));}
        else thinker_result=need_text_logits?minimindo_thinker_forward_bridge(
            thinker,prompt[p],text_logits,tv,bridge,hidden,error,sizeof(error)):
            minimindo_thinker_prefill_bridge(thinker,prompt[p],bridge,hidden,
                                             error,sizeof(error));
        prefill_thinker_seconds+=monotonic_seconds()-stage_start;
        stage_start=monotonic_seconds();
        int talker_result=minimindo_talker_forward_masked(
            talker,bridge,hidden,current_audio,NULL,0,0,NULL,0,error,sizeof(error));
        prefill_talker_seconds+=monotonic_seconds()-stage_start;
        if(thinker_result||talker_result){fprintf(stderr,"prefill: %s\n",error);return -1;}
        }
#else
        double stage_start=monotonic_seconds();
        int thinker_result=minimindo_thinker_prefill_sequence(
            thinker,prompt,prompt_count,replacement_embeddings,replacement_mask,
            text_logits,tv,bridge,prompt_count*(size_t)hidden,error,sizeof(error));
        prefill_thinker_seconds=monotonic_seconds()-stage_start;
        if(thinker_result){fprintf(stderr,"prefill: %s\n",error);return -1;}
        stage_start=monotonic_seconds();
        int talker_result=minimindo_talker_prefill_sequence(
            talker,bridge,prompt_count*(size_t)hidden,current_audio,prompt_count,
            error,sizeof(error));
        prefill_talker_seconds=monotonic_seconds()-stage_start;
        if(talker_result) { fprintf(stderr,"prefill: %s\n",error);return -1; }
#endif
        prefill_end=monotonic_seconds();prefill_cpu_end=process_cpu_seconds();
        prefill_seconds=prefill_end-prefill_start;
        prefill_cpu_seconds=prefill_cpu_end-prefill_cpu_start;
    }
    if(!bridge||prompt_count+generation_capacity>resident_context){
        fprintf(stderr,"streamed prompt plus text/audio drain exceeds context: %zu + %zu > %u\n",
                prompt_count,generation_capacity,resident_context);return -1;
    }
    mimi_decode_worker decoder;
    int decoder_started = 0;
    alsa_playback_worker playback;
    pthread_t playback_thread;
    int playback_started = 0;
    if (mimi_decode_worker_start(&decoder, mimi, generation_capacity,
                                 live_turn, run_start) != 0) {
        fprintf(stderr, "Mimi stream worker: %s\n", decoder.error);
        return -1;
    }
    decoder_started = 1;
    if (playback_device != NULL) {
        memset(&playback, 0, sizeof(playback));
        playback.decoder = &decoder;
        playback.device = playback_device;
        playback.turn = live_turn;
        playback.inference_start = run_start;
        if (pthread_create(&playback_thread, NULL, alsa_playback_thread,
                           &playback) != 0) {
            fprintf(stderr, "ALSA playback worker start failed\n");
            minimindo_parallel_session_end();
            mimi_decode_worker_signal_done(&decoder);
            (void)mimi_decode_worker_finish(&decoder);
            free(decoder.audio);
            return -1;
        }
        playback_started = 1;
    }
    random_state=seed?seed:UINT64_C(1); size_t steps=0, frame_count=0;
    size_t text_steps=0;
    int text_finished=0, text_limit_hit=0, sentence_complete=0;
    int force_text_eos=0;
    int audio_drain_complete=0, first_finished=1;
    int audio_finished=0;
    double generate_thinker_seconds=0.0,generate_talker_seconds=0.0;
    /* Audio is externally observable every 80 ms.  A large Thinker batch here
     * stalls Talker for the whole batch even though it improves aggregate
     * throughput, which violates real streaming.  Advance exactly one bridge
     * per Talker step so codec codes keep arriving incrementally. */
    enum { THINKER_DRAIN_CHUNK = 1 };
    uint32_t drain_tokens[THINKER_DRAIN_CHUNK] = {0};
    float drain_bridges[THINKER_DRAIN_CHUNK * hidden];
    size_t drain_bridge_count=0,drain_bridge_index=0;
    int stop[8]; for(int i=0;i<8;++i)stop[i]=-1;
    while(steps<generation_capacity) {
        uint32_t text_token;
        if(text_finished) text_token=first_finished?201:0;
        else if(force_text_eos) text_token=2;
        else if(steps>=max_tokens) { text_token=2; text_limit_hit=1; }
        else text_token=sample_top_p(text_logits,tv,0.75f,0.90f,generated,steps,1.0f,work);
        first_finished=0; generated[steps]=text_token;
        if(!text_finished&&!force_text_eos&&text_token!=2&&steps>=2U&&
           generated_sentence_complete(tokenizer,generated,steps+1U,
                                       error,sizeof(error))){
            force_text_eos=1;sentence_complete=1;
        }
        const int audio_step=(int)steps-1;
        for(uint32_t c=0;c<8;++c) {
            uint32_t code=pad;
            if(audio_step>=(int)c) {
                uint32_t recent[3]; size_t recent_count=0;
                for(size_t back=1;back<=3&&back<=steps;++back) recent[recent_count++]=all_codes[(size_t)c*generation_capacity+steps-back];
                code=sample_top_k(audio_logits+(size_t)c*av,av,50,0.2f,recent,recent_count,1.05f,work);
                if(stop[c]<0&&code>=2048)stop[c]=(int)steps;
            }
            all_codes[(size_t)c*generation_capacity+steps]=code;
        }
        if(audio_step>=7) {
            int active=1; uint32_t frame[8];
            for(uint32_t c=0;c<8;++c) {
                size_t index=steps-7+c; frame[c]=all_codes[(size_t)c*generation_capacity+index];
                if(frame[c]>=2048||(stop[c]>=0&&(int)index>=stop[c]))active=0;
            }
            if(active) {
                for(uint32_t c=0;c<8;++c)
                    frames[(size_t)c*generation_capacity+frame_count]=frame[c];
                if (decoder_started && mimi_decode_worker_push(&decoder, frame) != 0) {
                    fprintf(stderr, "Mimi stream queue failed\n");
                    minimindo_parallel_session_end();
                    mimi_decode_worker_signal_done(&decoder);
                    if (playback_started)
                        (void)pthread_join(playback_thread, NULL);
                    (void)mimi_decode_worker_finish(&decoder);
                    free(decoder.audio);
                    return -1;
                }
                ++frame_count;
            } else audio_finished=1;
        }
        ++steps;
        if(audio_finished){audio_drain_complete=1;break;}
        int all_stopped=1;for(int c=0;c<8;++c)if(stop[c]<0)all_stopped=0;
        if(text_finished&&all_stopped){audio_drain_complete=1;break;}
        if(steps>=generation_capacity)break;
        for(uint32_t c=0;c<8;++c)current_audio[c]=pad;
        for(int c=0;c<8&&c<audio_step+1;++c)current_audio[c]=all_codes[(size_t)c*generation_capacity+steps-1];
        double stage_start=monotonic_seconds();
        int thinker_result=0;
        if(text_finished) {
            if(drain_bridge_index==drain_bridge_count) {
                drain_bridge_count=generation_capacity-steps;
                if(drain_bridge_count>THINKER_DRAIN_CHUNK)
                    drain_bridge_count=THINKER_DRAIN_CHUNK;
                drain_bridge_index=0;
                thinker_result=minimindo_thinker_prefill_sequence(
                    thinker,drain_tokens,drain_bridge_count,NULL,NULL,NULL,0,
                    drain_bridges,drain_bridge_count*(size_t)hidden,
                    error,sizeof(error));
                printf("STREAM stage=thinker_drain_prefill turn=%u "
                       "positions=%zu elapsed_ms=%.0f\n",
                       live_turn,drain_bridge_count,
                       (monotonic_seconds()-run_start)*1000.0);
                fflush(stdout);
            }
            if(!thinker_result) {
                memcpy(bridge,drain_bridges+drain_bridge_index*(size_t)hidden,
                       hidden*sizeof(float));
                ++drain_bridge_index;
            }
        } else {
            thinker_result=minimindo_thinker_forward_bridge(
                thinker,text_token,text_logits,tv,bridge,hidden,
                error,sizeof(error));
        }
        generate_thinker_seconds+=monotonic_seconds()-stage_start;
        uint32_t logits_mask=0;
        for(uint32_t c=0;c<8&&c<steps;++c)
            if(stop[c]<0)logits_mask|=UINT32_C(1)<<c;
        stage_start=monotonic_seconds();
        int talker_result=minimindo_talker_forward_masked(talker,bridge,hidden,current_audio,NULL,0,logits_mask,audio_logits,(size_t)8*av,error,sizeof(error));
        generate_talker_seconds+=monotonic_seconds()-stage_start;
        if(thinker_result||talker_result) {
            fprintf(stderr,"decode: %s\n",error);
            if (decoder_started) {
                minimindo_parallel_session_end();
                mimi_decode_worker_signal_done(&decoder);
                if (playback_started)
                    (void)pthread_join(playback_thread, NULL);
                (void)mimi_decode_worker_finish(&decoder);
                free(decoder.audio);
            }
            return -1;
        }
        if(!text_finished&&text_token==2){text_finished=1;text_steps=steps;}
    }
    const double generation_end = monotonic_seconds();
    const double generation_cpu_end = process_cpu_seconds();
    /* Release the fixed 0-2 compute group before the decoder expands from its
     * dedicated CPU3 lane to all four cores. */
    minimindo_parallel_session_end();
    if (decoder_started) mimi_decode_worker_signal_done(&decoder);
    if (playback_device != NULL) {
        printf("EVENT model_end turn=%u elapsed_ms=%.0f\n", live_turn,
               (generation_end - run_start) * 1000.0);
        fflush(stdout);
    }
    size_t text_count=steps; while(text_count&&generated[text_count-1]==0) --text_count;
    char *answer=decode_text(tokenizer,generated,text_count,error,sizeof(error));
    if(frame_count==0){
        fprintf(stderr,"Talker produced no complete Mimi frames\n");
        if (decoder_started) {
            if (playback_started)
                (void)pthread_join(playback_thread, NULL);
            (void)mimi_decode_worker_finish(&decoder);
            free(decoder.audio);
        }
        return -1;
    }
    uint32_t *mimi_codes = NULL;
    float *audio = NULL;
    size_t samples = 0;
    int playback_result = 0;
    double first_audio_ms = 0.0;
    size_t queue_waits = 0;
    double queue_wait_ms = 0.0;
    double mimi_end = 0.0;
    double mimi_cpu_end = generation_cpu_end;
    if (decoder_started) {
        if (playback_started) {
            (void)pthread_join(playback_thread, NULL);
            playback_result = playback.result;
            first_audio_ms = playback.first_audio_ms;
            queue_waits = playback.queue_waits;
            queue_wait_ms = playback.queue_wait_ms;
            printf("EVENT playback_end turn=%u result=%d "
                   "elapsed_ms=%.0f queue_waits=%zu queue_wait_ms=%.0f\n",
                   live_turn, playback_result,
                   (monotonic_seconds() - run_start) * 1000.0,
                   queue_waits, queue_wait_ms);
            fflush(stdout);
            hub_led_publish(playback_result == 0 ? HUB_LED_LISTENING :
                                                      HUB_LED_ERROR);
        }
        if (mimi_decode_worker_finish(&decoder) != 0) {
            fprintf(stderr, "Mimi stream decode: %s\n", decoder.error);
            free(decoder.audio);
            return -1;
        }
        audio = decoder.audio;
        samples = decoder.samples;
        mimi_end = decoder.decode_end;
        mimi_cpu_end = decoder.decode_cpu_end;
    } else {
        mimi_codes=malloc((size_t)8*frame_count*sizeof(uint32_t));
        if(!mimi_codes){fprintf(stderr,"Mimi code compaction allocation failed\n");return -1;}
        for(uint32_t c=0;c<8;++c)for(size_t t=0;t<frame_count;++t)
            mimi_codes[(size_t)c*frame_count+t]=frames[(size_t)c*generation_capacity+t];
        const size_t sample_capacity=minimindo_mimi_samples_for_frames(mimi,frame_count);
        audio=malloc(sample_capacity*sizeof(float));
        if(!audio||minimindo_mimi_decode(mimi,mimi_codes,frame_count,audio,sample_capacity,&samples,error,sizeof(error))){
            fprintf(stderr,"Mimi decode: %s\n",error);
            free(audio);
            free(mimi_codes);
            return -1;
        }
        mimi_end = monotonic_seconds();
        mimi_cpu_end = process_cpu_seconds();
    }
    if(write_wav(wav_path,audio,samples,minimindo_mimi_sample_rate(mimi))){
        fprintf(stderr,"Mimi/WAV write failed\n");
        free(audio);
        free(mimi_codes);
        return -1;
    }
    printf("{\"text\":\"");
    for(const unsigned char *p=(unsigned char *)(answer?answer:"");*p;++p){if(*p=='"'||*p=='\\')putchar('\\');if(*p=='\n'){fputs("\\n",stdout);continue;}putchar(*p);}
    printf("\",\"steps\":%zu,\"text_steps\":%zu,\"audio_frames\":%zu,\"input_frames\":%zu,"
           "\"prompt_tokens\":%zu,\"samples\":%zu,\"seed\":%llu,"
           "\"audio_encode_ms\":%.0f,\"audio_frontend_ms\":%.0f,"
           "\"audio_encoder_ms\":%.0f,\"audio_projector_ms\":%.0f,\"prefill_ms\":%.0f,"
           "\"prefill_thinker_ms\":%.0f,\"prefill_talker_ms\":%.0f,"
           "\"generate_ms\":%.0f,\"generate_thinker_ms\":%.0f,"
           "\"generate_talker_ms\":%.0f,\"mimi_drain_ms\":%.0f,"
           "\"audio_encode_cpu_ms\":%.0f,\"audio_encode_cpu_pct\":%.0f,"
           "\"prefill_cpu_ms\":%.0f,\"prefill_cpu_pct\":%.0f,"
           "\"generate_cpu_ms\":%.0f,\"generate_cpu_pct\":%.0f,"
           "\"mimi_cpu_ms\":%.0f,\"mimi_cpu_pct\":%.0f,"
           "\"model_ms\":%.0f,\"streaming\":%s,"
           "\"input_streaming\":%s,\"input_center_ms\":480,"
           "\"input_right_context_ms\":240,\"input_kv_cache_ms\":1920,"
           "\"decode_overlapped_with_generation\":%s,"
           "\"decode_overlap_frames\":%zu,"
           "\"decoder_to_alsa_streaming\":%s,"
           "\"playback_fully_buffered\":%s,"
           "\"end_to_end_streaming\":%s,\"producer_end_ms\":%.0f,"
           "\"first_audio_ms\":%.0f,"
           "\"streaming_lead_ms\":%.0f,\"queue_waits\":%zu,"
           "\"queue_wait_ms\":%.0f,\"text_limit_hit\":%s,"
           "\"sentence_complete\":%s,\"audio_drain_complete\":%s}\n",
           steps,text_steps?text_steps:steps,frame_count,audio_frames,prompt_count,samples,
           (unsigned long long)seed,
           audio_encode_seconds*1000,
           audio_profile.frontend_ms,audio_profile.encoder_ms,
           audio_profile.projector_ms,
           prefill_seconds*1000,
           prefill_thinker_seconds*1000,prefill_talker_seconds*1000,
           (generation_end-prefill_end)*1000,
           generate_thinker_seconds*1000,generate_talker_seconds*1000,
           (mimi_end-generation_end)*1000,
           audio_encode_cpu_seconds*1000,
           audio_encode_seconds>0.0?
               audio_encode_cpu_seconds*100.0/audio_encode_seconds:0.0,
           prefill_cpu_seconds*1000,
           prefill_seconds>0.0?
               prefill_cpu_seconds*100.0/prefill_seconds:0.0,
           (generation_cpu_end-prefill_cpu_end)*1000,
           (generation_cpu_end-prefill_cpu_end)*100.0/
               (generation_end-prefill_end),
           (mimi_cpu_end-generation_cpu_end)*1000,
           (mimi_cpu_end-generation_cpu_end)*100.0/
               (mimi_end-generation_end),
           (mimi_end-run_start)*1000,
           decoder.decode_overlap_frames?"true":"false",
           prefilled?"true":"false",
           decoder.decode_overlap_frames?"true":"false",
           decoder.decode_overlap_frames,
           playback_device!=NULL?"true":"false",
           "false",
           first_audio_ms>0.0&&first_audio_ms<
               (generation_end-run_start)*1000.0?"true":"false",
           (generation_end-run_start)*1000.0,first_audio_ms,
           first_audio_ms>0.0&&(mimi_end-run_start)*1000.0>first_audio_ms?
               (mimi_end-run_start)*1000.0-first_audio_ms:0.0,
           queue_waits,queue_wait_ms,
           text_limit_hit?"true":"false",
           sentence_complete?"true":"false",
           audio_drain_complete?"true":"false");
    free(audio);free(mimi_codes);free(answer);free(prompt);free(formatted);free(audio_user);free(audio_embeddings);free(replacement_embeddings);free(replacement_mask);free(text_logits);free(audio_logits);free(bridge);free(work);free(generated);free(all_codes);free(frames);
    return playback_result == 0 ? 0 : -1;
}

static int warm_resident(const char *thinker,const char *talker,const char *tokenizer,
                         const char *mimi,const char *audio_encoder)
{
    char error[256]={0};if(ensure_resident(thinker,talker,tokenizer,mimi,audio_encoder,error,sizeof(error))){fprintf(stderr,"warm-up open: %s\n",error);return -1;}
    uint32_t tv=minimindo_thinker_vocab_size(resident_thinker),av=minimindo_talker_vocab_size(resident_talker),h=minimindo_thinker_hidden_size(resident_thinker);
    float *tl=malloc((size_t)tv*4),*al=malloc((size_t)8*av*4),*bridge=malloc((size_t)h*4);uint32_t ids[8];for(int i=0;i<8;++i)ids[i]=minimindo_talker_pad_token(resident_talker);
    if(!tl||!al||!bridge||minimindo_thinker_forward_bridge(resident_thinker,1,tl,tv,bridge,h,error,sizeof(error))||minimindo_talker_forward(resident_talker,bridge,h,ids,NULL,0,al,(size_t)8*av,error,sizeof(error)))return -1;
    if(audio_encoder){int16_t silence[800]={0};float embedding[768*2];size_t frames=0;if(minimindo_audio_encoder_encode_pcm16(resident_audio_encoder,silence,800,embedding,768*2,&frames,error,sizeof(error)))return -1;}
    uint32_t codes[8]={0};float audio[1920];size_t samples=0;if(minimindo_mimi_decode(resident_mimi,codes,1,audio,1920,&samples,error,sizeof(error)))return -1;
    free(tl);free(al);free(bridge);minimindo_thinker_reset(resident_thinker);minimindo_talker_reset(resident_talker);return 0;
}

static double monotonic_seconds(void)
{struct timespec value;clock_gettime(CLOCK_MONOTONIC,&value);return value.tv_sec+value.tv_nsec*1e-9;}

static double process_cpu_seconds(void)
{struct timespec value;clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&value);return value.tv_sec+value.tv_nsec*1e-9;}

static int safe_alsa_name(const char *name)
{if(!name||!*name)return 0;for(const unsigned char *p=(const unsigned char *)name;*p;++p)if(!((*p>='a'&&*p<='z')||(*p>='A'&&*p<='Z')||(*p>='0'&&*p<='9')||strchr(":,._-",*p)))return 0;return 1;}

static FILE *open_capture(const char *device)
{
    char command[256];if(!safe_alsa_name(device))return NULL;
    snprintf(command,sizeof(command),"arecord -q -D %s -t raw -f S16_LE -c 1 -r 16000",device);
    return popen(command,"r");
}

enum { CAPTURE_CHUNK=512, CAPTURE_QUEUE=64 };
typedef struct {
    FILE *stream;
    pthread_t thread;
    /* The mutex is only the empty-queue sleep boundary.  ALSA capture is the
     * sole chunk writer and VAD is the sole reader. */
    pthread_mutex_t wait_mutex;
    pthread_cond_t ready;
    int16_t chunks[CAPTURE_QUEUE][CAPTURE_CHUNK];
    atomic_size_t read_sequence;
    atomic_size_t write_sequence;
    atomic_int stopped;
    atomic_int failed;
} capture_queue;

static void *capture_worker(void *opaque)
{
    capture_queue *queue = opaque;
    int16_t chunk[CAPTURE_CHUNK];
    while (!atomic_load_explicit(&queue->stopped, memory_order_acquire)) {
        const size_t got = fread(chunk, sizeof(int16_t), CAPTURE_CHUNK,
                                 queue->stream);
        if (got != CAPTURE_CHUNK) {
            if (feof(queue->stream)) { clearerr(queue->stream); continue; }
            atomic_store_explicit(&queue->failed, 1, memory_order_release);
            pthread_mutex_lock(&queue->wait_mutex);
            pthread_cond_signal(&queue->ready);
            pthread_mutex_unlock(&queue->wait_mutex);
            break;
        }
        const size_t write = atomic_load_explicit(
            &queue->write_sequence, memory_order_relaxed);
        const size_t read = atomic_load_explicit(
            &queue->read_sequence, memory_order_acquire);
        /* During inference capture intentionally stays live.  If its 2 s ring
         * is full, discard new echo; live() flushes the ring before listening
         * again.  The producer never mutates the consumer's read cursor. */
        if (write - read >= CAPTURE_QUEUE) continue;
        memcpy(queue->chunks[write % CAPTURE_QUEUE], chunk, sizeof(chunk));
        atomic_store_explicit(&queue->write_sequence, write + 1U,
                              memory_order_release);
        pthread_mutex_lock(&queue->wait_mutex);
        pthread_cond_signal(&queue->ready);
        pthread_mutex_unlock(&queue->wait_mutex);
    }
    return NULL;
}

static int capture_start(capture_queue *queue,const char *device)
{
    memset(queue, 0, sizeof(*queue));
    atomic_init(&queue->read_sequence, 0U);
    atomic_init(&queue->write_sequence, 0U);
    atomic_init(&queue->stopped, 0);
    atomic_init(&queue->failed, 0);
    queue->stream = open_capture(device);
    if (!queue->stream) return -1;
    if (pthread_mutex_init(&queue->wait_mutex, NULL) != 0) {
        pclose(queue->stream);
        return -1;
    }
    if (pthread_cond_init(&queue->ready, NULL) != 0) {
        pthread_mutex_destroy(&queue->wait_mutex);
        pclose(queue->stream);
        return -1;
    }
    if (pthread_create(&queue->thread, NULL, capture_worker, queue) != 0) {
        pthread_cond_destroy(&queue->ready);
        pthread_mutex_destroy(&queue->wait_mutex);
        pclose(queue->stream);
        return -1;
    }
    return 0;
}

static int capture_next(capture_queue *queue,int16_t output[CAPTURE_CHUNK])
{
    pthread_mutex_lock(&queue->wait_mutex);
    while (atomic_load_explicit(&queue->read_sequence,
                                memory_order_relaxed) ==
               atomic_load_explicit(&queue->write_sequence,
                                    memory_order_acquire) &&
           !atomic_load_explicit(&queue->failed, memory_order_acquire))
        pthread_cond_wait(&queue->ready, &queue->wait_mutex);
    const int failed = atomic_load_explicit(
        &queue->failed, memory_order_acquire);
    pthread_mutex_unlock(&queue->wait_mutex);
    if (failed) return -1;
    const size_t read = atomic_load_explicit(
        &queue->read_sequence, memory_order_relaxed);
    memcpy(output, queue->chunks[read % CAPTURE_QUEUE],
           CAPTURE_CHUNK * sizeof(int16_t));
    atomic_store_explicit(&queue->read_sequence, read + 1U,
                          memory_order_release);
    return 0;
}

static void capture_flush(capture_queue *queue)
{
    const size_t write = atomic_load_explicit(
        &queue->write_sequence, memory_order_acquire);
    atomic_store_explicit(&queue->read_sequence, write, memory_order_release);
}

static void capture_stop(capture_queue *queue)
{
    atomic_store_explicit(&queue->stopped, 1, memory_order_release);
    pclose(queue->stream);
    pthread_join(queue->thread,NULL);
    pthread_cond_destroy(&queue->ready);
    pthread_mutex_destroy(&queue->wait_mutex);
}

enum { INPUT_PCM_QUEUE=512, INPUT_EMBEDDING_FRAMES=16 };

typedef struct {
    int16_t samples[CAPTURE_CHUNK];
    size_t count;
    int end_of_stream;
} input_pcm_item;

typedef struct {
    pthread_t thread;
    pthread_mutex_t wait_mutex;
    pthread_cond_t ready;
    input_pcm_item items[INPUT_PCM_QUEUE];
    atomic_size_t read_sequence;
    atomic_size_t write_sequence;
    atomic_int abort_requested;
    int started;
    int result;
    unsigned turn;
    prefilled_audio_input prefilled;
    char error[256];
} live_input_stream;

static int live_input_enqueue(live_input_stream *worker,const int16_t *samples,
                              size_t count,int end_of_stream)
{
    if(!worker||!worker->started||count>CAPTURE_CHUNK||(!samples&&count))
        return -1;
    const size_t write=atomic_load_explicit(&worker->write_sequence,
                                            memory_order_relaxed);
    const size_t read=atomic_load_explicit(&worker->read_sequence,
                                           memory_order_acquire);
    if(write-read>=INPUT_PCM_QUEUE)return -1;
    input_pcm_item *item=&worker->items[write%INPUT_PCM_QUEUE];
    if(count)memcpy(item->samples,samples,count*sizeof(int16_t));
    item->count=count;item->end_of_stream=end_of_stream;
    atomic_store_explicit(&worker->write_sequence,write+1U,
                          memory_order_release);
    /* The ring itself is lock-free SPSC.  This mutex exists only to close the
     * enqueue/dequeue sleep race around the condition variable. */
    pthread_mutex_lock(&worker->wait_mutex);
    pthread_cond_signal(&worker->ready);
    pthread_mutex_unlock(&worker->wait_mutex);
    return 0;
}

static int live_input_next(live_input_stream *worker,input_pcm_item *item)
{
    pthread_mutex_lock(&worker->wait_mutex);
    while(atomic_load_explicit(&worker->read_sequence,memory_order_relaxed)==
          atomic_load_explicit(&worker->write_sequence,memory_order_acquire))
        pthread_cond_wait(&worker->ready,&worker->wait_mutex);
    pthread_mutex_unlock(&worker->wait_mutex);
    const size_t read=atomic_load_explicit(&worker->read_sequence,
                                           memory_order_relaxed);
    *item=worker->items[read%INPUT_PCM_QUEUE];
    atomic_store_explicit(&worker->read_sequence,read+1U,memory_order_release);
    return 0;
}

static int live_input_prefill(live_input_stream *worker,const uint32_t *tokens,
                              size_t count,const float *embeddings,
                              int final_logits)
{
    if(count==0)return 0;
    const uint32_t hidden=minimindo_thinker_hidden_size(resident_thinker);
    const uint32_t vocab=minimindo_thinker_vocab_size(resident_thinker);
    const uint32_t pad=minimindo_talker_pad_token(resident_talker);
    float *bridges=malloc(count*(size_t)hidden*sizeof(float));
    uint8_t *mask=embeddings?malloc(count):NULL;
    if(!bridges||(embeddings&&!mask)){free(bridges);free(mask);return -1;}
    if(mask)memset(mask,1,count);
    uint32_t audio_ids[8];for(size_t i=0;i<8;++i)audio_ids[i]=pad;
    double cpu_start=process_cpu_seconds();
    double stage=monotonic_seconds();
    int thinker_result=minimindo_thinker_prefill_sequence(
        resident_thinker,tokens,count,embeddings,mask,
        final_logits?worker->prefilled.text_logits:NULL,
        final_logits?vocab:0,bridges,count*(size_t)hidden,
        worker->error,sizeof(worker->error));
    worker->prefilled.prefill_thinker_seconds+=monotonic_seconds()-stage;
    stage=monotonic_seconds();
    int talker_result=thinker_result?-1:minimindo_talker_prefill_sequence(
        resident_talker,bridges,count*(size_t)hidden,audio_ids,count,
        worker->error,sizeof(worker->error));
    worker->prefilled.prefill_talker_seconds+=monotonic_seconds()-stage;
    worker->prefilled.prefill_cpu_seconds+=process_cpu_seconds()-cpu_start;
    free(bridges);free(mask);
    return thinker_result||talker_result?-1:0;
}

static int live_input_verify_tokens(live_input_stream *worker,
                                    const uint32_t *prefix,size_t prefix_count,
                                    const uint32_t *suffix,size_t suffix_count)
{
    char *audio=audio_prompt(worker->prefilled.audio_frames);
    char *formatted=audio?format_prompt(audio):NULL;
    uint32_t *whole=NULL;size_t whole_count=0;
    int result=-1;
    if(formatted&&encode(resident_tokenizer,formatted,&whole,&whole_count,
                         worker->error,sizeof(worker->error))==0&&
       whole_count==prefix_count+worker->prefilled.audio_frames+suffix_count){
        result=0;
        for(size_t i=0;i<prefix_count;++i)if(whole[i]!=prefix[i])result=-1;
        for(size_t i=0;i<worker->prefilled.audio_frames;++i)
            if(whole[prefix_count+i]!=16U)result=-1;
        for(size_t i=0;i<suffix_count;++i)
            if(whole[prefix_count+worker->prefilled.audio_frames+i]!=suffix[i])
                result=-1;
    }
    if(result)snprintf(worker->error,sizeof(worker->error),
                       "incremental tokenizer boundary mismatch");
    free(whole);free(formatted);free(audio);return result;
}

static void *live_input_thread(void *opaque)
{
    live_input_stream *worker=opaque;
    uint32_t *prefix=NULL,*suffix=NULL;size_t prefix_count=0,suffix_count=0;
    minimindo_audio_encoder_stream *encoder=NULL;
    float embeddings[INPUT_EMBEDDING_FRAMES*768];
    worker->result=-1;
    (void)minimindo_parallel_pin_current(0U);
    if(minimindo_parallel_session_begin(4U)!=0){
        snprintf(worker->error,sizeof(worker->error),
                 "input compute session failed");
        goto done;
    }
    minimindo_thinker_reset(resident_thinker);
    minimindo_talker_reset(resident_talker);
    worker->prefilled.text_logits=malloc(
        (size_t)minimindo_thinker_vocab_size(resident_thinker)*sizeof(float));
    encoder=minimindo_audio_encoder_stream_open(resident_audio_encoder,
                                                 worker->error,
                                                 sizeof(worker->error));
    if(!worker->prefilled.text_logits||!encoder||
       encode(resident_tokenizer,prompt_prefix,&prefix,&prefix_count,
              worker->error,sizeof(worker->error))||
       encode(resident_tokenizer,prompt_suffix,&suffix,&suffix_count,
              worker->error,sizeof(worker->error))||
       live_input_prefill(worker,prefix,prefix_count,NULL,0))goto done;
    printf("EVENT input_stream_start turn=%u prefix_tokens=%zu\n",
           worker->turn,prefix_count);fflush(stdout);
    for(;;){
        input_pcm_item item;
        live_input_next(worker,&item);
        if(atomic_load_explicit(&worker->abort_requested,memory_order_acquire)){
            snprintf(worker->error,sizeof(worker->error),"input stream aborted");
            goto done;
        }
        size_t frames=0;
        double cpu_start=process_cpu_seconds();
        double stage=monotonic_seconds();
        int encode_result=minimindo_audio_encoder_stream_push_pcm16(
            encoder,item.count?item.samples:NULL,item.count,item.end_of_stream,
            embeddings,sizeof(embeddings)/sizeof(embeddings[0]),&frames,
            worker->error,sizeof(worker->error));
        worker->prefilled.audio_encode_seconds+=monotonic_seconds()-stage;
        worker->prefilled.audio_encode_cpu_seconds+=
            process_cpu_seconds()-cpu_start;
        if(encode_result)goto done;
        if(frames){
            uint32_t audio_tokens[INPUT_EMBEDDING_FRAMES];
            for(size_t i=0;i<frames;++i)audio_tokens[i]=16U;
            worker->prefilled.audio_frames+=frames;
            printf("STREAM stage=input_audio_commit turn=%u frames=%zu "
                   "total_frames=%zu pcm_ms=%zu elapsed_ms=%.0f\n",
                   worker->turn,frames,worker->prefilled.audio_frames,
                   minimindo_audio_encoder_stream_total_frames(encoder)*60U,
                   (monotonic_seconds()-worker->prefilled.run_start)*1000.0);
            fflush(stdout);
            if(live_input_prefill(worker,audio_tokens,frames,embeddings,0))
                goto done;
            printf("STREAM stage=input_prefill turn=%u positions=%zu "
                   "thinker_position=%u talker_position=%u elapsed_ms=%.0f\n",
                   worker->turn,frames,
                   minimindo_thinker_position(resident_thinker),
                   minimindo_talker_position(resident_talker),
                   (monotonic_seconds()-worker->prefilled.run_start)*1000.0);
            fflush(stdout);
        }
        if(item.end_of_stream){
            if(live_input_verify_tokens(worker,prefix,prefix_count,suffix,
                                        suffix_count)||
               live_input_prefill(worker,suffix,suffix_count,NULL,1))goto done;
            worker->prefilled.prompt_count=prefix_count+
                worker->prefilled.audio_frames+suffix_count;
            minimindo_audio_encoder_stream_profile(
                encoder,&worker->prefilled.audio_profile);
            printf("EVENT input_stream_eos turn=%u input_frames=%zu "
                   "prompt_tokens=%zu thinker_position=%u talker_position=%u "
                   "elapsed_ms=%.0f\n",worker->turn,
                   worker->prefilled.audio_frames,worker->prefilled.prompt_count,
                   minimindo_thinker_position(resident_thinker),
                   minimindo_talker_position(resident_talker),
                   (monotonic_seconds()-worker->prefilled.run_start)*1000.0);
            fflush(stdout);worker->result=0;break;
        }
    }
done:
    minimindo_audio_encoder_stream_close(encoder);
    free(prefix);free(suffix);
    minimindo_parallel_session_end();
    if(worker->result)
        fprintf(stderr,"input stream turn=%u failed: %s\n",worker->turn,
                worker->error[0]?worker->error:"unknown error");
    return NULL;
}

static int live_input_finish(live_input_stream *worker,int abort_stream);

static int live_input_start(live_input_stream *worker,unsigned turn,
                            double run_start,const int16_t *preroll,
                            size_t preroll_count)
{
    memset(worker,0,sizeof(*worker));worker->turn=turn;
    worker->prefilled.run_start=run_start;
    atomic_init(&worker->read_sequence,0U);
    atomic_init(&worker->write_sequence,0U);
    atomic_init(&worker->abort_requested,0);
    if(pthread_mutex_init(&worker->wait_mutex,NULL))return -1;
    if(pthread_cond_init(&worker->ready,NULL)){
        pthread_mutex_destroy(&worker->wait_mutex);return -1;
    }
    if(pthread_create(&worker->thread,NULL,live_input_thread,worker)!=0){
        pthread_cond_destroy(&worker->ready);
        pthread_mutex_destroy(&worker->wait_mutex);return -1;
    }
    worker->started=1;
    for(size_t offset=0;offset<preroll_count;offset+=CAPTURE_CHUNK){
        size_t count=preroll_count-offset;
        if(count>CAPTURE_CHUNK)count=CAPTURE_CHUNK;
        if(live_input_enqueue(worker,preroll+offset,count,0)){
            (void)live_input_finish(worker,1);return -1;
        }
    }
    return 0;
}

static int live_input_finish(live_input_stream *worker,int abort_stream)
{
    if(!worker||!worker->started)return -1;
    if(abort_stream)
        atomic_store_explicit(&worker->abort_requested,1,memory_order_release);
    if(live_input_enqueue(worker,NULL,0,1))return -1;
    pthread_join(worker->thread,NULL);worker->started=0;
    pthread_cond_destroy(&worker->ready);
    pthread_mutex_destroy(&worker->wait_mutex);
    return worker->result;
}

static int run_audio_file_streaming(
    const char *thinker,const char *talker,const char *tokenizer,
    const char *mimi,const char *audio_encoder,const char *audio_path,
    const char *output,const char *playback,uint32_t max_tokens,uint64_t seed)
{
    char error[256]={0};int16_t *samples=NULL;size_t sample_count=0;
    if(ensure_resident(thinker,talker,tokenizer,mimi,audio_encoder,
                       error,sizeof(error))||
       load_wav(audio_path,&samples,&sample_count)){
        fprintf(stderr,"streaming audio file setup failed: %s\n",
                error[0]?error:"input must be 16 kHz mono PCM WAV");
        free(samples);return -1;
    }
    live_input_stream input;
    const double started=monotonic_seconds();
    if(live_input_start(&input,0,started,NULL,0)){
        free(samples);return -1;
    }
    int queue_result=0;
    for(size_t offset=0;offset<sample_count;offset+=CAPTURE_CHUNK){
        size_t count=sample_count-offset;if(count>CAPTURE_CHUNK)count=CAPTURE_CHUNK;
        if(live_input_enqueue(&input,samples+offset,count,0)){
            queue_result=-1;break;
        }
    }
    free(samples);
    if(queue_result){
        (void)live_input_finish(&input,1);
        free(input.prefilled.text_logits);return -1;
    }
    if(live_input_finish(&input,0)){
        free(input.prefilled.text_logits);return -1;
    }
    (void)minimindo_parallel_pin_current(0U);
    (void)minimindo_parallel_session_begin(3U);
    int result=run(thinker,talker,tokenizer,mimi,NULL,output,audio_encoder,
                   NULL,NULL,0,max_tokens,seed,playback,0,&input.prefilled);
    minimindo_parallel_session_end();
    free(input.prefilled.text_logits);return result;
}

static int live(const char *thinker,const char *talker,const char *tokenizer,
                const char *mimi,const char *audio_encoder,const char *capture_device,
                const char *playback_device,uint32_t max_tokens,uint64_t seed)
{
    if(!safe_alsa_name(capture_device)||!safe_alsa_name(playback_device))return -1;
    if(hub_led_start()!=0)
        fprintf(stderr,"status LED worker unavailable; voice path continues\n");
    if(minimindo_volume_monitor_start()!=0)
        fprintf(stderr,"volume event monitor unavailable; voice path continues\n");
    hub_led_publish(HUB_LED_THINKING);
    const double warm_start=monotonic_seconds();
    (void)minimindo_parallel_pin_current(0U);
    (void)minimindo_parallel_session_begin(4U);
    const int warm_result=warm_resident(thinker,talker,tokenizer,mimi,audio_encoder);
    minimindo_parallel_session_end();
    if(warm_result){minimindo_volume_monitor_stop();hub_led_stop(HUB_LED_STOP_ERROR);return -1;}
    printf("READY pipeline=MiniMind-O-native-C input_streaming=always "
           "input_center_ms=480 input_right_context_ms=240 "
           "input_kv_cache_ms=1920 warmup_ms=%.0f capture=%s playback=%s\n",
           (monotonic_seconds()-warm_start)*1000,capture_device,
           playback_device);fflush(stdout);
    hub_led_publish(HUB_LED_LISTENING);
    capture_queue capture;if(capture_start(&capture,capture_device)){perror("arecord");minimindo_volume_monitor_stop();hub_led_stop(HUB_LED_STOP_ERROR);return -1;}
    enum{CHUNK=CAPTURE_CHUNK,PREROLL=8192,MAX_SPEECH=3*16000};
    int16_t chunk[CHUNK],ring[PREROLL],pending_silence[20][CHUNK];
    int16_t *speech=malloc(MAX_SPEECH*sizeof(int16_t));
    live_input_stream input={0};
    size_t ring_write=0,ring_count=0,speech_count=0,pending_count=0;
    double noise=120.0,last_monitor=0;
    int speaking=0,hot=0,silent=0,active_chunks=0,cooldown=0;
    unsigned turn=0;
    if(!speech){capture_stop(&capture);minimindo_volume_monitor_stop();hub_led_stop(HUB_LED_STOP_ERROR);return -1;}
    while(1){if(capture_next(&capture,chunk))break;size_t got=CHUNK;
        double squares=0,peak=0;for(size_t i=0;i<got;++i){double v=chunk[i];squares+=v*v;if(fabs(v)>peak)peak=fabs(v);}double rms=sqrt(squares/got);
        double threshold=fmax(280.0,noise*3.2);double now=monotonic_seconds();
        const double monitor_interval=speaking?0.5:5.0;
        if(now-last_monitor>=monitor_interval){printf("%s level_rms=%.0f peak=%.0f noise=%.0f threshold=%.0f turn=%u\n",speaking?"SPEECH":"LISTEN",rms,peak,noise,threshold,turn);fflush(stdout);last_monitor=now;}
        if(cooldown){--cooldown;if(rms<threshold)noise=noise*.98+rms*.02;continue;}
        if(!speaking){if(rms<threshold)noise=noise*.995+rms*.005;
            for(size_t i=0;i<got;++i){ring[ring_write]=chunk[i];ring_write=(ring_write+1)%PREROLL;if(ring_count<PREROLL)++ring_count;}
            hot=rms>threshold?hot+1:0;if(hot>=2){speaking=1;silent=0;active_chunks=hot;pending_count=0;speech_count=0;size_t start=(ring_write+PREROLL-ring_count)%PREROLL;
                for(size_t i=0;i<ring_count&&speech_count<MAX_SPEECH;++i)speech[speech_count++]=ring[(start+i)%PREROLL];
                const double speech_start=monotonic_seconds();
                printf("EVENT speech_start turn=%u preroll_ms=%zu\n",turn+1,ring_count*1000/16000);fflush(stdout);
                hub_led_publish(HUB_LED_LISTENING);
                if(live_input_start(&input,turn+1,speech_start,speech,speech_count)){
                    fprintf(stderr,"input streaming worker start failed\n");break;}}}
        else{const int hit_limit=speech_count+got>MAX_SPEECH;
            if(!hit_limit){memcpy(speech+speech_count,chunk,got*sizeof(int16_t));speech_count+=got;}
            if(rms>threshold)++active_chunks;
            if(!hit_limit&&rms<threshold*.65){
                if(pending_count<20){
                    memcpy(pending_silence[pending_count],chunk,sizeof(chunk));
                    ++pending_count;
                }
                silent=(int)pending_count;
            }else if(!hit_limit){
                int enqueue_failed=0;
                for(size_t p=0;p<pending_count;++p)
                    if(live_input_enqueue(&input,pending_silence[p],CHUNK,0))
                        enqueue_failed=1;
                pending_count=0;silent=0;
                if(live_input_enqueue(&input,chunk,got,0))enqueue_failed=1;
                if(enqueue_failed){
                    fprintf(stderr,"input PCM queue overflow\n");
                    (void)live_input_finish(&input,1);break;
                }
            }
            if(silent>=20||hit_limit){speaking=0;hot=0;ring_count=0;++turn;size_t trim=(size_t)silent*CHUNK;if(trim<speech_count)speech_count-=trim;
                pending_count=0;
                if(speech_count<4000||active_chunks<3){
                    printf("EVENT empty_vad turn=%u samples=%zu active_chunks=%d action=discard\n",turn,speech_count,active_chunks);fflush(stdout);
                    (void)live_input_finish(&input,1);
                    free(input.prefilled.text_logits);input.prefilled.text_logits=NULL;
                    hub_led_publish(HUB_LED_LISTENING);
                    speech_count=0;continue;}
                printf("EVENT speech_end turn=%u samples=%zu duration_ms=%zu end=%s\n",turn,speech_count,speech_count*1000/16000,hit_limit?"limit":"silence");fflush(stdout);
                hub_led_publish(HUB_LED_THINKING);
                const double inference_start=monotonic_seconds();const char *response="/dev/shm/minimindo-live-response.wav";
                if(live_input_finish(&input,0)){
                    fprintf(stderr,"input streaming turn=%u did not finish\n",turn);
                    free(input.prefilled.text_logits);input.prefilled.text_logits=NULL;
                    break;
                }
                printf("EVENT input_caught_up turn=%u speech_end_to_ready_ms=%.0f\n",
                       turn,(monotonic_seconds()-inference_start)*1000.0);fflush(stdout);
                (void)minimindo_parallel_pin_current(0U);
                (void)minimindo_parallel_session_begin(3U);
                int result=run(thinker,talker,tokenizer,mimi,NULL,response,
                               audio_encoder,NULL,NULL,0,max_tokens,seed+turn,
                               playback_device,turn,&input.prefilled);
                minimindo_parallel_session_end();
                free(input.prefilled.text_logits);input.prefilled.text_logits=NULL;
                if(result)hub_led_publish(HUB_LED_ERROR);
                printf("EVENT inference_end turn=%u elapsed_ms=%.0f result=%d\n",turn,(monotonic_seconds()-inference_start)*1000,result);fflush(stdout);speech_count=0;
                capture_flush(&capture);cooldown=32;
            }}
    }
    if(input.started)(void)live_input_finish(&input,1);
    free(input.prefilled.text_logits);
    free(speech);capture_stop(&capture);minimindo_volume_monitor_stop();hub_led_stop(HUB_LED_STOP_ERROR);return -1;
}

int main(int argc,char **argv)
{
    (void)signal(SIGPIPE, SIG_IGN);
    if(argc<8){fprintf(stderr,"usage: %s THINKER TALKER TOKENIZER MIMI (--prompt TEXT | --audio INPUT.wav --audio-encoder ENCODER.mmo) --output FILE.wav [--playback-device ALSA] [--max-tokens N] [--seed N]\n",argv[0]);return 2;}
    const char *prompt=NULL,*output=NULL,*audio_path=NULL,*audio_encoder=NULL,*capture=NULL,*playback=NULL;int live_mode=0;uint32_t max_tokens=96;uint64_t seed=1;
    for(int i=5;i<argc;++i){if(!strcmp(argv[i],"--prompt")&&i+1<argc)prompt=argv[++i];else if(!strcmp(argv[i],"--audio")&&i+1<argc)audio_path=argv[++i];else if(!strcmp(argv[i],"--audio-encoder")&&i+1<argc)audio_encoder=argv[++i];else if(!strcmp(argv[i],"--output")&&i+1<argc)output=argv[++i];else if(!strcmp(argv[i],"--live"))live_mode=1;else if(!strcmp(argv[i],"--capture-device")&&i+1<argc)capture=argv[++i];else if(!strcmp(argv[i],"--playback-device")&&i+1<argc)playback=argv[++i];else if(!strcmp(argv[i],"--max-tokens")&&i+1<argc)max_tokens=(uint32_t)strtoul(argv[++i],NULL,10);else if(!strcmp(argv[i],"--seed")&&i+1<argc)seed=strtoull(argv[++i],NULL,10);else return 2;}
    if(max_tokens<16||max_tokens>resident_context-192U)return 2;
    if(live_mode)return live(argv[1],argv[2],argv[3],argv[4],audio_encoder,capture?capture:"plughw:1,0",playback?playback:"plughw:0,0",max_tokens,seed)!=0;
    if((!prompt&&!audio_path)||(prompt&&audio_path)||!output)return 2;
    if (playback != NULL && !safe_alsa_name(playback)) return 2;
    int result;
    if(audio_path)result=run_audio_file_streaming(
        argv[1],argv[2],argv[3],argv[4],audio_encoder,audio_path,output,
        playback,max_tokens,seed);
    else{
        (void)minimindo_parallel_pin_current(0U);
        (void)minimindo_parallel_session_begin(3U);
        result=run(argv[1],argv[2],argv[3],argv[4],prompt,output,
                   audio_encoder,NULL,NULL,0,max_tokens,seed,playback,0,NULL);
        minimindo_parallel_session_end();
    }
    return result!=0;
}
