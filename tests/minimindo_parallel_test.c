#include "minimindo_parallel.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>

enum { VALUE_COUNT = 4096, PASSES = 2000 };

typedef struct {
    unsigned *values;
    unsigned pass;
} fill_context;

static void fill_range(void *opaque, size_t begin, size_t end)
{
    fill_context *context = opaque;
    for (size_t index = begin; index < end; ++index)
        context->values[index] = context->pass + (unsigned)index;
}

typedef struct {
    atomic_int *start;
    unsigned values[VALUE_COUNT];
    int failed;
} concurrent_context;

static void *concurrent_dispatcher(void *opaque)
{
    concurrent_context *thread = opaque;
    minimindo_parallel_set_threads(3U);
    while (!atomic_load_explicit(thread->start, memory_order_acquire)) {
    }
    for (unsigned pass = 1U; pass <= PASSES; ++pass) {
        fill_context context = {thread->values, pass + 10000U};
        minimindo_parallel_for(VALUE_COUNT, fill_range, &context);
        for (size_t index = 0; index < VALUE_COUNT; ++index) {
            if (thread->values[index] !=
                pass + 10000U + (unsigned)index) {
                thread->failed = 1;
                return NULL;
            }
        }
    }
    return NULL;
}

int main(void)
{
    unsigned values[VALUE_COUNT] = {0};
    if (minimindo_parallel_session_begin(4U) != 0) {
        fputs("parallel pool did not start\n", stderr);
        return 1;
    }
    for (unsigned pass = 1U; pass <= PASSES; ++pass) {
        fill_context context = {values, pass};
        minimindo_parallel_for(VALUE_COUNT, fill_range, &context);
        for (size_t index = 0; index < VALUE_COUNT; ++index) {
            if (values[index] != pass + (unsigned)index) {
                fprintf(stderr, "bad value at pass=%u index=%zu\n",
                        pass, index);
                return 1;
            }
        }
    }

    minimindo_parallel_session_end();
    if (minimindo_parallel_session_begin(3U) != 0) return 1;
    atomic_int concurrent_start;
    atomic_init(&concurrent_start, 0);
    concurrent_context concurrent = {&concurrent_start,{0},0};
    pthread_t dispatcher;
    if (pthread_create(&dispatcher,NULL,concurrent_dispatcher,
                       &concurrent) != 0)
        return 1;
    minimindo_parallel_set_threads(3U);
    atomic_store_explicit(&concurrent_start,1,memory_order_release);
    for (unsigned pass = 1U; pass <= PASSES; ++pass) {
        fill_context context = {values, pass + 20000U};
        minimindo_parallel_for(VALUE_COUNT, fill_range, &context);
        for (size_t index = 0; index < VALUE_COUNT; ++index) {
            if (values[index] != pass + 20000U + (unsigned)index)
                return 1;
        }
    }
    if (pthread_join(dispatcher,NULL) != 0 || concurrent.failed)
        return 1;

    minimindo_parallel_session_end();
    /* Exercise the main->Mimi ownership handoff and a futex wake after idle. */
    for (unsigned transition = 0; transition < 100U; ++transition) {
        if (minimindo_parallel_session_begin(4U) != 0) return 1;
        fill_context context = {values, PASSES + transition + 1U};
        minimindo_parallel_for(VALUE_COUNT, fill_range, &context);
        minimindo_parallel_session_end();
    }
    puts("minimindo parallel test passed");
    return 0;
}
