#ifndef LLM_IN_C_MINIMINDO_PARALLEL_H
#define LLM_IN_C_MINIMINDO_PARALLEL_H

#include <stddef.h>

typedef void (*minimindo_parallel_task)(void *context, size_t begin,
                                        size_t end);

/*
 * A process-wide, persistent three-worker pool. The calling thread is lane 0;
 * workers 1..3 are pinned to lanes 1..3 on Linux. A lane is CPU
 * MINIMINDO_CPU_BASE + lane (base 0 by default; RK3588 uses 4 to reach the
 * Cortex-A76 cluster). Parallel jobs use one small
 * shared FIFO; the only spin lock protects enqueue/dequeue. Compute never
 * holds a lock. Multiple dispatchers may therefore overlap Talker and Mimi
 * without OpenMP teams or a global inference lock.
 */
int minimindo_parallel_session_begin(unsigned threads);
void minimindo_parallel_session_end(void);
void minimindo_parallel_set_threads(unsigned threads);
unsigned minimindo_parallel_threads(void);
void minimindo_parallel_for(size_t count, minimindo_parallel_task task,
                            void *context);

/* Pin the calling thread to a logical lane for explicit stage ownership. */
int minimindo_parallel_pin_current(unsigned cpu);

#endif
