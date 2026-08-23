#define _GNU_SOURCE

#include "minimindo_volume.h"

#ifndef MINIMINDO_ALSA_VOLUME_MONITOR
#define MINIMINDO_ALSA_VOLUME_MONITOR 0
#endif

#if defined(__linux__) && MINIMINDO_ALSA_VOLUME_MONITOR

#include <alsa/asoundlib.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/input.h>
#include <poll.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

enum {
    P10S_VENDOR = 0x1234,
    P10S_PRODUCT = 0x5684,
    VOLUME_STEP_PERCENT = 2,
    INPUT_RESCAN_MS = 1000
};

typedef struct {
    snd_mixer_t *handle;
    snd_mixer_elem_t *element;
    long minimum;
    long maximum;
} volume_mixer;

typedef struct {
    pthread_t thread;
    int wake_read;
    int wake_write;
    int started;
} volume_monitor;

static volume_monitor resident_volume;

static void volume_log(const char *format, ...)
{
    char line[320];
    va_list arguments;
    va_start(arguments, format);
    const int length = vsnprintf(line, sizeof(line), format, arguments);
    va_end(arguments);
    if (length <= 0) return;
    size_t bytes = (size_t)length;
    if (bytes >= sizeof(line)) bytes = sizeof(line) - 1U;
    const ssize_t ignored = write(STDOUT_FILENO, line, bytes);
    (void)ignored;
}

static int test_bit(const unsigned long *bits, unsigned bit)
{
    const unsigned width = (unsigned)(sizeof(*bits) * 8U);
    return (bits[bit / width] & (1UL << (bit % width))) != 0UL;
}

static int validate_p10s_input(int descriptor, int *monotonic_events)
{
    enum { WORD_BITS = (int)(sizeof(unsigned long) * 8U) };
    unsigned long keys[(KEY_MAX + WORD_BITS) / WORD_BITS];
    struct input_id identity;
    memset(&identity, 0, sizeof(identity));
    memset(keys, 0, sizeof(keys));
    const int identity_result = ioctl(descriptor, EVIOCGID, &identity);
    const int key_result = ioctl(
        descriptor, EVIOCGBIT(EV_KEY, sizeof(keys)), keys);
    if (identity_result != 0 || key_result < 0 ||
        identity.vendor != P10S_VENDOR || identity.product != P10S_PRODUCT ||
        !test_bit(keys, KEY_VOLUMEUP) ||
        !test_bit(keys, KEY_VOLUMEDOWN) || !test_bit(keys, KEY_MUTE))
        return -1;
    const clockid_t event_clock = CLOCK_MONOTONIC;
    *monotonic_events =
        ioctl(descriptor, EVIOCSCLOCKID, &event_clock) == 0;
    return 0;
}

static int open_p10s_input(char *selected, size_t selected_size,
                            int *monotonic_events)
{
    const char *configured = getenv("MINIMINDO_VOLUME_INPUT_DEVICE");
    if (configured != NULL && configured[0] != '\0') {
        const int descriptor = open(
            configured, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
        if (descriptor >= 0) {
            if (validate_p10s_input(descriptor, monotonic_events) == 0) {
                (void)snprintf(selected, selected_size, "%s", configured);
                return descriptor;
            }
            close(descriptor);
        }
        return -1;
    }
    for (unsigned index = 0; index < 32U; ++index) {
        char path[64];
        (void)snprintf(path, sizeof(path), "/dev/input/event%u", index);
        const int descriptor = open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
        if (descriptor < 0) continue;
        if (validate_p10s_input(descriptor, monotonic_events) == 0) {
            (void)snprintf(selected, selected_size, "%s", path);
            return descriptor;
        }
        close(descriptor);
    }
    return -1;
}

static void close_mixer(volume_mixer *mixer)
{
    if (mixer->handle != NULL) snd_mixer_close(mixer->handle);
    memset(mixer, 0, sizeof(*mixer));
}

/*
 * The P10S mixer is card 0 on the A113X hub and card 4 on the RK3588 hub,
 * where the on-SoC DP/HDMI/ES8388 devices enumerate first. The card follows
 * the enclosure, so it is configuration rather than a build-time constant.
 */
static const char *mixer_device(void)
{
    const char *configured = getenv("MINIMINDO_VOLUME_MIXER");
    if (configured != NULL && configured[0] != '\0') return configured;
    return "hw:0";
}

static int open_mixer(volume_mixer *mixer)
{
    snd_mixer_selem_id_t *identifier = NULL;
    memset(mixer, 0, sizeof(*mixer));
    if (snd_mixer_open(&mixer->handle, 0) < 0 ||
        snd_mixer_attach(mixer->handle, mixer_device()) < 0 ||
        snd_mixer_selem_register(mixer->handle, NULL, NULL) < 0 ||
        snd_mixer_load(mixer->handle) < 0 ||
        snd_mixer_selem_id_malloc(&identifier) < 0) {
        snd_mixer_selem_id_free(identifier);
        close_mixer(mixer);
        return -1;
    }
    snd_mixer_selem_id_set_index(identifier, 0U);
    snd_mixer_selem_id_set_name(identifier, "PCM");
    mixer->element = snd_mixer_find_selem(mixer->handle, identifier);
    snd_mixer_selem_id_free(identifier);
    if (mixer->element == NULL ||
        !snd_mixer_selem_has_playback_volume(mixer->element) ||
        snd_mixer_selem_get_playback_volume_range(
            mixer->element, &mixer->minimum, &mixer->maximum) < 0 ||
        mixer->maximum <= mixer->minimum) {
        close_mixer(mixer);
        return -1;
    }
    return 0;
}

static long mixer_percent(const volume_mixer *mixer, long value)
{
    const long span = mixer->maximum - mixer->minimum;
    return ((value - mixer->minimum) * 100L + span / 2L) / span;
}

static int read_mixer_state(volume_mixer *mixer, long *value, int *enabled)
{
    (void)snd_mixer_handle_events(mixer->handle);
    if (snd_mixer_selem_get_playback_volume(
            mixer->element, SND_MIXER_SCHN_FRONT_LEFT, value) < 0)
        return -1;
    if (snd_mixer_selem_has_playback_switch(mixer->element)) {
        if (snd_mixer_selem_get_playback_switch(
                mixer->element, SND_MIXER_SCHN_FRONT_LEFT, enabled) < 0)
            return -1;
    } else {
        *enabled = 1;
    }
    return 0;
}

static double event_latency_ms(const struct input_event *event,
                               int monotonic_events)
{
    if (!monotonic_events) return -1.0;
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return -1.0;
    const double event_time = (double)event->time.tv_sec +
                              (double)event->time.tv_usec / 1000000.0;
    return ((double)now.tv_sec + (double)now.tv_nsec / 1000000000.0 -
            event_time) * 1000.0;
}

static int apply_volume_event(volume_mixer *mixer,
                              const struct input_event *event,
                              int monotonic_events)
{
    long value = 0;
    int enabled = 1;
    if (read_mixer_state(mixer, &value, &enabled) != 0) return -1;
    const long span = mixer->maximum - mixer->minimum;
    long step = (span * VOLUME_STEP_PERCENT + 99L) / 100L;
    if (step < 1L) step = 1L;
    const char *event_name = NULL;
    if (event->code == KEY_MUTE) {
        event_name = "mute";
        enabled = !enabled;
        if (snd_mixer_selem_has_playback_switch(mixer->element) &&
            snd_mixer_selem_set_playback_switch_all(
                mixer->element, enabled) < 0)
            return -1;
    } else if (event->code == KEY_VOLUMEUP) {
        event_name = "up";
        if (value > mixer->maximum - step) value = mixer->maximum;
        else value += step;
        enabled = 1;
        if (snd_mixer_selem_set_playback_volume_all(
                mixer->element, value) < 0 ||
            (snd_mixer_selem_has_playback_switch(mixer->element) &&
             snd_mixer_selem_set_playback_switch_all(
                 mixer->element, enabled) < 0))
            return -1;
    } else if (event->code == KEY_VOLUMEDOWN) {
        event_name = "down";
        if (value < mixer->minimum + step) value = mixer->minimum;
        else value -= step;
        enabled = 1;
        if (snd_mixer_selem_set_playback_volume_all(
                mixer->element, value) < 0 ||
            (snd_mixer_selem_has_playback_switch(mixer->element) &&
             snd_mixer_selem_set_playback_switch_all(
                 mixer->element, enabled) < 0))
            return -1;
    } else {
        return 0;
    }
    volume_log("VOLUME event=%s percent=%ld muted=%d raw=%ld latency_ms=%.3f\n",
               event_name, mixer_percent(mixer, value), !enabled, value,
               event_latency_ms(event, monotonic_events));
    return 0;
}

static int wait_for_stop(int wake_read, int timeout_ms)
{
    struct pollfd wake;
    wake.fd = wake_read;
    wake.events = POLLIN;
    wake.revents = 0;
    for (;;) {
        const int result = poll(&wake, 1, timeout_ms);
        if (result >= 0) return result > 0;
        if (errno != EINTR) return 1;
    }
}

static void *volume_thread(void *opaque)
{
    volume_monitor *monitor = opaque;
    (void)pthread_setname_np(pthread_self(), "volume-event");
    volume_mixer mixer;
    memset(&mixer, 0, sizeof(mixer));
    for (;;) {
        char input_path[64];
        int monotonic_events = 0;
        const int input = open_p10s_input(
            input_path, sizeof(input_path), &monotonic_events);
        if (input < 0) {
            if (wait_for_stop(monitor->wake_read, INPUT_RESCAN_MS)) break;
            continue;
        }
        if (open_mixer(&mixer) != 0) {
            close(input);
            if (wait_for_stop(monitor->wake_read, INPUT_RESCAN_MS)) break;
            continue;
        }
        long value = 0;
        int enabled = 1;
        if (read_mixer_state(&mixer, &value, &enabled) == 0)
            volume_log("VOLUME ready device=%s mixer=%s element=PCM "
                       "step_percent=%d percent=%ld muted=%d\n",
                       input_path, mixer_device(), VOLUME_STEP_PERCENT,
                       mixer_percent(&mixer, value), !enabled);
        struct pollfd descriptors[2];
        descriptors[0].fd = monitor->wake_read;
        descriptors[0].events = POLLIN;
        descriptors[1].fd = input;
        descriptors[1].events = POLLIN;
        int stop = 0;
        int mixer_failed = 0;
        while (!stop) {
            descriptors[0].revents = 0;
            descriptors[1].revents = 0;
            const int poll_result = poll(descriptors, 2, -1);
            if (poll_result < 0) {
                if (errno == EINTR) continue;
                break;
            }
            if ((descriptors[0].revents & POLLIN) != 0) {
                stop = 1;
                break;
            }
            if ((descriptors[1].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
                break;
            if ((descriptors[1].revents & POLLIN) != 0) {
                struct input_event events[16];
                const ssize_t bytes = read(input, events, sizeof(events));
                if (bytes < 0) {
                    if (errno == EINTR || errno == EAGAIN) continue;
                    break;
                }
                if (bytes == 0) break;
                const size_t count = (size_t)bytes / sizeof(events[0]);
                for (size_t index = 0; index < count; ++index) {
                    const struct input_event *event = &events[index];
                    if (event->type == EV_KEY &&
                        (event->value == 1 || event->value == 2) &&
                        apply_volume_event(
                            &mixer, event, monotonic_events) != 0) {
                        volume_log("VOLUME error=alsa_update code=%u\n",
                                   event->code);
                        mixer_failed = 1;
                        break;
                    }
                }
                if (mixer_failed) break;
            }
        }
        close_mixer(&mixer);
        close(input);
        if (stop) break;
    }
    return NULL;
}

int minimindo_volume_monitor_start(void)
{
    volume_monitor *monitor = &resident_volume;
    memset(monitor, 0, sizeof(*monitor));
    monitor->wake_read = -1;
    monitor->wake_write = -1;
    int wake[2];
    if (pipe(wake) != 0) return -1;
    monitor->wake_read = wake[0];
    monitor->wake_write = wake[1];
    const int flags = fcntl(monitor->wake_write, F_GETFL, 0);
    if (flags < 0 ||
        fcntl(monitor->wake_write, F_SETFL, flags | O_NONBLOCK) != 0 ||
        fcntl(monitor->wake_read, F_SETFD, FD_CLOEXEC) != 0 ||
        fcntl(monitor->wake_write, F_SETFD, FD_CLOEXEC) != 0 ||
        pthread_create(&monitor->thread, NULL, volume_thread, monitor) != 0) {
        close(monitor->wake_read);
        close(monitor->wake_write);
        monitor->wake_read = -1;
        monitor->wake_write = -1;
        return -1;
    }
    monitor->started = 1;
    return 0;
}

void minimindo_volume_monitor_stop(void)
{
    volume_monitor *monitor = &resident_volume;
    if (!monitor->started) return;
    const unsigned char wake = 1U;
    const ssize_t ignored = write(monitor->wake_write, &wake, sizeof(wake));
    (void)ignored;
    pthread_join(monitor->thread, NULL);
    close(monitor->wake_read);
    close(monitor->wake_write);
    memset(monitor, 0, sizeof(*monitor));
    monitor->wake_read = -1;
    monitor->wake_write = -1;
}

#else

int minimindo_volume_monitor_start(void)
{
    return 0;
}

void minimindo_volume_monitor_stop(void)
{
}

#endif
