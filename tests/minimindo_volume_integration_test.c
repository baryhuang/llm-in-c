#define _GNU_SOURCE

#include "minimindo_volume.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/input.h>
#include <linux/uinput.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

static const char *const test_device_name = "MiniMind P10S volume test";

static int pause_ms(long milliseconds)
{
    struct timespec delay;
    delay.tv_sec = milliseconds / 1000L;
    delay.tv_nsec = milliseconds % 1000L * 1000000L;
    while (nanosleep(&delay, &delay) != 0) {
        if (errno != EINTR) return -1;
    }
    return 0;
}

static int create_test_device(void)
{
    const int descriptor = open("/dev/uinput", O_WRONLY | O_CLOEXEC);
    if (descriptor < 0 ||
        ioctl(descriptor, UI_SET_EVBIT, EV_KEY) != 0 ||
        ioctl(descriptor, UI_SET_EVBIT, EV_SYN) != 0 ||
        ioctl(descriptor, UI_SET_KEYBIT, KEY_VOLUMEUP) != 0 ||
        ioctl(descriptor, UI_SET_KEYBIT, KEY_VOLUMEDOWN) != 0 ||
        ioctl(descriptor, UI_SET_KEYBIT, KEY_MUTE) != 0) {
        if (descriptor >= 0) close(descriptor);
        return -1;
    }
    struct uinput_setup setup;
    memset(&setup, 0, sizeof(setup));
    setup.id.bustype = BUS_USB;
    setup.id.vendor = 0x1234;
    setup.id.product = 0x5684;
    setup.id.version = 0x0201;
    (void)snprintf(setup.name, sizeof(setup.name), "%s", test_device_name);
    if (ioctl(descriptor, UI_DEV_SETUP, &setup) != 0 ||
        ioctl(descriptor, UI_DEV_CREATE) != 0) {
        close(descriptor);
        return -1;
    }
    return descriptor;
}

static int find_test_event(char *path, size_t path_size)
{
    for (unsigned attempt = 0; attempt < 50U; ++attempt) {
        for (unsigned index = 0; index < 32U; ++index) {
            char candidate[64];
            char name[128];
            (void)snprintf(
                candidate, sizeof(candidate), "/dev/input/event%u", index);
            const int descriptor = open(candidate, O_RDONLY | O_CLOEXEC);
            if (descriptor < 0) continue;
            memset(name, 0, sizeof(name));
            const int result = ioctl(descriptor, EVIOCGNAME(sizeof(name)), name);
            close(descriptor);
            if (result >= 0 && strcmp(name, test_device_name) == 0) {
                (void)snprintf(path, path_size, "%s", candidate);
                return 0;
            }
        }
        if (pause_ms(20L) != 0) break;
    }
    return -1;
}

static int emit_event(int descriptor, unsigned short type,
                      unsigned short code, int value)
{
    struct input_event event;
    memset(&event, 0, sizeof(event));
    event.type = type;
    event.code = code;
    event.value = value;
    return write(descriptor, &event, sizeof(event)) ==
                   (ssize_t)sizeof(event) ? 0 : -1;
}

static int emit_key(int descriptor, unsigned short code)
{
    return emit_event(descriptor, EV_KEY, code, 1) != 0 ||
           emit_event(descriptor, EV_SYN, SYN_REPORT, 0) != 0 ||
           emit_event(descriptor, EV_KEY, code, 0) != 0 ||
           emit_event(descriptor, EV_SYN, SYN_REPORT, 0) != 0 ? -1 : 0;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s up|down|mute\n", argv[0]);
        return 2;
    }
    unsigned short code = 0;
    if (strcmp(argv[1], "up") == 0) code = KEY_VOLUMEUP;
    else if (strcmp(argv[1], "down") == 0) code = KEY_VOLUMEDOWN;
    else if (strcmp(argv[1], "mute") == 0) code = KEY_MUTE;
    else return 2;

    int result = 1;
    const int device = create_test_device();
    if (device < 0) {
        perror("uinput create");
        return 1;
    }
    char event_path[64];
    if (find_test_event(event_path, sizeof(event_path)) != 0 ||
        setenv("MINIMINDO_VOLUME_INPUT_DEVICE", event_path, 1) != 0 ||
        minimindo_volume_monitor_start() != 0 || pause_ms(100L) != 0 ||
        emit_key(device, code) != 0 || pause_ms(100L) != 0) {
        fprintf(stderr, "volume integration test failed\n");
    } else {
        minimindo_volume_monitor_stop();
        printf("volume integration event sent: %s via %s\n",
               argv[1], event_path);
        result = 0;
    }
    if (result != 0) minimindo_volume_monitor_stop();
    (void)ioctl(device, UI_DEV_DESTROY);
    close(device);
    return result;
}
