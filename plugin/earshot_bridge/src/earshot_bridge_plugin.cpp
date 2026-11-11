#include <clap/clap.h>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <atomic>
#include <thread>
#include <chrono>
#include <random>
#include <filesystem>
#include <cmath>
#include <sys/stat.h>
#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>
#include <signal.h>
#include <cerrno>
extern char **environ;

static const char *features[] = {
    CLAP_PLUGIN_FEATURE_AUDIO_EFFECT,
    CLAP_PLUGIN_FEATURE_MONO,
    CLAP_PLUGIN_FEATURE_STEREO,
    CLAP_PLUGIN_FEATURE_RESTORATION,
    CLAP_PLUGIN_FEATURE_UTILITY,
    nullptr
};

static const clap_plugin_descriptor_t s_bridge_desc = {
    CLAP_VERSION_INIT,
    "fi.iki.dst.earshot.bridge",
    "Earshot Bridge",
    "Olli Sulopuisto",
    "https://github.com/ollisulopuisto/earshot",
    "",
    "",
    "0.1.0",
    "Audio bridge plugin for earshot speech restoration",
    &features[0]
};

// Parameter IDs
enum {
    PARAM_CAPTURE = 0,
    PARAM_RESTORE = 1,
    PARAM_COUNT = 2
};

#include "wav_io.h"

typedef struct earshot_bridge_plug {
    clap_plugin_t plugin;
    const clap_host_t *host;
    uint32_t channels;
    double sample_rate;

    std::atomic<bool> capturing { false };
    std::atomic<bool> trigger_restore_flag { false };
    std::atomic<bool> restoring_in_background { false };
    std::atomic<bool> playing_restored { false };

    size_t max_capture_samples { 0 };
    std::vector<float> captured_buffer;
    std::vector<float> restored_buffer;

    // Audio owns capture while job_pending is false; the worker only reads it
    // after the audio thread publishes a job. New capture is refused until done.
    std::atomic<bool> job_pending { false };
    std::atomic<bool> result_ready { false };
    std::atomic<bool> stop_worker { false };
    std::vector<float> pending_result;
    int64_t capture_start { 0 }, result_start { 0 }, active_start { 0 };
    bool capture_has_position { false }, result_has_position { false }, active_has_position { false };
    bool capture_first_block { true };
    bool active_result { false };
    std::string bridge_dir;
    std::string engine { "route" };
    std::thread worker_thread;
} earshot_bridge_plug_t;

///////////////////////////////////
// clap_configurable_audio_ports //
///////////////////////////////////

static bool bridge_can_apply_configuration(
    const clap_plugin_t *plugin,
    const struct clap_audio_port_configuration_request *requests,
    uint32_t request_count)
{
    if (request_count != 2) return false;
    if (requests[0].is_input == requests[1].is_input) return false;
    if (requests[0].port_index != 0 || requests[1].port_index != 0) return false;
    if (requests[0].channel_count != requests[1].channel_count) return false;
    if (requests[0].channel_count < 1 || requests[0].channel_count > 2) return false;
    return true;
}

static bool bridge_apply_configuration(
    const clap_plugin_t *plugin,
    const struct clap_audio_port_configuration_request *requests,
    uint32_t request_count)
{
    if (!bridge_can_apply_configuration(plugin, requests, request_count)) return false;
    auto *plug = (earshot_bridge_plug_t *)plugin->plugin_data;
    plug->channels = requests[0].channel_count;
    return true;
}

static const clap_plugin_configurable_audio_ports_t s_bridge_configurable_audio_ports = {
    bridge_can_apply_configuration,
    bridge_apply_configuration
};

//////////////////////
// clap_audio_ports //
//////////////////////

static uint32_t bridge_audio_ports_count(const clap_plugin_t *plugin, bool is_input) {
    return 1;
}

static bool bridge_audio_ports_get(const clap_plugin_t *plugin, uint32_t index, bool is_input,
                                   clap_audio_port_info_t *info) {
    auto *plug = (earshot_bridge_plug_t *)plugin->plugin_data;
    if (index > 0) return false;
    info->id = 0;
    if (is_input) {
        snprintf(info->name, sizeof(info->name), "%s", "Audio In");
    } else {
        snprintf(info->name, sizeof(info->name), "%s", "Audio Out");
    }
    info->flags = CLAP_AUDIO_PORT_IS_MAIN;
    info->in_place_pair = 0;
    info->channel_count = plug ? plug->channels : 1;
    info->port_type = (plug && plug->channels == 2) ? CLAP_PORT_STEREO : CLAP_PORT_MONO;
    return true;
}

static const clap_plugin_audio_ports_t s_bridge_audio_ports = {
    bridge_audio_ports_count,
    bridge_audio_ports_get
};

/////////////////
// clap_params //
/////////////////

static uint32_t bridge_params_count(const clap_plugin_t *plugin) {
    return PARAM_COUNT;
}

static bool bridge_params_get_info(const clap_plugin_t *plugin, uint32_t param_index,
                                   clap_param_info_t *param_info) {
    if (param_index == PARAM_CAPTURE) {
        param_info->id = PARAM_CAPTURE;
        strncpy(param_info->name, "Capture Selection", CLAP_NAME_SIZE);
        param_info->module[0] = 0;
        param_info->min_value = 0.0;
        param_info->max_value = 1.0;
        param_info->default_value = 0.0;
        param_info->flags = CLAP_PARAM_IS_AUTOMATABLE | CLAP_PARAM_IS_STEPPED;
        param_info->cookie = nullptr;
        return true;
    } else if (param_index == PARAM_RESTORE) {
        param_info->id = PARAM_RESTORE;
        strncpy(param_info->name, "Restore With Earshot", CLAP_NAME_SIZE);
        param_info->module[0] = 0;
        param_info->min_value = 0.0;
        param_info->max_value = 1.0;
        param_info->default_value = 0.0;
        param_info->flags = CLAP_PARAM_IS_AUTOMATABLE | CLAP_PARAM_IS_STEPPED;
        param_info->cookie = nullptr;
        return true;
    }
    return false;
}

// Spawn without a shell and retain the child process group so teardown can
// cancel an outstanding restore rather than waiting for a model to finish.
static bool bridge_run_restore(earshot_bridge_plug_t *plug, const std::string& input, const std::string& output) {
    char *args[] = {const_cast<char*>("earshot"), const_cast<char*>("restore"),
        const_cast<char*>(input.c_str()), const_cast<char*>("-o"),
        const_cast<char*>(output.c_str()), const_cast<char*>("--engine"),
        const_cast<char*>(plug->engine.c_str()), nullptr};
    posix_spawnattr_t attr;
    if (posix_spawnattr_init(&attr) != 0) return false;
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&attr, 0);
    pid_t child = -1;
    const int spawned = posix_spawnp(&child, "earshot", nullptr, &attr, args, environ);
    posix_spawnattr_destroy(&attr);
    if (spawned != 0) return false;
    int status = 0;
    while (true) {
        const pid_t result = waitpid(child, &status, WNOHANG);
        if (result == child) return WIFEXITED(status) && WEXITSTATUS(status) == 0;
        if (result < 0 && errno != EINTR) return false;
        if (plug->stop_worker.load()) {
            kill(-child, SIGTERM);
            for (int i = 0; i < 20; ++i) {
                if (waitpid(child, &status, WNOHANG) == child) return false;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            kill(-child, SIGKILL);
            while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// Runs on activation, never in process/params.flush. One worker owns all
// filesystem, subprocess and decode work for this instance until deactivation.
static void bridge_worker(earshot_bridge_plug_t *plug) {
    while (!plug->stop_worker.load(std::memory_order_acquire)) {
        if (!plug->job_pending.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        const std::string in_wav = plug->bridge_dir + "/selection_in.wav";
        const std::string out_wav = plug->bridge_dir + "/selection_out.wav";
        std::error_code file_error;
        std::filesystem::remove(out_wav, file_error); // never accept stale output after a failed run
        if (WavWriter::writeWavHeaderAndData(in_wav, plug->captured_buffer, static_cast<int>(plug->sample_rate))) {
            const bool restored_ok = bridge_run_restore(plug, in_wav, out_wav);
            int sr = 0;
            std::vector<float> restored;
            if (restored_ok && WavWriter::readWavData(out_wav, restored, sr) && sr == int(plug->sample_rate) && restored.size() == plug->captured_buffer.size()) {
                // At most one unpublished result. Audio swaps this vector into
                // its own storage before releasing the slot back to the worker.
                while (plug->result_ready.load(std::memory_order_acquire) && !plug->stop_worker.load())
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (!plug->stop_worker.load()) {
                    plug->pending_result = std::move(restored);
                    plug->result_start = plug->capture_start;
                    plug->result_has_position = plug->capture_has_position;
                    plug->result_ready.store(true, std::memory_order_release);
                }
            }
        }
        plug->restoring_in_background.store(false);
        plug->job_pending.store(false, std::memory_order_release);
    }
}

// Audio-thread handoff only: no allocation, join, subprocess or file access.
static void bridge_queue_restore(earshot_bridge_plug_t *plug) {
    if (plug->job_pending.load(std::memory_order_acquire) || plug->captured_buffer.empty()) return;
    plug->capturing.store(false);
    plug->restoring_in_background.store(true);
    plug->job_pending.store(true, std::memory_order_release);
}

static bool bridge_params_get_value(const clap_plugin_t *plugin, clap_id param_id, double *out_value) {
    auto *plug = (earshot_bridge_plug_t *)plugin->plugin_data;
    if (param_id == PARAM_CAPTURE) {
        *out_value = plug->capturing.load() ? 1.0 : 0.0;
        return true;
    } else if (param_id == PARAM_RESTORE) {
        *out_value = plug->playing_restored.load() ? 1.0 : 0.0;
        return true;
    }
    return false;
}

static bool bridge_params_value_to_text(const clap_plugin_t *plugin, clap_id param_id,
                                        double value, char *out_buffer, uint32_t out_buffer_capacity) {
    snprintf(out_buffer, out_buffer_capacity, "%s", (value >= 0.5) ? "ON" : "OFF");
    return true;
}

static bool bridge_params_text_to_value(const clap_plugin_t *plugin, clap_id param_id,
                                        const char *param_value_text, double *out_value) {
    if (!strcmp(param_value_text, "ON") || !strcmp(param_value_text, "1")) {
        *out_value = 1.0;
        return true;
    }
    *out_value = 0.0;
    return true;
}

static void bridge_handle_param_event(earshot_bridge_plug_t *plug, const clap_event_header_t *hdr) {
    if (hdr->type == CLAP_EVENT_PARAM_VALUE) {
        const auto *ev = (const clap_event_param_value_t *)hdr;
        if (ev->param_id == PARAM_CAPTURE) {
            bool start = ev->value >= 0.5;
            if (start && !plug->job_pending.load(std::memory_order_acquire)) {
                // A new selection supersedes both published and active audio.
                // job_pending=false acquires the worker's completed handoff,
                // so discarding its ready slot cannot race with publication.
                plug->active_result = false;
                plug->playing_restored.store(false);
                plug->result_ready.store(false, std::memory_order_release);
                plug->captured_buffer.clear();
                plug->capture_first_block = true;
                plug->capture_has_position = false;
                plug->capturing.store(true);
            } else {
                plug->capturing.store(false);
            }
        } else if (ev->param_id == PARAM_RESTORE) {
            if (ev->value >= 0.5) {
                plug->trigger_restore_flag.store(true);
            }
        }
    }
}

static void bridge_params_flush(const clap_plugin_t *plugin,
                                const clap_input_events_t *in,
                                const clap_output_events_t *out) {
    auto *plug = (earshot_bridge_plug_t *)plugin->plugin_data;
    uint32_t size = in->size(in);
    for (uint32_t i = 0; i < size; ++i) {
        const auto *hdr = in->get(in, i);
        bridge_handle_param_event(plug, hdr);
    }
    if (plug->trigger_restore_flag.exchange(false)) {
        bridge_queue_restore(plug);
    }
}

static const clap_plugin_params_t s_bridge_params = {
    bridge_params_count,
    bridge_params_get_info,
    bridge_params_get_value,
    bridge_params_value_to_text,
    bridge_params_text_to_value,
    bridge_params_flush
};

static bool bridge_init(const struct clap_plugin *plugin) {
    auto *plug = (earshot_bridge_plug_t *)plugin->plugin_data;
    plug->channels = 1;
    plug->sample_rate = 48000.0;
    return true;
}

static void bridge_destroy(const struct clap_plugin *plugin) {
    auto *plug = (earshot_bridge_plug_t *)plugin->plugin_data;
    plug->stop_worker.store(true);
    if (plug->worker_thread.joinable()) plug->worker_thread.join();
    std::error_code cleanup_error;
    std::filesystem::remove_all(plug->bridge_dir, cleanup_error);
    delete plug;
}

static bool bridge_activate(const struct clap_plugin *plugin, double sample_rate,
                            uint32_t min_frames_count, uint32_t max_frames_count) {
    auto *plug = (earshot_bridge_plug_t *)plugin->plugin_data;
    plug->sample_rate = sample_rate;
    // Fixed capture budget, allocated while the audio callback is inactive.
    plug->max_capture_samples = static_cast<size_t>(sample_rate * 60);
    plug->captured_buffer.clear();
    plug->capture_first_block = true;
    plug->capture_has_position = false;
    plug->captured_buffer.reserve(plug->max_capture_samples);
    plug->stop_worker.store(false);
    if (!plug->worker_thread.joinable()) plug->worker_thread = std::thread(bridge_worker, plug);
    return true;
}

static void bridge_deactivate(const struct clap_plugin *plugin) {
    auto *plug = (earshot_bridge_plug_t *)plugin->plugin_data;
    plug->stop_worker.store(true);
    if (plug->worker_thread.joinable()) plug->worker_thread.join();
    plug->job_pending.store(false);
    plug->capturing.store(false);
    plug->result_ready.store(false);
    plug->active_result = false;
    plug->playing_restored.store(false);
}

static bool bridge_start_processing(const struct clap_plugin *plugin) {
    return true;
}

static void bridge_stop_processing(const struct clap_plugin *plugin) {}

static void bridge_reset(const struct clap_plugin *plugin) {
    auto *plug = (earshot_bridge_plug_t *)plugin->plugin_data;
    // Playback is always re-indexed from transport, never from elapsed blocks.
    plug->playing_restored.store(false);
    // Reset does not identify stop versus seek. Keep the recorded selection
    // and arm; process checks the next playing position before appending.

}

static clap_process_status bridge_process(const struct clap_plugin *plugin,
                                          const clap_process_t *process) {
    auto *plug = (earshot_bridge_plug_t *)plugin->plugin_data;

    if (process->audio_inputs_count == 0 || process->audio_outputs_count == 0) {
        return CLAP_PROCESS_CONTINUE;
    }
    const uint32_t nframes = process->frames_count;
    const auto &in_buf = process->audio_inputs[0];
    auto &out_buf = process->audio_outputs[0];

    uint32_t channels = (in_buf.channel_count < out_buf.channel_count)
                            ? in_buf.channel_count
                            : out_buf.channel_count;

    if (plug->result_ready.load(std::memory_order_acquire)) {
        plug->restored_buffer.swap(plug->pending_result);
        plug->active_start = plug->result_start;
        plug->active_has_position = plug->result_has_position;
        plug->active_result = true;
        plug->result_ready.store(false, std::memory_order_release);
    }
    const auto *transport = process->transport;
    const bool has_position = transport && (transport->flags & CLAP_TRANSPORT_HAS_SECONDS_TIMELINE);
    const bool host_playing = has_position && (transport->flags & CLAP_TRANSPORT_IS_PLAYING);
    const int64_t block_start = has_position ? static_cast<int64_t>(std::llround(
        double(transport->song_pos_seconds) / double(CLAP_SECTIME_FACTOR) * plug->sample_rate)) : 0;
    bool replaced = false;
    uint32_t event_index = 0;
    const uint32_t event_count = process->in_events ? process->in_events->size(process->in_events) : 0;
    for (uint32_t f = 0; f < nframes; ++f) {
        while (event_index < event_count) {
            const auto *event = process->in_events->get(process->in_events, event_index);
            if (event->time > f) break;
            bridge_handle_param_event(plug, event);
            ++event_index;
        }
        if (plug->trigger_restore_flag.exchange(false)) bridge_queue_restore(plug);
        // Capture precedes output so in-place host buffers do not record the
        // previous restoration instead of the incoming selection.
        if (plug->capturing.load() && !plug->job_pending.load(std::memory_order_acquire) && host_playing && in_buf.data32 && in_buf.channel_count > 0) {
            if (plug->capture_first_block) {
                plug->capture_start = block_start + f;
                plug->capture_has_position = host_playing;
                plug->capture_first_block = false;
            } else if (!has_position || block_start + f != plug->capture_start + int64_t(plug->captured_buffer.size())) {
                // Do not join audio from different timeline regions or erase
                // the completed selection on stop/seek. Restore can still use it.
                plug->capturing.store(false);
            }
            if (!plug->capturing.load()) {
                // Existing selection remains immutable until Restore/recapture.
            } else if (plug->captured_buffer.size() >= plug->max_capture_samples) {
                plug->capturing.store(false);
            } else {
                float sum = 0.0f;
                for (uint32_t c = 0; c < in_buf.channel_count; ++c) sum += in_buf.data32[c][f];
                plug->captured_buffer.push_back(sum / in_buf.channel_count);
            }
        }
        const int64_t pos = block_start + f - plug->active_start;
        const bool use = host_playing && !plug->capturing.load() && plug->active_result && plug->active_has_position && pos >= 0 && uint64_t(pos) < plug->restored_buffer.size();
        for (uint32_t c = 0; c < channels; ++c) {
            if (in_buf.data32 && out_buf.data32) {
                out_buf.data32[c][f] = use ? plug->restored_buffer[size_t(pos)] : in_buf.data32[c][f];
                replaced = replaced || use;
            } else if (in_buf.data64 && out_buf.data64) {
                out_buf.data64[c][f] = in_buf.data64[c][f];
            }
        }
    }
    plug->playing_restored.store(replaced);

    return CLAP_PROCESS_CONTINUE;
}

static const void *bridge_get_extension(const struct clap_plugin *plugin, const char *id) {
    if (!strcmp(id, CLAP_EXT_AUDIO_PORTS)) return &s_bridge_audio_ports;
    if (!strcmp(id, CLAP_EXT_CONFIGURABLE_AUDIO_PORTS)) return &s_bridge_configurable_audio_ports;
    if (!strcmp(id, CLAP_EXT_PARAMS)) return &s_bridge_params;
    return nullptr;
}

static void bridge_on_main_thread(const struct clap_plugin *plugin) {}

clap_plugin_t *create_earshot_bridge_plugin(const clap_host_t *host) {
    auto *plug = new earshot_bridge_plug_t();
    plug->host = host;
    plug->channels = 1;
    plug->sample_rate = 48000.0;

    // Atomic, private directory: no shared predictable recording path or
    // symlink-following mkdir. mkdtemp creates it with mode 0700.
    std::string pattern = "/tmp/earshot-bridge-XXXXXX";
    std::vector<char> directory(pattern.begin(), pattern.end());
    directory.push_back('\0');
    const char *made = mkdtemp(directory.data());
    if (!made) { delete plug; return nullptr; }
    plug->bridge_dir = made;

    plug->plugin.desc = &s_bridge_desc;
    plug->plugin.plugin_data = plug;
    plug->plugin.init = bridge_init;
    plug->plugin.destroy = bridge_destroy;
    plug->plugin.activate = bridge_activate;
    plug->plugin.deactivate = bridge_deactivate;
    plug->plugin.start_processing = bridge_start_processing;
    plug->plugin.stop_processing = bridge_stop_processing;
    plug->plugin.reset = bridge_reset;
    plug->plugin.process = bridge_process;
    plug->plugin.get_extension = bridge_get_extension;
    plug->plugin.on_main_thread = bridge_on_main_thread;
    return &plug->plugin;
}

uint32_t bridge_factory_get_plugin_count(const struct clap_plugin_factory *factory) {
    return 1;
}

const clap_plugin_descriptor_t *bridge_factory_get_plugin_descriptor(
    const struct clap_plugin_factory *factory, uint32_t index) {
    return &s_bridge_desc;
}

const clap_plugin_t *bridge_factory_create_plugin(const struct clap_plugin_factory *factory,
                                                  const clap_host_t *host, const char *plugin_id) {
    if (!clap_version_is_compatible(host->clap_version)) return nullptr;
    if (!strcmp(plugin_id, s_bridge_desc.id)) return create_earshot_bridge_plugin(host);
    return nullptr;
}

static const clap_plugin_factory_t s_bridge_factory = {
    bridge_factory_get_plugin_count,
    bridge_factory_get_plugin_descriptor,
    bridge_factory_create_plugin
};

bool earshot_bridge_entry_init(const char *plugin_path) {
    return true;
}

void earshot_bridge_entry_deinit(void) {}

const void *earshot_bridge_entry_get_factory(const char *factory_id) {
    if (!strcmp(factory_id, CLAP_PLUGIN_FACTORY_ID)) return &s_bridge_factory;
    return nullptr;
}
