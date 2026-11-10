#include <clap/clap.h>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <atomic>
#include <chrono>
#include <sys/stat.h>
#include <unistd.h>

#if defined(_WIN32)
#include <direct.h>
#define mkdir_compat(dir) _mkdir(dir)
#else
#define mkdir_compat(dir) mkdir(dir, 0755)
#endif

// ARA extension declarations
static CLAP_CONSTEXPR const char CLAP_EXT_ARA_FACTORY[] = "org.ara-audio.ara.factory/2";
static CLAP_CONSTEXPR const char CLAP_EXT_ARA_PLUGINEXTENSION[] = "org.ara-audio.ara.pluginextension/2";
#define CLAP_PLUGIN_FEATURE_ARA_SUPPORTED "ara:supported"

typedef void *ARAFactoryPtr;
typedef void *ARAPlugInExtensionInstancePtr;
typedef void *ARADocumentControllerRef;
typedef int32_t ARAPlugInInstanceRoleFlags;

typedef struct clap_ara_factory {
    uint32_t(CLAP_ABI *get_factory_count)(const struct clap_ara_factory *factory);
    ARAFactoryPtr(CLAP_ABI *get_ara_factory)(const struct clap_ara_factory *factory, uint32_t index);
    const char *(CLAP_ABI *get_plugin_id)(const struct clap_ara_factory *factory, uint32_t index);
} clap_ara_factory_t;

typedef struct clap_ara_plugin_extension {
    ARAFactoryPtr(CLAP_ABI *get_factory)(const clap_plugin_t *plugin);
    ARAPlugInExtensionInstancePtr(CLAP_ABI *bind_to_document_controller)(
        const clap_plugin_t *plugin, ARADocumentControllerRef documentControllerRef,
        ARAPlugInInstanceRoleFlags knownRoles, ARAPlugInInstanceRoleFlags assignedRoles);
} clap_ara_plugin_extension_t;

static const char *features[] = {
    CLAP_PLUGIN_FEATURE_AUDIO_EFFECT,
    CLAP_PLUGIN_FEATURE_MONO,
    CLAP_PLUGIN_FEATURE_STEREO,
    CLAP_PLUGIN_FEATURE_RESTORATION,
    CLAP_PLUGIN_FEATURE_UTILITY,
    CLAP_PLUGIN_FEATURE_ARA_SUPPORTED,
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
    "Audio bridge plugin for earshot speech restoration (VST3, AU, ARA2, AAX)",
    &features[0]
};

// Parameter IDs
enum {
    PARAM_CAPTURE = 0,
    PARAM_RESTORE = 1,
    PARAM_COUNT = 2
};

struct WavWriter {
    static bool writeWavHeaderAndData(const std::string& path, const std::vector<float>& monoData, int sampleRate) {
        std::ofstream out(path, std::ios::binary);
        if (!out) return false;

        uint32_t numSamples = static_cast<uint32_t>(monoData.size());
        uint32_t byteRate = sampleRate * 4;
        uint32_t dataSize = numSamples * 4;
        uint32_t chunkSize = 36 + dataSize;
        uint16_t numChannels = 1;
        uint16_t bitsPerSample = 32;
        uint16_t audioFormat = 3; // IEEE float
        uint16_t blockAlign = 4;
        uint32_t subchunk1Size = 16;

        out.write("RIFF", 4);
        out.write(reinterpret_cast<const char*>(&chunkSize), 4);
        out.write("WAVE", 4);
        out.write("fmt ", 4);
        out.write(reinterpret_cast<const char*>(&subchunk1Size), 4);
        out.write(reinterpret_cast<const char*>(&audioFormat), 2);
        out.write(reinterpret_cast<const char*>(&numChannels), 2);
        uint32_t sr = static_cast<uint32_t>(sampleRate);
        out.write(reinterpret_cast<const char*>(&sr), 4);
        out.write(reinterpret_cast<const char*>(&byteRate), 4);
        out.write(reinterpret_cast<const char*>(&blockAlign), 2);
        out.write(reinterpret_cast<const char*>(&bitsPerSample), 2);
        out.write("data", 4);
        out.write(reinterpret_cast<const char*>(&dataSize), 4);
        if (!monoData.empty()) {
            out.write(reinterpret_cast<const char*>(monoData.data()), dataSize);
        }
        return true;
    }

    static bool readWavData(const std::string& path, std::vector<float>& monoData, int& sampleRate) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;

        char id[4];
        in.read(id, 4);
        if (strncmp(id, "RIFF", 4) != 0) return false;
        uint32_t fileSize = 0;
        in.read(reinterpret_cast<char*>(&fileSize), 4);
        in.read(id, 4);
        if (strncmp(id, "WAVE", 4) != 0) return false;

        uint16_t format = 0;
        uint16_t channels = 0;
        uint32_t sr = 0;
        uint32_t dataSize = 0;

        while (in.read(id, 4)) {
            uint32_t subchunkSize = 0;
            in.read(reinterpret_cast<char*>(&subchunkSize), 4);
            if (strncmp(id, "fmt ", 4) == 0) {
                in.read(reinterpret_cast<char*>(&format), 2);
                in.read(reinterpret_cast<char*>(&channels), 2);
                in.read(reinterpret_cast<char*>(&sr), 4);
                sampleRate = static_cast<int>(sr);
                in.seekg(subchunkSize - 8, std::ios::cur);
            } else if (strncmp(id, "data", 4) == 0) {
                dataSize = subchunkSize;
                break;
            } else {
                in.seekg(subchunkSize, std::ios::cur);
            }
        }

        if (dataSize == 0 || channels == 0) return false;

        if (format == 3) { // IEEE Float
            uint32_t totalFloats = dataSize / 4;
            std::vector<float> all(totalFloats);
            in.read(reinterpret_cast<char*>(all.data()), dataSize);
            uint32_t frames = totalFloats / channels;
            monoData.resize(frames);
            for (uint32_t i = 0; i < frames; ++i) {
                float sum = 0.0f;
                for (uint16_t c = 0; c < channels; ++c) {
                    sum += all[i * channels + c];
                }
                monoData[i] = sum / channels;
            }
            return true;
        } else if (format == 1) { // PCM 16 or 24
            uint32_t bytesPerSample = (dataSize / channels) / (dataSize / (channels * 2));
            if (bytesPerSample == 2) {
                uint32_t totalSamples = dataSize / 2;
                std::vector<int16_t> all(totalSamples);
                in.read(reinterpret_cast<char*>(all.data()), dataSize);
                uint32_t frames = totalSamples / channels;
                monoData.resize(frames);
                for (uint32_t i = 0; i < frames; ++i) {
                    float sum = 0.0f;
                    for (uint16_t c = 0; c < channels; ++c) {
                        sum += all[i * channels + c] / 32768.0f;
                    }
                    monoData[i] = sum / channels;
                }
                return true;
            }
        }
        return false;
    }
};

typedef struct {
    clap_plugin_t plugin;
    const clap_host_t *host;
    uint32_t channels;
    double sample_rate;

    std::atomic<bool> capturing { false };
    std::atomic<bool> playing_restored { false };
    std::vector<float> captured_buffer;
    std::vector<float> restored_buffer;
    size_t playback_pos { 0 };

    std::string bridge_dir { "/tmp/earshot_bridge" };
    std::string engine { "route" };
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

static bool bridge_trigger_restore(earshot_bridge_plug_t *plug) {
    if (plug->captured_buffer.empty()) return false;

    mkdir_compat(plug->bridge_dir.c_str());
    std::string in_wav = plug->bridge_dir + "/selection_in.wav";
    std::string out_wav = plug->bridge_dir + "/selection_out.wav";

    if (!WavWriter::writeWavHeaderAndData(in_wav, plug->captured_buffer, static_cast<int>(plug->sample_rate))) {
        return false;
    }

    std::ostringstream cmd;
    cmd << "earshot restore \"" << in_wav << "\" -o \"" << out_wav << "\" --engine " << plug->engine;
    int rc = std::system(cmd.str().c_str());
    if (rc != 0) return false;

    int sr = 0;
    if (WavWriter::readWavData(out_wav, plug->restored_buffer, sr)) {
        plug->playback_pos = 0;
        plug->playing_restored.store(true);
        return true;
    }
    return false;
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
            if (start) {
                plug->captured_buffer.clear();
                plug->capturing.store(true);
                plug->playing_restored.store(false);
            } else {
                plug->capturing.store(false);
            }
        } else if (ev->param_id == PARAM_RESTORE) {
            if (ev->value >= 0.5) {
                bridge_trigger_restore(plug);
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
}

static const clap_plugin_params_t s_bridge_params = {
    bridge_params_count,
    bridge_params_get_info,
    bridge_params_get_value,
    bridge_params_value_to_text,
    bridge_params_text_to_value,
    bridge_params_flush
};

/////////////////////////////////
// ARA2 plugin extension stubs //
/////////////////////////////////

static ARAFactoryPtr bridge_ara_get_factory(const clap_plugin_t *plugin) {
    return nullptr;
}

static ARAPlugInExtensionInstancePtr bridge_ara_bind_to_document_controller(
    const clap_plugin_t *plugin, ARADocumentControllerRef documentControllerRef,
    ARAPlugInInstanceRoleFlags knownRoles, ARAPlugInInstanceRoleFlags assignedRoles) {
    return nullptr;
}

static const clap_ara_plugin_extension_t s_bridge_ara_extension = {
    bridge_ara_get_factory,
    bridge_ara_bind_to_document_controller
};

static bool bridge_init(const struct clap_plugin *plugin) {
    auto *plug = (earshot_bridge_plug_t *)plugin->plugin_data;
    plug->channels = 1;
    plug->sample_rate = 48000.0;
    return true;
}

static void bridge_destroy(const struct clap_plugin *plugin) {
    auto *plug = (earshot_bridge_plug_t *)plugin->plugin_data;
    delete plug;
}

static bool bridge_activate(const struct clap_plugin *plugin, double sample_rate,
                            uint32_t min_frames_count, uint32_t max_frames_count) {
    auto *plug = (earshot_bridge_plug_t *)plugin->plugin_data;
    plug->sample_rate = sample_rate;
    return true;
}

static void bridge_deactivate(const struct clap_plugin *plugin) {}

static bool bridge_start_processing(const struct clap_plugin *plugin) {
    return true;
}

static void bridge_stop_processing(const struct clap_plugin *plugin) {}

static void bridge_reset(const struct clap_plugin *plugin) {}

static clap_process_status bridge_process(const struct clap_plugin *plugin,
                                          const clap_process_t *process) {
    auto *plug = (earshot_bridge_plug_t *)plugin->plugin_data;

    // Process parameter events
    if (process->in_events) {
        uint32_t nev = process->in_events->size(process->in_events);
        for (uint32_t i = 0; i < nev; ++i) {
            const auto *hdr = process->in_events->get(process->in_events, i);
            bridge_handle_param_event(plug, hdr);
        }
    }

    if (process->audio_inputs_count == 0 || process->audio_outputs_count == 0) {
        return CLAP_PROCESS_CONTINUE;
    }
    const uint32_t nframes = process->frames_count;
    const auto &in_buf = process->audio_inputs[0];
    auto &out_buf = process->audio_outputs[0];

    uint32_t channels = (in_buf.channel_count < out_buf.channel_count)
                            ? in_buf.channel_count
                            : out_buf.channel_count;

    bool is_capturing = plug->capturing.load();
    bool is_playing_restored = plug->playing_restored.load();

    for (uint32_t c = 0; c < channels; ++c) {
        if (in_buf.data32 && out_buf.data32) {
            if (is_playing_restored && !plug->restored_buffer.empty()) {
                for (uint32_t f = 0; f < nframes; ++f) {
                    size_t p = plug->playback_pos + f;
                    if (p < plug->restored_buffer.size()) {
                        out_buf.data32[c][f] = plug->restored_buffer[p];
                    } else {
                        out_buf.data32[c][f] = in_buf.data32[c][f];
                    }
                }
            } else {
                if (in_buf.data32[c] != out_buf.data32[c]) {
                    memcpy(out_buf.data32[c], in_buf.data32[c], nframes * sizeof(float));
                }
            }
        }
    }

    if (is_capturing && in_buf.data32 && in_buf.channel_count > 0) {
        for (uint32_t f = 0; f < nframes; ++f) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < in_buf.channel_count; ++c) {
                sum += in_buf.data32[c][f];
            }
            plug->captured_buffer.push_back(sum / in_buf.channel_count);
        }
    }

    if (is_playing_restored) {
        plug->playback_pos += nframes;
        if (plug->playback_pos >= plug->restored_buffer.size()) {
            plug->playing_restored.store(false);
            plug->playback_pos = 0;
        }
    }

    return CLAP_PROCESS_CONTINUE;
}

static const void *bridge_get_extension(const struct clap_plugin *plugin, const char *id) {
    if (!strcmp(id, CLAP_EXT_AUDIO_PORTS)) return &s_bridge_audio_ports;
    if (!strcmp(id, CLAP_EXT_CONFIGURABLE_AUDIO_PORTS)) return &s_bridge_configurable_audio_ports;
    if (!strcmp(id, CLAP_EXT_PARAMS)) return &s_bridge_params;
    if (!strcmp(id, CLAP_EXT_ARA_PLUGINEXTENSION)) return &s_bridge_ara_extension;
    return nullptr;
}

static void bridge_on_main_thread(const struct clap_plugin *plugin) {}

clap_plugin_t *create_earshot_bridge_plugin(const clap_host_t *host) {
    auto *plug = new earshot_bridge_plug_t();
    plug->host = host;
    plug->channels = 1;
    plug->sample_rate = 48000.0;
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

//////////////////////////////////////
// ARA2 factory extension at entry  //
//////////////////////////////////////

static uint32_t bridge_ara_factory_get_count(const struct clap_ara_factory *factory) {
    return 1;
}

static ARAFactoryPtr bridge_ara_factory_get_factory(const struct clap_ara_factory *factory, uint32_t index) {
    return nullptr;
}

static const char *bridge_ara_factory_get_plugin_id(const struct clap_ara_factory *factory, uint32_t index) {
    return s_bridge_desc.id;
}

static const clap_ara_factory_t s_bridge_ara_factory = {
    bridge_ara_factory_get_count,
    bridge_ara_factory_get_factory,
    bridge_ara_factory_get_plugin_id
};

bool earshot_bridge_entry_init(const char *plugin_path) {
    return true;
}

void earshot_bridge_entry_deinit(void) {}

const void *earshot_bridge_entry_get_factory(const char *factory_id) {
    if (!strcmp(factory_id, CLAP_PLUGIN_FACTORY_ID)) return &s_bridge_factory;
    if (!strcmp(factory_id, CLAP_EXT_ARA_FACTORY)) return &s_bridge_ara_factory;
    return nullptr;
}
