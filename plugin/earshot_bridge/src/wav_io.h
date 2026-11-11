#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>
#include <string>
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
        return bool(out);
    }

    static bool readWavData(const std::string& path, std::vector<float>& monoData, int& sampleRate) {
        std::ifstream in(path, std::ios::binary);
        const auto u16 = [](const uint8_t* p) { return uint16_t(p[0] | (uint16_t(p[1]) << 8)); };
        const auto u32 = [](const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24); };
        uint8_t header[12];
        if (!in.read(reinterpret_cast<char*>(header), 12) || std::memcmp(header, "RIFF", 4) || std::memcmp(header + 8, "WAVE", 4)) return false;
        in.seekg(0, std::ios::end);
        const auto fileEnd = in.tellg();
        const uint64_t riffEnd = uint64_t(u32(header + 4)) + 8;
        if (fileEnd < 12 || riffEnd < 12 || riffEnd > uint64_t(fileEnd)) return false;
        in.seekg(12);
        uint16_t format = 0, channels = 0, bits = 0, align = 0;
        uint32_t rate = 0;
        while (uint64_t(in.tellg()) + 8 <= riffEnd) {
            uint8_t chunk[8];
            if (!in.read(reinterpret_cast<char*>(chunk), 8)) return false;
            const uint32_t size = u32(chunk + 4);
            const auto start = in.tellg();
            const uint64_t end = uint64_t(start) + size;
            if (end > riffEnd) return false;
            if (!std::memcmp(chunk, "fmt ", 4)) {
                uint8_t fmt[16];
                if (size < 16 || !in.read(reinterpret_cast<char*>(fmt), 16)) return false;
                format = u16(fmt); channels = u16(fmt + 2); rate = u32(fmt + 4);
                align = u16(fmt + 12); bits = u16(fmt + 14);
            } else if (!std::memcmp(chunk, "data", 4)) {
                const bool pcm = format == 1 && (bits == 16 || bits == 24 || bits == 32);
                const bool ieee = format == 3 && bits == 32;
                if ((!pcm && !ieee) || !channels || !rate || rate > uint32_t(std::numeric_limits<int>::max())) return false;
                const size_t bytes = bits / 8;
                if (align != channels * bytes || size == 0 || size % align != 0) return false;
                // Bound allocation before reading any sender-controlled payload.
                if (size > 256u * 1024u * 1024u) return false;
                std::vector<uint8_t> data(size);
                if (!in.read(reinterpret_cast<char*>(data.data()), size)) return false;
                const size_t frames = size / align;
                std::vector<float> decoded(frames);
                for (size_t f = 0; f < frames; ++f) {
                    double sum = 0;
                    for (size_t c = 0; c < channels; ++c) {
                        const auto* p = data.data() + f * align + c * bytes;
                        float v;
                        if (ieee) {
                            const uint32_t raw = u32(p);
                            std::memcpy(&v, &raw, sizeof(v));
                            if (!std::isfinite(v)) return false;
                        } else {
                            uint32_t raw = bits == 16 ? u16(p) : bits == 24 ? uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 : u32(p);
                            const int64_t signedValue = (raw & (uint32_t(1) << (bits - 1))) ? int64_t(raw) - (int64_t(1) << bits) : int64_t(raw);
                            v = float(double(signedValue) / double(uint64_t(1) << (bits - 1)));
                        }
                        sum += v;
                    }
                    decoded[f] = float(sum / channels);
                }
                monoData.swap(decoded);
                sampleRate = int(rate);
                return true;
            }
            in.seekg(static_cast<std::streamoff>(end + (size & 1)));
            if (!in) return false;
        }
        return false;
    }
};
