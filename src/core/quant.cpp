#include "core/quant.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace nano {

QuantMatrix quantize_rows(const float* w, int64_t rows, int64_t cols) {
    QuantMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.q.resize(static_cast<size_t>(rows * cols));
    m.scales.resize(static_cast<size_t>(rows));

    for (int64_t r = 0; r < rows; ++r) {
        const float* row = w + r * cols;
        float absmax = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            absmax = std::max(absmax, std::fabs(row[c]));
        }
        const float scale = absmax / 127.0f;
        m.scales[static_cast<size_t>(r)] = scale;

        int8_t* out = m.q.data() + r * cols;
        if (scale == 0.0f) {  // all-zero row: avoid 0/0
            for (int64_t c = 0; c < cols; ++c) {
                out[c] = 0;
            }
            continue;
        }
        const float inv_scale = 127.0f / absmax;
        for (int64_t c = 0; c < cols; ++c) {
            // lrintf = round to nearest (ties to even, the default FP mode).
            // |row[c]| <= absmax, so the result is already within [-127, 127];
            // the clamp guards the absmax value itself landing on 127.00001f.
            const long v = std::lrintf(row[c] * inv_scale);
            out[c] = static_cast<int8_t>(std::min<long>(127, std::max<long>(-127, v)));
        }
    }
    return m;
}

void write_safetensors(const std::string& path, const std::vector<SaveTensor>& tensors,
                       const std::vector<std::pair<std::string, std::string>>& metadata) {
    // Build the JSON header by hand — every name and metadata string we emit
    // is plain ASCII with no quotes/backslashes, checked here so a future
    // caller can't silently produce a malformed file.
    const auto check_plain = [](const std::string& s) {
        for (char ch : s) {
            if (ch == '"' || ch == '\\' || static_cast<unsigned char>(ch) < 0x20) {
                throw std::runtime_error("write_safetensors: string needs escaping: " + s);
            }
        }
    };

    std::string header = "{";
    if (!metadata.empty()) {
        header += "\"__metadata__\":{";
        for (size_t i = 0; i < metadata.size(); ++i) {
            check_plain(metadata[i].first);
            check_plain(metadata[i].second);
            if (i > 0) {
                header += ",";
            }
            header += "\"" + metadata[i].first + "\":\"" + metadata[i].second + "\"";
        }
        header += "},";
    }

    size_t offset = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const SaveTensor& t = tensors[i];
        check_plain(t.name);
        int64_t numel = 1;
        for (int64_t d : t.shape) {
            numel *= d;
        }
        if (static_cast<size_t>(numel) * dtype_size(t.dtype) != t.nbytes) {
            throw std::runtime_error("write_safetensors: shape/bytes mismatch for " + t.name);
        }
        if (i > 0) {
            header += ",";
        }
        header += "\"" + t.name + "\":{\"dtype\":\"" + std::string(dtype_name(t.dtype)) +
                  "\",\"shape\":[";
        for (size_t d = 0; d < t.shape.size(); ++d) {
            if (d > 0) {
                header += ",";
            }
            header += std::to_string(t.shape[d]);
        }
        header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
                  std::to_string(offset + t.nbytes) + "]}";
        offset += t.nbytes;
    }
    header += "}";

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        throw std::runtime_error("write_safetensors: cannot open " + path);
    }
    const auto fail = [&](const std::string& why) {
        std::fclose(f);
        throw std::runtime_error("write_safetensors: " + path + ": " + why);
    };

    const uint64_t header_len = header.size();
    if (std::fwrite(&header_len, 8, 1, f) != 1 ||
        std::fwrite(header.data(), 1, header.size(), f) != header.size()) {
        fail("short write (header)");
    }
    for (const SaveTensor& t : tensors) {
        if (t.nbytes > 0 && std::fwrite(t.data, 1, t.nbytes, f) != t.nbytes) {
            fail("short write (tensor " + t.name + ")");
        }
    }
    if (std::fclose(f) != 0) {
        throw std::runtime_error("write_safetensors: close failed for " + path);
    }
}

}  // namespace nano
