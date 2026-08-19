#include "core/safetensors.hpp"

#include <cstring>
#include <stdexcept>

#include "core/json.hpp"

namespace nano {

namespace {

[[noreturn]] void malformed(const std::string& path, const std::string& why) {
    throw std::runtime_error("safetensors: " + path + ": " + why);
}

}  // namespace

SafeTensors::SafeTensors(const std::string& path) : file_(path) {
    if (file_.size() < 8) {
        malformed(path, "file too small for header length");
    }

    uint64_t header_len = 0;
    std::memcpy(&header_len, file_.data(), 8);  // format is little-endian, as are our targets
    if (header_len > file_.size() - 8) {
        malformed(path, "header length exceeds file size");
    }

    std::string_view header_text(reinterpret_cast<const char*>(file_.data()) + 8,
                                 static_cast<size_t>(header_len));
    json::Value header;
    try {
        header = json::parse(header_text);
    } catch (const std::exception& e) {
        malformed(path, std::string("bad header JSON: ") + e.what());
    }
    if (!header.is_object()) {
        malformed(path, "header is not a JSON object");
    }

    const size_t data_base = 8 + static_cast<size_t>(header_len);
    const size_t data_size = file_.size() - data_base;

    for (const auto& [name, entry] : header.members()) {
        if (name == "__metadata__") {
            continue;
        }
        if (!entry.is_object()) {
            malformed(path, "tensor entry '" + name + "' is not an object");
        }

        TensorInfo info;
        info.name = name;
        info.dtype = dtype_from_string(entry.at("dtype").as_string());

        for (const auto& dim : entry.at("shape").items()) {
            int64_t d = dim.as_int();
            if (d < 0) {
                malformed(path, "negative dimension in '" + name + "'");
            }
            info.shape.push_back(d);
        }

        const auto& offsets = entry.at("data_offsets").items();
        if (offsets.size() != 2) {
            malformed(path, "data_offsets of '" + name + "' must have 2 entries");
        }
        int64_t begin = offsets[0].as_int();
        int64_t end = offsets[1].as_int();
        if (begin < 0 || end < begin || static_cast<uint64_t>(end) > data_size) {
            malformed(path, "data_offsets of '" + name + "' out of bounds");
        }
        info.data_begin = data_base + static_cast<size_t>(begin);
        info.data_end = data_base + static_cast<size_t>(end);

        const uint64_t expected = static_cast<uint64_t>(info.numel()) * dtype_size(info.dtype);
        if (expected != info.nbytes()) {
            malformed(path, "size mismatch for '" + name + "': shape implies " +
                                std::to_string(expected) + " bytes, offsets give " +
                                std::to_string(info.nbytes()));
        }

        index_.emplace(info.name, tensors_.size());
        tensors_.push_back(std::move(info));
    }
}

const TensorInfo* SafeTensors::find(std::string_view name) const {
    auto it = index_.find(std::string(name));
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

const TensorInfo& SafeTensors::require(std::string_view name) const {
    const TensorInfo* t = find(name);
    if (t == nullptr) {
        throw std::runtime_error("safetensors: tensor not found: " + std::string(name));
    }
    return *t;
}

std::span<const std::byte> SafeTensors::raw(const TensorInfo& t) const {
    return {file_.data() + t.data_begin, t.nbytes()};
}

std::vector<float> SafeTensors::to_f32(const TensorInfo& t) const {
    const std::span<const std::byte> bytes = raw(t);
    const size_t n = static_cast<size_t>(t.numel());
    std::vector<float> out(n);

    switch (t.dtype) {
        case DType::F32:
            std::memcpy(out.data(), bytes.data(), bytes.size());
            break;
        case DType::BF16:
        case DType::F16: {
            const bool is_bf16 = t.dtype == DType::BF16;
            for (size_t i = 0; i < n; ++i) {
                uint16_t bits = 0;
                std::memcpy(&bits, bytes.data() + i * 2, 2);
                out[i] = is_bf16 ? bf16_to_f32(bits) : f16_to_f32(bits);
            }
            break;
        }
        default:
            throw std::runtime_error("to_f32: unsupported dtype " +
                                     std::string(dtype_name(t.dtype)));
    }
    return out;
}

int64_t SafeTensors::total_params() const {
    int64_t total = 0;
    for (const TensorInfo& t : tensors_) {
        total += t.numel();
    }
    return total;
}

}  // namespace nano
