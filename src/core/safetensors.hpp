// safetensors reader.
//
// File layout (https://github.com/huggingface/safetensors):
//   [8 bytes]  little-endian u64: header length N
//   [N bytes]  JSON header: { "tensor.name": { "dtype": "BF16",
//                "shape": [151936, 896], "data_offsets": [begin, end] }, ... }
//   [rest]     raw tensor bytes; data_offsets are relative to this section
//
// The file is mmap'd; tensor data is returned as views into the mapping.
// Every offset is bounds-checked against the file before any view is handed out.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/dtype.hpp"
#include "core/mmap_file.hpp"

namespace nano {

struct TensorInfo {
    std::string name;
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
    size_t data_begin = 0;  // absolute byte offset into the file
    size_t data_end = 0;

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) {
            n *= d;
        }
        return n;
    }
    size_t nbytes() const { return data_end - data_begin; }
};

class SafeTensors {
public:
    explicit SafeTensors(const std::string& path);  // throws on any malformed input

    /// Tensors in header order.
    const std::vector<TensorInfo>& tensors() const { return tensors_; }

    /// Returns nullptr if absent.
    const TensorInfo* find(std::string_view name) const;
    /// Throws with the tensor name if absent.
    const TensorInfo& require(std::string_view name) const;

    /// Raw bytes of a tensor, as stored (view into the mapped file).
    std::span<const std::byte> raw(const TensorInfo& t) const;

    /// Tensor data widened to fp32 (F32 copied; BF16/F16 converted).
    std::vector<float> to_f32(const TensorInfo& t) const;

    int64_t total_params() const;
    size_t file_size() const { return file_.size(); }

private:
    MmapFile file_;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, size_t> index_;  // name -> tensors_ position
};

}  // namespace nano
