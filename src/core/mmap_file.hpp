// Read-only memory-mapped file (RAII). The safetensors loader maps the whole
// checkpoint and hands out views into it — the OS pages weights in on demand,
// which is how real engines avoid a full copy of the model at load time.
#pragma once

#include <cstddef>
#include <string>

namespace nano {

class MmapFile {
public:
    explicit MmapFile(const std::string& path);  // throws std::runtime_error
    ~MmapFile();

    MmapFile(MmapFile&& other) noexcept;
    MmapFile& operator=(MmapFile&& other) noexcept;
    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;

    const std::byte* data() const { return data_; }
    size_t size() const { return size_; }

private:
    void release() noexcept;

    const std::byte* data_ = nullptr;
    size_t size_ = 0;
};

}  // namespace nano
