#include "core/mmap_file.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace nano {

MmapFile::MmapFile(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
    }

    struct stat st = {};
    if (::fstat(fd, &st) != 0 || st.st_size < 0) {
        int err = errno;
        ::close(fd);
        throw std::runtime_error("cannot stat " + path + ": " + std::strerror(err));
    }
    size_ = static_cast<size_t>(st.st_size);

    if (size_ == 0) {
        // mmap of length 0 is invalid; an empty file maps to an empty view.
        ::close(fd);
        return;
    }

    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);  // the mapping keeps the file alive
    if (p == MAP_FAILED) {
        throw std::runtime_error("cannot mmap " + path + ": " + std::strerror(errno));
    }
    data_ = static_cast<const std::byte*>(p);
}

MmapFile::~MmapFile() {
    release();
}

MmapFile::MmapFile(MmapFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
    if (this != &other) {
        release();
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

void MmapFile::release() noexcept {
    if (data_ != nullptr) {
        ::munmap(const_cast<std::byte*>(data_), size_);
        data_ = nullptr;
        size_ = 0;
    }
}

}  // namespace nano
