#include "viz_shared_memory.h"

#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace vnema {

VizSharedMemoryWriter::~VizSharedMemoryWriter()
{
    close();
}

bool VizSharedMemoryWriter::start(const std::string& name, std::string& error)
{
    close();
    name_ = name;
#ifdef _WIN32
    if (name_.empty()) {
        error = "visualization shared-memory name is empty";
        return false;
    }
    mapping_handle_ = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, static_cast<DWORD>(sizeof(VizSharedMemoryLayout)), name_.c_str());
    if (mapping_handle_ == nullptr) {
        error = "CreateFileMappingA failed for visualization shared memory (Win32 error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    layout_ = static_cast<VizSharedMemoryLayout*>(MapViewOfFile(mapping_handle_, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(VizSharedMemoryLayout)));
    if (layout_ == nullptr) {
        error = "MapViewOfFile failed for visualization shared memory (Win32 error " + std::to_string(GetLastError()) + ")";
        CloseHandle(static_cast<HANDLE>(mapping_handle_));
        mapping_handle_ = nullptr;
        return false;
    }
    *layout_ = VizSharedMemoryLayout{};
    layout_->schema_version = kVizSchemaVersion;
    layout_->layout_size = static_cast<int64_t>(sizeof(VizSharedMemoryLayout));
    layout_->seqlock = 0;
    layout_->writer_pid = static_cast<int64_t>(GetCurrentProcessId());
    layout_->publish_count = 0;
    publish_count_ = 0;
    return true;
#else
    (void)name;
    error = "visualization shared memory is currently implemented for Windows only";
    return false;
#endif
}

void VizSharedMemoryWriter::close()
{
#ifdef _WIN32
    if (layout_ != nullptr) {
        UnmapViewOfFile(layout_);
        layout_ = nullptr;
    }
    if (mapping_handle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(mapping_handle_));
        mapping_handle_ = nullptr;
    }
#else
    layout_ = nullptr;
#endif
}

bool VizSharedMemoryWriter::publish(const VizSharedSample& sample, std::string& error)
{
    (void)error;
    if (layout_ == nullptr) return true;
#ifdef _WIN32
    int64_t sequence = layout_->seqlock;
    if ((sequence & 1) != 0) ++sequence;
    layout_->seqlock = sequence + 1;
    MemoryBarrier();
    layout_->sample = sample;
    ++publish_count_;
    layout_->publish_count = static_cast<int64_t>(publish_count_);
    MemoryBarrier();
    layout_->seqlock = sequence + 2;
    return true;
#else
    (void)sample;
    error = "visualization shared memory is unavailable on this platform";
    return false;
#endif
}

} // namespace vnema
