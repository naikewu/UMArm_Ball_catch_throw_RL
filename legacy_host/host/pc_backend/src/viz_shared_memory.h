#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace vnema {

constexpr int64_t kVizSchemaVersion = 2;
constexpr std::size_t kVizJointCount = 12;
constexpr std::size_t kVizActuatorCount = 24;
constexpr std::size_t kVizUJointCount = 6;
constexpr std::size_t kVizRigidBodyCount = 6;

struct VizSharedSample {
    int64_t cycle = 0;
    int64_t mocap_frame = 0;
    int64_t mocap_valid = 0;
    int64_t mocap_stale = 1;
    int64_t joint_current_valid = 0;
    int64_t joint_fixed_delay_valid = 0;
    int64_t fk_valid = 0;
    int64_t mocap_body_mask = 0;
    int64_t observer_over_budget_count = 0;
    int64_t fk_over_budget_count = 0;
    int64_t publish_over_budget_count = 0;
    int64_t base_frame_valid = 0;
    double cycle_start_time_s = 0.0;
    double can_sync_time_s = 0.0;
    double mocap_timestamp_s = 0.0;
    double mocap_latency_ms = 0.0;
    double mocap_frame_rate_hz = 0.0;
    double observer_time_ms = 0.0;
    double observer_time_max_ms = 0.0;
    double observer_budget_ms = 0.0;
    double fk_time_ms = 0.0;
    double fk_time_max_ms = 0.0;
    double fk_budget_ms = 0.0;
    double publish_time_ms = 0.0;
    double publish_time_max_ms = 0.0;
    double publish_budget_ms = 0.0;
    std::array<double, kVizJointCount> q{};
    std::array<double, kVizJointCount> qdot{};
    std::array<double, kVizUJointCount * 3> fk_centers{};
    std::array<double, 3> fk_tip{};
    std::array<double, kVizRigidBodyCount * 3> mocap_centers{};
    std::array<double, kVizActuatorCount> pressure{};
    std::array<double, kVizActuatorCount> target{};
    std::array<double, 16> mocap_from_fk{};
    std::array<double, 16> reserved_f64{};
};

struct VizSharedMemoryLayout {
    int64_t schema_version = kVizSchemaVersion;
    int64_t layout_size = 0;
    int64_t seqlock = 0;
    int64_t writer_pid = 0;
    int64_t publish_count = 0;
    std::array<int64_t, 11> reserved_i64{};
    VizSharedSample sample{};
};

class VizSharedMemoryWriter {
public:
    VizSharedMemoryWriter() = default;
    ~VizSharedMemoryWriter();

    VizSharedMemoryWriter(const VizSharedMemoryWriter&) = delete;
    VizSharedMemoryWriter& operator=(const VizSharedMemoryWriter&) = delete;

    bool start(const std::string& name, std::string& error);
    void close();
    bool enabled() const { return layout_ != nullptr; }
    const std::string& name() const { return name_; }
    uint64_t publish_count() const { return publish_count_; }
    bool publish(const VizSharedSample& sample, std::string& error);

private:
#ifdef _WIN32
    void* mapping_handle_ = nullptr;
#endif
    VizSharedMemoryLayout* layout_ = nullptr;
    std::string name_;
    uint64_t publish_count_ = 0;
};

} // namespace vnema
