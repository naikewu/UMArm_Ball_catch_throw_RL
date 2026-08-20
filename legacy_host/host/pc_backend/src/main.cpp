#include "umarm_fk.h"
#include "viz_shared_memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>
#endif

namespace vnema {

using Clock = std::chrono::steady_clock;

constexpr uint16_t kBroadcastSyncId = 0x090;
constexpr uint16_t kFirstDtId = 0x101;
constexpr uint16_t kLastDtId = 0x108;
constexpr uint16_t kFirstRuntimeId = 0x101;
constexpr uint16_t kLastRuntimeId = 0x118;
constexpr uint16_t kRuntimeTableBaseId = 0x091;
constexpr uint16_t kPressureMask = 0x0FFF;
constexpr uint16_t kFlagMask = 0x000F;
constexpr size_t kRigidBodyCount = 6;
constexpr size_t kJointCount = 12;
static_assert(kJointCount == kFkJointCount, "backend joint count must match FK module");
static_assert(kJointCount == kVizJointCount, "backend joint count must match visualization layout");
static_assert(kRigidBodyCount == kVizRigidBodyCount, "backend rigid body count must match visualization layout");
constexpr uint16_t kMocapBaseRigidId = 1000;
constexpr double kPi = 3.14159265358979323846;
constexpr double kFixedDelayStateS = 0.008;
constexpr double kMaxMocapExtrapolationS = 0.012;
constexpr double kObserverBudgetMs = 0.5;
constexpr double kFkBudgetMs = 0.25;
constexpr double kVizPublishBudgetMs = 0.25;
constexpr uint8_t kRuntimeTableMarker = 0x80;
constexpr uint8_t kRuntimeTableSlotMask = 0x1F;
constexpr size_t kRuntimeTableSlots = 3;
constexpr uint8_t kControlEnable = 0x01;
constexpr uint8_t kStatusEnabled = 0x01;
constexpr uint8_t kStatusOtaActive = 0x02;
constexpr uint8_t kStatusCommandSeen = 0x04;
constexpr uint8_t kStatusError = 0x08;
constexpr double kCollectionCheckpointS = 60.0;
constexpr double kCollectionDeflateS = 2.0;
constexpr double kCollectionSlewPsiPerS = 15.0;
constexpr double kCollectionAdcRangePsi = 40.0;
constexpr double kCollectionSingleLimitPsi = 30.0;
constexpr double kCollectionPairSumLimitPsi = 35.0;
constexpr double kCollectionSingleLimitNormalized = kCollectionSingleLimitPsi / kCollectionAdcRangePsi;
constexpr double kCollectionPairSumLimitNormalized = kCollectionPairSumLimitPsi / kCollectionAdcRangePsi;

struct ActuatorAdcRange {
    uint16_t id = 0;
    uint16_t min_adc = 0;
    uint16_t max_adc = kPressureMask;
};

ActuatorAdcRange default_collection_adc_range(uint16_t id);

struct CollectionChunkInfo {
    std::string path;
    std::string reason;
    uint64_t samples = 0;
    uint64_t start_cycle = 0;
    uint64_t end_cycle = 0;
    double start_time_s = 0.0;
    double end_time_s = 0.0;
};

struct CanFrame {
    uint16_t id = 0;
    std::vector<uint8_t> data;
};

struct TransportStats {
    uint64_t tx_frames = 0;
    uint64_t rx_frames = 0;
    uint64_t parse_errors = 0;
    uint64_t send_errors = 0;
};

struct BoardState {
    uint16_t id = 0;
    uint16_t target = 0;
    uint8_t flags = 0;
    uint16_t pressure = 0;
    uint16_t pressure_filtered = 0;
    double pressure_calibrated = 0.0;
    uint8_t status = 0;
    uint64_t status_errors = 0;
    bool stale = true;
    uint64_t missed = 0;
    uint64_t replies = 0;
    double latency_ms = 0.0;
    double latency_sum_ms = 0.0;
    double latency_max_ms = 0.0;
    Clock::time_point last_reply{};
};

struct BoardCommand {
    uint16_t id = 0;
    uint16_t target = 0;
    uint8_t flags = 0;
};

enum class RuntimeProtocol {
    Auto,
    Broadcast,
    Unicast,
};

enum class CanOrder {
    Normal,
    Reverse,
    Rotate,
};

struct MocapSample {
    bool valid = false;
    bool stale = true;
    uint64_t frame = 0;
    double timestamp_s = 0.0;
    double raw_timestamp_s = 0.0;
    double received_s = 0.0;
    double age_ms = 0.0;
    double latency_ms = 0.0;
    double timestamp_offset_ms = 0.0;
    double frame_rate_hz = 0.0;
    uint64_t frame_drop_count = 0;
    size_t clock_sample_count = 0;
    uint64_t clock_update_count = 0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
    int body_count = 0;
    std::string body_ids;
    std::string body_points;
    std::string body_poses;
    std::array<double, kRigidBodyCount * 3> body_centers{};
    std::array<double, kRigidBodyCount * 9> body_rotations{};
    std::array<double, kRigidBodyCount * 4> body_quaternions{};
    uint32_t body_mask = 0;
};

struct JointState {
    std::array<double, kJointCount> theta{};
    std::array<double, kJointCount> theta_dot{};
    bool valid = false;
    bool extrapolated = false;
    double estimate_time_s = 0.0;
    double source_time_error_ms = 0.0;
    double extrapolation_ms = 0.0;
};

struct RobotStateSample {
    uint64_t cycle = 0;
    double cycle_start_time_s = 0.0;
    double can_sync_time_s = 0.0;
    std::vector<uint16_t> ids;
    std::vector<uint16_t> pressure_adc_filtered;
    std::vector<double> pressure_calibrated;
    std::vector<uint8_t> actuator_status;
    std::vector<bool> actuator_stale;
    std::vector<uint16_t> target_next_sync;
    std::vector<uint8_t> control_next_sync;
    JointState joint_current_estimate;
    JointState joint_fixed_delay;
    FkResult fk_current;
    bool fk_valid = false;
    std::array<double, kRigidBodyCount * 3> mocap_body_centers{};
    std::array<double, kRigidBodyCount * 9> mocap_body_rotations{};
    uint32_t mocap_body_mask = 0;
    FkTransform mocap_from_fk{};
    bool mocap_base_frame_valid = false;
    uint64_t mocap_frame = 0;
    double mocap_timestamp_s = 0.0;
    double mocap_raw_timestamp_s = 0.0;
    double mocap_timestamp_offset_ms = 0.0;
    double mocap_frame_rate_hz = 0.0;
    uint64_t mocap_frame_drop_count = 0;
    size_t mocap_clock_sample_count = 0;
    uint64_t mocap_clock_update_count = 0;
    double mocap_queue_age_ms = 0.0;
    double mocap_extrapolation_ms = 0.0;
    bool mocap_stale = true;
    bool calibration_default = true;
    int cycle_responded = 0;
    int cycle_expected = 0;
    double observer_time_ms = 0.0;
    double observer_time_max_ms = 0.0;
    uint64_t observer_over_budget_count = 0;
    double fk_time_ms = 0.0;
    double fk_time_max_ms = 0.0;
    uint64_t fk_over_budget_count = 0;
    double viz_publish_time_ms = 0.0;
    double viz_publish_time_max_ms = 0.0;
    uint64_t viz_publish_over_budget_count = 0;
};

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

using Matrix3 = std::array<std::array<double, 3>, 3>;

struct RigidBodyPose {
    Vec3 position;
    Matrix3 rotation{};
    std::array<double, 4> quaternion{0.0, 0.0, 0.0, 1.0};
    bool valid = false;
};

struct MocapKinematicSample {
    uint64_t frame = 0;
    double timestamp_s = 0.0;
    double raw_timestamp_s = 0.0;
    double received_s = 0.0;
    std::array<double, kJointCount> theta{};
    bool valid = false;
};

struct Options {
    std::string port = "COM4";
    int tty_baud = 2000000;
    int rate_hz = 150;
    bool simulate_can = false;
    bool mocap_sim = false;
    bool mocap_live = false;
    bool mocap_multicast = true;
    bool viz_enable = false;
    bool self_test = false;
    bool status_only = false;
    RuntimeProtocol runtime_protocol = RuntimeProtocol::Auto;
    CanOrder can_order = CanOrder::Normal;
    double rx_window_frac = 0.82;
    int duration_s = 0;
    std::string mocap_server = "127.0.0.1";
    std::string mocap_local = "127.0.0.1";
    std::string mocap_rigid_ids = "1000-1005";
    std::string mocap_python = "python";
    std::string mocap_script = "mocap\\mocap.py";
    std::string viz_shm_name = "vnema_viz";
    bool stream_state_every_cycle = false;
    std::filesystem::path calibration_path;
    std::filesystem::path log_dir = std::filesystem::path("reports");
    std::vector<uint16_t> ids;
};

class PressureCalibration {
public:
    void set_path(std::filesystem::path path)
    {
        ranges_.clear();
        loaded_ = false;
        path_.clear();
        const bool explicit_path = !path.empty();
        std::filesystem::path candidate = explicit_path ? std::move(path) : (std::filesystem::current_path() / "calibration.json");
        if (candidate.is_relative()) candidate = std::filesystem::current_path() / candidate;
        if (!std::filesystem::exists(candidate)) {
            if (explicit_path) throw std::runtime_error("pressure calibration file not found: " + candidate.string());
            source_ = "builtin-0-40psi-adc-endpoints";
            return;
        }
        load_from_file(candidate);
    }

    double apply(uint16_t id, uint16_t raw_adc) const
    {
        const ActuatorAdcRange range = range_for(id);
        const double span = std::max(1.0, static_cast<double>(range.max_adc) - static_cast<double>(range.min_adc));
        return (static_cast<double>(raw_adc) - static_cast<double>(range.min_adc)) * kCollectionAdcRangePsi / span;
    }

    ActuatorAdcRange range_for(uint16_t id) const
    {
        const auto it = ranges_.find(id);
        return it != ranges_.end() ? it->second : default_collection_adc_range(id);
    }

    bool using_defaults() const { return !loaded_; }

    std::string source() const
    {
        return source_;
    }

private:
    static void skip_ws(const std::string& text, size_t& pos)
    {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) pos++;
    }

    static std::string parse_json_string(const std::string& text, size_t& pos)
    {
        skip_ws(text, pos);
        if (pos >= text.size() || text[pos] != '"') throw std::runtime_error("expected JSON string in adc_ranges");
        pos++;
        std::string value;
        while (pos < text.size()) {
            const char ch = text[pos++];
            if (ch == '"') return value;
            if (ch == '\\') {
                if (pos >= text.size()) throw std::runtime_error("unterminated escape in calibration JSON");
                value.push_back(text[pos++]);
            } else {
                value.push_back(ch);
            }
        }
        throw std::runtime_error("unterminated JSON string in calibration JSON");
    }

    static double parse_json_number(const std::string& text, size_t& pos)
    {
        skip_ws(text, pos);
        const char* begin = text.c_str() + pos;
        char* end = nullptr;
        const double value = std::strtod(begin, &end);
        if (end == begin || !std::isfinite(value)) throw std::runtime_error("expected finite number in adc_ranges");
        pos = static_cast<size_t>(end - text.c_str());
        return value;
    }

    static void expect_char(const std::string& text, size_t& pos, char expected)
    {
        skip_ws(text, pos);
        if (pos >= text.size() || text[pos] != expected) {
            throw std::runtime_error(std::string("expected '") + expected + "' in adc_ranges");
        }
        pos++;
    }

    static uint16_t parse_id_key(const std::string& key)
    {
        const int base = key.rfind("0x", 0) == 0 || key.rfind("0X", 0) == 0 ? 16 : 10;
        char* end = nullptr;
        const unsigned long value = std::strtoul(key.c_str(), &end, base);
        if (end == key.c_str() || *end != '\0' || value > 0x7FFu) throw std::runtime_error("invalid actuator ID in calibration: " + key);
        return static_cast<uint16_t>(value);
    }

    static uint16_t adc_endpoint(double value)
    {
        const long rounded = std::lround(value);
        if (rounded < 0) return 0;
        if (rounded > kPressureMask) return kPressureMask;
        return static_cast<uint16_t>(rounded);
    }

    static size_t matching_brace(const std::string& text, size_t open_pos)
    {
        int depth = 0;
        bool in_string = false;
        bool escaped = false;
        for (size_t pos = open_pos; pos < text.size(); ++pos) {
            const char ch = text[pos];
            if (in_string) {
                if (escaped) {
                    escaped = false;
                } else if (ch == '\\') {
                    escaped = true;
                } else if (ch == '"') {
                    in_string = false;
                }
                continue;
            }
            if (ch == '"') {
                in_string = true;
            } else if (ch == '{') {
                depth++;
            } else if (ch == '}') {
                depth--;
                if (depth == 0) return pos;
            }
        }
        throw std::runtime_error("unterminated adc_ranges object in calibration JSON");
    }

    void load_from_file(const std::filesystem::path& path)
    {
        std::ifstream input(path);
        if (!input.is_open()) throw std::runtime_error("failed to open pressure calibration file: " + path.string());
        const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        const size_t key_pos = text.find("\"adc_ranges\"");
        if (key_pos == std::string::npos) throw std::runtime_error("pressure calibration missing adc_ranges: " + path.string());
        const size_t object_start = text.find('{', key_pos);
        if (object_start == std::string::npos) throw std::runtime_error("pressure calibration adc_ranges is not an object: " + path.string());
        const size_t object_end = matching_brace(text, object_start);
        size_t pos = object_start + 1;
        std::map<uint16_t, ActuatorAdcRange> parsed;
        while (pos < object_end) {
            skip_ws(text, pos);
            if (pos < object_end && text[pos] == ',') {
                pos++;
                continue;
            }
            if (pos >= object_end) break;
            const std::string key = parse_json_string(text, pos);
            expect_char(text, pos, ':');
            expect_char(text, pos, '[');
            const double min_value = parse_json_number(text, pos);
            expect_char(text, pos, ',');
            const double max_value = parse_json_number(text, pos);
            expect_char(text, pos, ']');
            const uint16_t id = parse_id_key(key);
            const uint16_t min_adc = adc_endpoint(min_value);
            const uint16_t max_adc = adc_endpoint(max_value);
            if (max_adc <= min_adc) throw std::runtime_error("invalid adc range for " + key + " in " + path.string());
            parsed[id] = ActuatorAdcRange{id, min_adc, max_adc};
        }
        if (parsed.empty()) throw std::runtime_error("pressure calibration adc_ranges is empty: " + path.string());
        ranges_ = std::move(parsed);
        path_ = path;
        loaded_ = true;
        source_ = "calibration-json:" + path_.string();
    }

    std::filesystem::path path_;
    std::map<uint16_t, ActuatorAdcRange> ranges_;
    bool loaded_ = false;
    std::string source_ = "builtin-0-40psi-adc-endpoints";
};

double now_seconds()
{
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

double time_point_seconds(Clock::time_point time_point)
{
    return std::chrono::duration<double>(time_point.time_since_epoch()).count();
}

Matrix3 identity_matrix()
{
    return {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
}

bool finite(double value)
{
    return std::isfinite(value);
}

bool finite(const Vec3& value)
{
    return finite(value.x) && finite(value.y) && finite(value.z);
}

Vec3 add(Vec3 a, Vec3 b)
{
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 subtract(Vec3 a, Vec3 b)
{
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 scale(Vec3 value, double factor)
{
    return Vec3{value.x * factor, value.y * factor, value.z * factor};
}

double dot(Vec3 a, Vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(Vec3 a, Vec3 b)
{
    return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

double norm(Vec3 value)
{
    return std::sqrt(dot(value, value));
}

bool normalize(Vec3 input, Vec3& output)
{
    if (!finite(input)) return false;
    const double length = norm(input);
    if (!finite(length) || length < 1e-10) return false;
    output = scale(input, 1.0 / length);
    return true;
}

Vec3 matrix_vector(const Matrix3& matrix, Vec3 value)
{
    return Vec3{matrix[0][0] * value.x + matrix[0][1] * value.y + matrix[0][2] * value.z,
                matrix[1][0] * value.x + matrix[1][1] * value.y + matrix[1][2] * value.z,
                matrix[2][0] * value.x + matrix[2][1] * value.y + matrix[2][2] * value.z};
}

Vec3 transpose_matrix_vector(const Matrix3& matrix, Vec3 value)
{
    return Vec3{matrix[0][0] * value.x + matrix[1][0] * value.y + matrix[2][0] * value.z,
                matrix[0][1] * value.x + matrix[1][1] * value.y + matrix[2][1] * value.z,
                matrix[0][2] * value.x + matrix[1][2] * value.y + matrix[2][2] * value.z};
}

Matrix3 matrix_multiply(const Matrix3& left, const Matrix3& right)
{
    Matrix3 out{};
    for (size_t row = 0; row < 3; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            out[row][col] = left[row][0] * right[0][col] + left[row][1] * right[1][col] + left[row][2] * right[2][col];
        }
    }
    return out;
}

bool normalize_quaternion(double qx, double qy, double qz, double qw, std::array<double, 4>& out)
{
    if (!finite(qx) || !finite(qy) || !finite(qz) || !finite(qw)) return false;
    const double length = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (!finite(length) || length < 1e-10) return false;
    out = {qx / length, qy / length, qz / length, qw / length};
    return true;
}

bool quaternion_to_matrix(const std::array<double, 4>& quaternion, Matrix3& out)
{
    const double qx = quaternion[0];
    const double qy = quaternion[1];
    const double qz = quaternion[2];
    const double qw = quaternion[3];
    if (!finite(qx) || !finite(qy) || !finite(qz) || !finite(qw)) return false;

    const double xx = qx * qx;
    const double yy = qy * qy;
    const double zz = qz * qz;
    const double xy = qx * qy;
    const double xz = qx * qz;
    const double yz = qy * qz;
    const double xw = qx * qw;
    const double yw = qy * qw;
    const double zw = qz * qw;

    out = {{{1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw)},
            {2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)},
            {2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)}}};
    return true;
}

bool quaternion_to_matrix(double qx, double qy, double qz, double qw, Matrix3& out)
{
    std::array<double, 4> quaternion{};
    return normalize_quaternion(qx, qy, qz, qw, quaternion) && quaternion_to_matrix(quaternion, out);
}

std::array<double, 4> interpolate_quaternion(const std::array<double, 4>& a, const std::array<double, 4>& b, double alpha)
{
    double dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    std::array<double, 4> adjusted_b = b;
    if (dot_product < 0.0) {
        for (double& value : adjusted_b) value = -value;
    }
    std::array<double, 4> out{};
    for (size_t index = 0; index < out.size(); ++index) out[index] = a[index] + (adjusted_b[index] - a[index]) * alpha;
    std::array<double, 4> normalized{};
    return normalize_quaternion(out[0], out[1], out[2], out[3], normalized) ? normalized : b;
}

FkTransform identity_fk_transform()
{
    FkTransform out{};
    out[0] = 1.0;
    out[5] = 1.0;
    out[10] = 1.0;
    out[15] = 1.0;
    return out;
}

bool finite_transform(const FkTransform& transform)
{
    return std::all_of(transform.begin(), transform.end(), [](double value) { return finite(value); });
}

FkVec3 transform_point(const FkTransform& transform, FkVec3 point)
{
    return FkVec3{transform[0] * point.x + transform[1] * point.y + transform[2] * point.z + transform[3],
                  transform[4] * point.x + transform[5] * point.y + transform[6] * point.z + transform[7],
                  transform[8] * point.x + transform[9] * point.y + transform[10] * point.z + transform[11]};
}

bool make_mocap_from_fk_transform(const FkResult& fk, const MocapSample& mocap, FkTransform& out)
{
    out = identity_fk_transform();
    if (!fk.valid || (mocap.body_mask & 1u) == 0) return false;
    const FkTransform& fk_base = fk.ujoint_transforms[0];
    if (!finite_transform(fk_base)) return false;

    Matrix3 mocap_rotation{};
    for (size_t row = 0; row < 3; ++row) {
        for (size_t col = 0; col < 3; ++col) mocap_rotation[row][col] = mocap.body_rotations[row * 3 + col];
    }
    const FkVec3 fk_origin{fk_base[3], fk_base[7], fk_base[11]};
    const FkVec3 mocap_origin{mocap.body_centers[0], mocap.body_centers[1], mocap.body_centers[2]};
    if (!finite(mocap_origin.x) || !finite(mocap_origin.y) || !finite(mocap_origin.z)) return false;

    for (size_t row = 0; row < 3; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            out[4 * row + col] = mocap_rotation[row][0] * fk_base[4 * col] + mocap_rotation[row][1] * fk_base[4 * col + 1] +
                                 mocap_rotation[row][2] * fk_base[4 * col + 2];
        }
    }
    const FkVec3 rotated_origin{out[0] * fk_origin.x + out[1] * fk_origin.y + out[2] * fk_origin.z,
                                out[4] * fk_origin.x + out[5] * fk_origin.y + out[6] * fk_origin.z,
                                out[8] * fk_origin.x + out[9] * fk_origin.y + out[10] * fk_origin.z};
    out[3] = mocap_origin.x - rotated_origin.x;
    out[7] = mocap_origin.y - rotated_origin.y;
    out[11] = mocap_origin.z - rotated_origin.z;
    out[12] = 0.0;
    out[13] = 0.0;
    out[14] = 0.0;
    out[15] = 1.0;
    return finite_transform(out);
}

bool rotation_from_to(Vec3 from, Vec3 to, Matrix3& out)
{
    Vec3 source;
    Vec3 target;
    if (!normalize(from, source) || !normalize(to, target)) return false;
    const double c = std::max(-1.0, std::min(1.0, dot(source, target)));
    if (c > 1.0 - 1e-12) {
        out = identity_matrix();
        return true;
    }
    if (c < -1.0 + 1e-12) {
        Vec3 axis = cross(source, Vec3{1.0, 0.0, 0.0});
        if (!normalize(axis, axis)) axis = cross(source, Vec3{0.0, 1.0, 0.0});
        if (!normalize(axis, axis)) return false;
        out = {{{2.0 * axis.x * axis.x - 1.0, 2.0 * axis.x * axis.y, 2.0 * axis.x * axis.z},
                {2.0 * axis.y * axis.x, 2.0 * axis.y * axis.y - 1.0, 2.0 * axis.y * axis.z},
                {2.0 * axis.z * axis.x, 2.0 * axis.z * axis.y, 2.0 * axis.z * axis.z - 1.0}}};
        return true;
    }

    const Vec3 v = cross(source, target);
    const double s2 = dot(v, v);
    if (s2 < 1e-20) return false;
    const Matrix3 skew = {{{0.0, -v.z, v.y}, {v.z, 0.0, -v.x}, {-v.y, v.x, 0.0}}};
    const Matrix3 skew2 = matrix_multiply(skew, skew);
    const double factor = (1.0 - c) / s2;
    const Matrix3 identity = identity_matrix();
    for (size_t row = 0; row < 3; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            out[row][col] = identity[row][col] + skew[row][col] + skew2[row][col] * factor;
        }
    }
    return true;
}

Vec3 rotate_z_negative_45(Vec3 value)
{
    const double c = std::sqrt(0.5);
    const double s = -std::sqrt(0.5);
    return Vec3{c * value.x - s * value.y, s * value.x + c * value.y, value.z};
}

bool normalize_in_base(const Matrix3& rot_base, Vec3 spatial, Vec3& robot)
{
    return normalize(transpose_matrix_vector(rot_base, spatial), robot);
}

bool angle_pair(Vec3 vector, double& theta1, double& theta2)
{
    if (!finite(vector)) return false;
    theta1 = std::atan2(vector.z, vector.y) - kPi * 0.5;
    theta2 = -1.0 * (std::atan2(std::sqrt(vector.z * vector.z + vector.y * vector.y), vector.x) - kPi * 0.5);
    return finite(theta1) && finite(theta2);
}

Vec3 z_axis(const Matrix3& rotation)
{
    return Vec3{rotation[0][2], rotation[1][2], rotation[2][2]};
}

bool mocap_poses_to_q_mk8(const std::array<RigidBodyPose, kRigidBodyCount>& poses, uint32_t body_mask, std::array<double, kJointCount>& q)
{
    constexpr uint32_t required_mask = (1u << kRigidBodyCount) - 1u;
    if ((body_mask & required_mask) != required_mask) return false;
    for (const RigidBodyPose& pose : poses) {
        if (!pose.valid || !finite(pose.position)) return false;
    }

    const Matrix3& rot_base = poses[0].rotation;
    const Vec3 v_base{0.0, 0.0, 1.0};
    Vec3 rv1;
    Vec3 rv2;
    Vec3 rv3;
    Vec3 rv4;
    Vec3 rv5;
    Vec3 rv6;
    if (!normalize_in_base(rot_base, subtract(poses[0].position, poses[1].position), rv1)) return false;
    if (!normalize_in_base(rot_base, subtract(poses[1].position, poses[2].position), rv2)) return false;
    if (!normalize_in_base(rot_base, subtract(poses[2].position, poses[3].position), rv3)) return false;
    if (!normalize_in_base(rot_base, subtract(poses[3].position, poses[4].position), rv4)) return false;
    if (!normalize_in_base(rot_base, subtract(poses[4].position, poses[5].position), rv5)) return false;
    if (!normalize_in_base(rot_base, z_axis(poses[5].rotation), rv6)) return false;

    size_t index = 0;
    if (!angle_pair(rv1, q[index], q[index + 1])) return false;
    index += 2;

    Matrix3 to_z;
    if (!rotation_from_to(rv1, v_base, to_z)) return false;
    Vec3 transformed = rotate_z_negative_45(matrix_vector(to_z, rv2));
    if (!angle_pair(transformed, q[index], q[index + 1])) return false;
    index += 2;

    if (!rotation_from_to(rv2, v_base, to_z)) return false;
    transformed = matrix_vector(to_z, rv3);
    if (!angle_pair(transformed, q[index], q[index + 1])) return false;
    index += 2;

    if (!rotation_from_to(rv3, v_base, to_z)) return false;
    transformed = rotate_z_negative_45(matrix_vector(to_z, rv4));
    if (!angle_pair(transformed, q[index], q[index + 1])) return false;
    index += 2;

    if (!rotation_from_to(rv4, v_base, to_z)) return false;
    transformed = matrix_vector(to_z, rv5);
    if (!angle_pair(transformed, q[index], q[index + 1])) return false;
    index += 2;

    if (!rotation_from_to(rv5, v_base, to_z)) return false;
    transformed = rotate_z_negative_45(matrix_vector(to_z, rv6));
    if (!angle_pair(transformed, q[index], q[index + 1])) return false;

    return std::all_of(q.begin(), q.end(), [](double value) { return std::isfinite(value); });
}

std::string json_escape(const std::string& input)
{
    std::ostringstream out;
    for (char c : input) {
        switch (c) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << c; break;
        }
    }
    return out.str();
}

std::string id_hex(uint16_t id)
{
    std::ostringstream out;
    out << "0x" << std::uppercase << std::hex << std::setw(3) << std::setfill('0') << id;
    return out.str();
}

std::string format_number(double value, int precision)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

std::string json_u16_array(const std::vector<uint16_t>& values)
{
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ',';
        out << values[i];
    }
    out << ']';
    return out.str();
}

std::string json_u8_array(const std::vector<uint8_t>& values)
{
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ',';
        out << static_cast<int>(values[i]);
    }
    out << ']';
    return out.str();
}

std::string json_bool_array(const std::vector<bool>& values)
{
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ',';
        out << (values[i] ? "true" : "false");
    }
    out << ']';
    return out.str();
}

std::string json_double_array(const std::vector<double>& values, int precision)
{
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ',';
        out << format_number(values[i], precision);
    }
    out << ']';
    return out.str();
}

std::string json_joint_array(const std::array<double, kJointCount>& values, int precision)
{
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ',';
        out << format_number(values[i], precision);
    }
    out << ']';
    return out.str();
}

template <size_t N>
std::string json_double_array(const std::array<double, N>& values, int precision)
{
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ',';
        out << format_number(values[i], precision);
    }
    out << ']';
    return out.str();
}

std::string json_u64_array(const std::vector<uint64_t>& values)
{
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ',';
        out << values[i];
    }
    out << ']';
    return out.str();
}

std::string json_string_array(const std::vector<std::string>& values)
{
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ',';
        out << '"' << json_escape(values[i]) << '"';
    }
    out << ']';
    return out.str();
}

double unix_seconds_now()
{
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

ActuatorAdcRange default_collection_adc_range(uint16_t id)
{
    switch (id) {
    case 0x101: return {id, 714, 2865};
    case 0x102: return {id, 686, 2755};
    case 0x103: return {id, 718, 2915};
    case 0x104: return {id, 708, 2835};
    case 0x105: return {id, 708, 2860};
    case 0x106: return {id, 700, 2840};
    case 0x107: return {id, 692, 2785};
    case 0x108: return {id, 696, 2785};
    case 0x109: return {id, 725, 2956};
    case 0x10A: return {id, 726, 2980};
    case 0x10B: return {id, 740, 2991};
    case 0x10C: return {id, 663, 2655};
    case 0x10D: return {id, 734, 3005};
    case 0x10E: return {id, 711, 2920};
    case 0x10F: return {id, 731, 2975};
    case 0x110: return {id, 731, 2984};
    case 0x111: return {id, 715, 2928};
    case 0x112: return {id, 703, 2870};
    case 0x113: return {id, 713, 2913};
    case 0x114: return {id, 714, 2898};
    case 0x115: return {id, 725, 2958};
    case 0x116: return {id, 717, 2926};
    case 0x117: return {id, 711, 2890};
    case 0x118: return {id, 740, 3010};
    default: return {id, 0, kPressureMask};
    }
}

std::vector<std::pair<uint16_t, uint16_t>> collection_actuator_pairs()
{
    return {
        {0x102, 0x106}, {0x104, 0x108}, {0x103, 0x107}, {0x105, 0x101},
        {0x10A, 0x10C}, {0x109, 0x10B}, {0x110, 0x10E}, {0x10D, 0x10F},
        {0x114, 0x112}, {0x111, 0x113}, {0x116, 0x118}, {0x115, 0x117},
    };
}

uint16_t clamp_pressure(int value)
{
    if (value < 0) return 0;
    if (value > static_cast<int>(kPressureMask)) return kPressureMask;
    return static_cast<uint16_t>(value);
}

uint16_t pack_compact(uint16_t pressure, uint8_t flags)
{
    return static_cast<uint16_t>((pressure & kPressureMask) | ((static_cast<uint16_t>(flags) & kFlagMask) << 12));
}

void unpack_compact(uint16_t payload, uint16_t& pressure, uint8_t& flags)
{
    pressure = payload & kPressureMask;
    flags = static_cast<uint8_t>((payload >> 12) & kFlagMask);
}

CanFrame compact_command_frame(uint16_t id, uint16_t target, uint8_t flags)
{
    const uint16_t payload = pack_compact(target, flags);
    return CanFrame{id, {static_cast<uint8_t>(payload & 0xFF), static_cast<uint8_t>((payload >> 8) & 0xFF)}};
}

bool runtime_broadcast_id_supported(uint16_t id)
{
    return id >= kFirstRuntimeId && id <= kLastRuntimeId;
}

bool all_runtime_broadcast_ids_supported(const std::vector<uint16_t>& ids)
{
    return std::all_of(ids.begin(), ids.end(), runtime_broadcast_id_supported);
}

std::string runtime_protocol_name(RuntimeProtocol protocol)
{
    switch (protocol) {
    case RuntimeProtocol::Auto: return "auto";
    case RuntimeProtocol::Broadcast: return "broadcast";
    case RuntimeProtocol::Unicast: return "unicast";
    }
    return "unknown";
}

std::string can_order_name(CanOrder order)
{
    switch (order) {
    case CanOrder::Normal: return "normal";
    case CanOrder::Reverse: return "reverse";
    case CanOrder::Rotate: return "rotate";
    }
    return "unknown";
}

RuntimeProtocol parse_runtime_protocol(const std::string& raw)
{
    if (raw == "auto") return RuntimeProtocol::Auto;
    if (raw == "broadcast") return RuntimeProtocol::Broadcast;
    if (raw == "unicast") return RuntimeProtocol::Unicast;
    throw std::runtime_error("invalid CAN protocol: " + raw);
}

CanOrder parse_can_order(const std::string& raw)
{
    if (raw == "normal") return CanOrder::Normal;
    if (raw == "reverse") return CanOrder::Reverse;
    if (raw == "rotate") return CanOrder::Rotate;
    throw std::runtime_error("invalid CAN order: " + raw);
}

double clamp_rx_window_frac(double value)
{
    if (!std::isfinite(value)) return 0.82;
    return std::max(0.10, std::min(0.98, value));
}

void apply_can_order(std::vector<BoardCommand>& commands, CanOrder order, uint64_t cycle)
{
    if (commands.empty()) return;
    if (order == CanOrder::Reverse) {
        std::reverse(commands.begin(), commands.end());
    } else if (order == CanOrder::Rotate) {
        const auto offset = static_cast<std::vector<BoardCommand>::difference_type>(cycle % commands.size());
        std::rotate(commands.begin(), commands.begin() + offset, commands.end());
    }
}

uint16_t runtime_table_frame_id_for_start_slot(size_t start_slot)
{
    return static_cast<uint16_t>(kRuntimeTableBaseId + (start_slot / kRuntimeTableSlots));
}

bool runtime_table_frame_id_supported(uint16_t id)
{
    const uint16_t last_group_id = runtime_table_frame_id_for_start_slot(static_cast<size_t>(kLastRuntimeId - kFirstRuntimeId));
    return id >= kRuntimeTableBaseId && id <= last_group_id;
}

std::vector<CanFrame> compact_broadcast_command_frames(const std::vector<BoardCommand>& commands)
{
    constexpr size_t slot_count = static_cast<size_t>(kLastRuntimeId - kFirstRuntimeId + 1);
    std::array<uint16_t, slot_count> payloads{};
    std::array<bool, slot_count> active{};

    for (const BoardCommand& command : commands) {
        if (!runtime_broadcast_id_supported(command.id)) {
            throw std::runtime_error("CAN ID cannot use broadcast runtime table: " + id_hex(command.id));
        }
        const size_t slot = static_cast<size_t>(command.id - kFirstRuntimeId);
        payloads[slot] = pack_compact(command.target, command.flags);
        active[slot] = true;
    }

    std::vector<CanFrame> frames;
    for (size_t start_slot = 0; start_slot < slot_count; start_slot += kRuntimeTableSlots) {
        uint8_t mask = 0;
        std::vector<uint8_t> data(8, 0);
        data[0] = static_cast<uint8_t>(kRuntimeTableMarker | (start_slot & kRuntimeTableSlotMask));
        for (size_t offset = 0; offset < kRuntimeTableSlots && start_slot + offset < slot_count; ++offset) {
            if (!active[start_slot + offset]) continue;
            mask |= static_cast<uint8_t>(1u << offset);
            const uint16_t payload = payloads[start_slot + offset];
            const size_t index = 2 + offset * 2;
            data[index] = static_cast<uint8_t>(payload & 0x00FFu);
            data[index + 1] = static_cast<uint8_t>((payload >> 8) & 0x00FFu);
        }
        if (mask != 0) {
            data[1] = mask;
            frames.push_back(CanFrame{runtime_table_frame_id_for_start_slot(start_slot), data});
        }
    }
    return frames;
}

bool compact_response_from_frame(const CanFrame& frame, uint16_t& pressure, uint8_t& status)
{
    if (frame.data.size() != 2) return false;
    const uint16_t payload = static_cast<uint16_t>(frame.data[0]) | (static_cast<uint16_t>(frame.data[1]) << 8);
    unpack_compact(payload, pressure, status);
    return true;
}

int parse_int_auto(const std::string& text)
{
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 0);
    if (end == text.c_str() || *end != '\0') {
        throw std::runtime_error("invalid integer: " + text);
    }
    return static_cast<int>(value);
}

std::vector<std::string> split(const std::string& text, char delimiter)
{
    std::vector<std::string> parts;
    std::string current;
    std::istringstream in(text);
    while (std::getline(in, current, delimiter)) {
        if (!current.empty()) parts.push_back(current);
    }
    return parts;
}

std::vector<uint16_t> default_dt_ids()
{
    std::vector<uint16_t> ids;
    for (uint16_t id = kFirstDtId; id <= kLastDtId; ++id) ids.push_back(id);
    return ids;
}

std::vector<uint16_t> parse_id_list(const std::string& raw)
{
    std::vector<uint16_t> ids;
    for (const auto& token : split(raw, ',')) {
        const auto dash = token.find('-');
        if (dash != std::string::npos) {
            const int first = parse_int_auto(token.substr(0, dash));
            const int last = parse_int_auto(token.substr(dash + 1));
            if (first > last) throw std::runtime_error("ID range is reversed: " + token);
            for (int id = first; id <= last; ++id) ids.push_back(static_cast<uint16_t>(id));
        } else if (token == "dt" || token == "DT") {
            auto defaults = default_dt_ids();
            ids.insert(ids.end(), defaults.begin(), defaults.end());
        } else {
            ids.push_back(static_cast<uint16_t>(parse_int_auto(token)));
        }
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    for (uint16_t id : ids) {
        if (id > 0x7FF) throw std::runtime_error("standard CAN ID out of range: " + id_hex(id));
    }
    return ids;
}

bool parse_double_token(const std::string& text, double& value)
{
    char* end = nullptr;
    value = std::strtod(text.c_str(), &end);
    return end != text.c_str() && *end == '\0' && std::isfinite(value);
}

bool parse_body_poses(const std::string& raw, std::array<RigidBodyPose, kRigidBodyCount>& poses, uint32_t& body_mask)
{
    poses = {};
    body_mask = 0;
    if (raw.empty()) return false;
    for (const std::string& item : split(raw, '|')) {
        const std::vector<std::string> fields = split(item, ':');
        if (fields.size() != 8) continue;
        int body_id = 0;
        try {
            body_id = parse_int_auto(fields[0]);
        } catch (const std::exception&) {
            continue;
        }
        const int index = body_id - static_cast<int>(kMocapBaseRigidId);
        if (index < 0 || index >= static_cast<int>(kRigidBodyCount)) continue;

        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double qx = 0.0;
        double qy = 0.0;
        double qz = 0.0;
        double qw = 1.0;
        if (!parse_double_token(fields[1], x) || !parse_double_token(fields[2], y) || !parse_double_token(fields[3], z) ||
            !parse_double_token(fields[4], qx) || !parse_double_token(fields[5], qy) || !parse_double_token(fields[6], qz) ||
            !parse_double_token(fields[7], qw)) {
            continue;
        }
        std::array<double, 4> quaternion{};
        Matrix3 rotation;
        if (!normalize_quaternion(qx, qy, qz, qw, quaternion) || !quaternion_to_matrix(quaternion, rotation)) continue;

        RigidBodyPose pose;
        pose.position = Vec3{x, y, z};
        pose.rotation = rotation;
        pose.quaternion = quaternion;
        pose.valid = true;
        poses[static_cast<size_t>(index)] = pose;
        body_mask |= 1u << static_cast<uint32_t>(index);
    }
    return body_mask != 0;
}

double wrap_angle(double angle)
{
    while (angle > kPi) angle -= 2.0 * kPi;
    while (angle <= -kPi) angle += 2.0 * kPi;
    return angle;
}

JointState joint_state_from_kinematic_samples(const std::deque<MocapKinematicSample>& samples, double target_time_s)
{
    JointState state;
    state.estimate_time_s = target_time_s;
    if (samples.size() < 2 || !std::isfinite(target_time_s)) return state;

    size_t upper = 0;
    while (upper < samples.size() && samples[upper].timestamp_s < target_time_s) ++upper;

    size_t lower = 0;
    bool extrapolated = false;
    if (upper == 0) {
        if (std::abs(samples[0].timestamp_s - target_time_s) > 1e-9) {
            state.source_time_error_ms = std::abs(samples[0].timestamp_s - target_time_s) * 1000.0;
            return state;
        }
        lower = 0;
        upper = 1;
    } else if (upper >= samples.size()) {
        const double extrapolation_s = target_time_s - samples.back().timestamp_s;
        if (extrapolation_s < 0.0 || extrapolation_s > kMaxMocapExtrapolationS) {
            state.source_time_error_ms = std::abs(extrapolation_s) * 1000.0;
            state.extrapolation_ms = std::max(0.0, extrapolation_s * 1000.0);
            return state;
        }
        lower = samples.size() - 2;
        upper = samples.size() - 1;
        extrapolated = true;
        state.extrapolation_ms = extrapolation_s * 1000.0;
    } else {
        lower = upper - 1;
    }

    const MocapKinematicSample& a = samples[lower];
    const MocapKinematicSample& b = samples[upper];
    const double span = b.timestamp_s - a.timestamp_s;
    if (!a.valid || !b.valid || span <= 1e-9 || !std::isfinite(span)) return state;
    const double alpha = (target_time_s - a.timestamp_s) / span;

    for (size_t index = 0; index < kJointCount; ++index) {
        const double delta = wrap_angle(b.theta[index] - a.theta[index]);
        state.theta[index] = wrap_angle(a.theta[index] + delta * alpha);
        state.theta_dot[index] = delta / span;
        if (!std::isfinite(state.theta[index]) || !std::isfinite(state.theta_dot[index])) return JointState{};
    }
    state.valid = true;
    state.extrapolated = extrapolated;
    state.source_time_error_ms = extrapolated ? state.extrapolation_ms
                                              : std::min(std::abs(target_time_s - a.timestamp_s), std::abs(b.timestamp_s - target_time_s)) * 1000.0;
    return state;
}

std::optional<std::string> json_string_value(const std::string& line, const std::string& key)
{
    const std::string needle = "\"" + key + "\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return std::nullopt;
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos) return std::nullopt;
    ++pos;
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) ++pos;
    if (pos >= line.size() || line[pos] != '"') return std::nullopt;
    ++pos;
    std::string value;
    bool escape = false;
    for (; pos < line.size(); ++pos) {
        char c = line[pos];
        if (escape) {
            value.push_back(c);
            escape = false;
        } else if (c == '\\') {
            escape = true;
        } else if (c == '"') {
            return value;
        } else {
            value.push_back(c);
        }
    }
    return std::nullopt;
}

std::optional<int> json_int_value(const std::string& line, const std::string& key)
{
    const std::string needle = "\"" + key + "\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return std::nullopt;
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos) return std::nullopt;
    ++pos;
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) ++pos;
    size_t end = pos;
    while (end < line.size() && (std::isdigit(static_cast<unsigned char>(line[end])) || line[end] == '-' || line[end] == '+')) ++end;
    if (end == pos) return std::nullopt;
    return parse_int_auto(line.substr(pos, end - pos));
}

std::optional<double> json_double_value(const std::string& line, const std::string& key)
{
    const std::string needle = "\"" + key + "\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return std::nullopt;
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos) return std::nullopt;
    ++pos;
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) ++pos;
    size_t end = pos;
    while (end < line.size()) {
        const char c = line[end];
        if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')) break;
        ++end;
    }
    if (end == pos) return std::nullopt;
    char* parse_end = nullptr;
    const double value = std::strtod(line.substr(pos, end - pos).c_str(), &parse_end);
    if (parse_end == nullptr || *parse_end != '\0') return std::nullopt;
    return value;
}

std::optional<bool> json_bool_value(const std::string& line, const std::string& key)
{
    const std::string needle = "\"" + key + "\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return std::nullopt;
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos) return std::nullopt;
    ++pos;
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) ++pos;
    if (line.compare(pos, 4, "true") == 0) return true;
    if (line.compare(pos, 5, "false") == 0) return false;
    return std::nullopt;
}

// Parse a flat JSON integer array for the named key, e.g. "targets":[1500,800,0].
// Returns nullopt if the key is missing or the value is not a well-formed array.
std::optional<std::vector<int>> json_int_array_value(const std::string& line, const std::string& key)
{
    const std::string needle = "\"" + key + "\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return std::nullopt;
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos) return std::nullopt;
    ++pos;
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) ++pos;
    if (pos >= line.size() || line[pos] != '[') return std::nullopt;
    ++pos;
    std::vector<int> values;
    while (pos < line.size()) {
        while (pos < line.size() && (std::isspace(static_cast<unsigned char>(line[pos])) || line[pos] == ',')) ++pos;
        if (pos < line.size() && line[pos] == ']') return values;
        size_t end = pos;
        while (end < line.size() && (std::isdigit(static_cast<unsigned char>(line[end])) || line[end] == '-' || line[end] == '+')) ++end;
        if (end == pos) return std::nullopt;
        values.push_back(parse_int_auto(line.substr(pos, end - pos)));
        pos = end;
    }
    return std::nullopt;
}

std::string encode_slcan(const CanFrame& frame)
{
    if (frame.id > 0x7FF || frame.data.size() > 8) throw std::runtime_error("invalid CAN frame for SLCAN encode");
    std::ostringstream out;
    out << 't' << std::uppercase << std::hex << std::setw(3) << std::setfill('0') << frame.id;
    out << std::dec << frame.data.size();
    for (uint8_t byte : frame.data) {
        out << std::uppercase << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
    }
    return out.str();
}

bool decode_hex_byte(const std::string& text, size_t pos, uint8_t& value)
{
    if (pos + 2 > text.size()) return false;
    char buf[3] = {text[pos], text[pos + 1], '\0'};
    char* end = nullptr;
    long parsed = std::strtol(buf, &end, 16);
    if (end == buf || *end != '\0' || parsed < 0 || parsed > 255) return false;
    value = static_cast<uint8_t>(parsed);
    return true;
}

bool decode_slcan(const std::string& line, CanFrame& frame)
{
    if (line.size() < 5 || line[0] != 't') return false;
    char id_buf[4] = {line[1], line[2], line[3], '\0'};
    char* id_end = nullptr;
    long id = std::strtol(id_buf, &id_end, 16);
    if (id_end == id_buf || *id_end != '\0' || id < 0 || id > 0x7FF) return false;
    if (!std::isdigit(static_cast<unsigned char>(line[4]))) return false;
    int dlc = line[4] - '0';
    if (dlc < 0 || dlc > 8) return false;
    if (line.size() < static_cast<size_t>(5 + dlc * 2)) return false;
    frame.id = static_cast<uint16_t>(id);
    frame.data.clear();
    for (int i = 0; i < dlc; ++i) {
        uint8_t byte = 0;
        if (!decode_hex_byte(line, static_cast<size_t>(5 + i * 2), byte)) return false;
        frame.data.push_back(byte);
    }
    return true;
}

class JsonEmitter {
public:
    void emit(const std::string& json)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << json << std::endl;
    }

private:
    std::mutex mutex_;
};

class ICanTransport {
public:
    virtual ~ICanTransport() = default;
    virtual bool open(std::string& error) = 0;
    virtual void close() = 0;
    virtual bool send_frame(const CanFrame& frame, std::string& error) = 0;
    virtual bool recv_frame(CanFrame& frame, int timeout_ms) = 0;
    virtual TransportStats stats() const = 0;
};

class FakeTransport : public ICanTransport {
public:
    bool open(std::string&) override
    {
        open_ = true;
        return true;
    }

    void close() override
    {
        open_ = false;
    }

    bool send_frame(const CanFrame& frame, std::string&) override
    {
        if (!open_) return false;
        stats_.tx_frames++;
        if (frame.id == kBroadcastSyncId && frame.data.empty()) {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const BoardCommand& command : staged_broadcast_) {
                pending_.push(fake_reply(command));
            }
            staged_broadcast_.clear();
        } else if (runtime_table_frame_id_supported(frame.id) && frame.data.size() == 8 && (frame.data[0] & kRuntimeTableMarker) != 0) {
            const size_t start_slot = static_cast<size_t>(frame.data[0] & kRuntimeTableSlotMask);
            const uint8_t mask = frame.data[1];
            std::lock_guard<std::mutex> lock(mutex_);
            for (size_t offset = 0; offset < kRuntimeTableSlots; ++offset) {
                if ((mask & (1u << offset)) == 0) continue;
                const uint16_t id = static_cast<uint16_t>(kFirstRuntimeId + start_slot + offset);
                if (!runtime_broadcast_id_supported(id)) continue;
                const size_t index = 2 + offset * 2;
                const uint16_t payload = static_cast<uint16_t>(frame.data[index]) | (static_cast<uint16_t>(frame.data[index + 1]) << 8);
                uint16_t target = 0;
                uint8_t flags = 0;
                unpack_compact(payload, target, flags);
                staged_broadcast_.push_back(BoardCommand{id, target, flags});
            }
        } else if (runtime_broadcast_id_supported(frame.id) && frame.data.size() == 2) {
            uint16_t target = 0;
            uint8_t flags = 0;
            compact_response_from_frame(frame, target, flags);
            std::lock_guard<std::mutex> lock(mutex_);
            pending_.push(fake_reply(BoardCommand{frame.id, target, flags}));
        }
        return true;
    }

    bool recv_frame(CanFrame& frame, int timeout_ms) override
    {
        const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
        while (Clock::now() <= deadline) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!pending_.empty()) {
                    frame = pending_.front();
                    pending_.pop();
                    stats_.rx_frames++;
                    return true;
                }
            }
            if (timeout_ms <= 1) return false;
            std::this_thread::yield();
        }
        return false;
    }

    TransportStats stats() const override { return stats_; }

private:
    CanFrame fake_reply(const BoardCommand& command) const
    {
        const double t = now_seconds();
        const uint16_t wave = static_cast<uint16_t>(1000 + ((static_cast<int>(command.id) - kFirstRuntimeId) * 30) + static_cast<int>(40.0 * std::sin(t * 2.0)));
        uint8_t status = kStatusCommandSeen;
        if ((command.flags & kControlEnable) != 0) status |= kStatusEnabled;
        return compact_command_frame(command.id, wave, status);
    }

    bool open_ = false;
    mutable std::mutex mutex_;
    std::queue<CanFrame> pending_;
    std::vector<BoardCommand> staged_broadcast_;
    TransportStats stats_;
};

#ifdef _WIN32
class WinSerial {
public:
    ~WinSerial() { close(); }

    bool open(const std::string& port, int baud, std::string& error)
    {
        close();
        const std::string path = "\\\\.\\" + port;
        handle_ = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) {
            error = "failed to open " + port + " (Win32 error " + std::to_string(GetLastError()) + ")";
            return false;
        }

        DCB dcb{};
        dcb.DCBlength = sizeof(dcb);
        if (!GetCommState(handle_, &dcb)) {
            error = "GetCommState failed";
            close();
            return false;
        }
        dcb.BaudRate = static_cast<DWORD>(baud);
        dcb.ByteSize = 8;
        dcb.Parity = NOPARITY;
        dcb.StopBits = ONESTOPBIT;
        dcb.fBinary = TRUE;
        dcb.fDtrControl = DTR_CONTROL_ENABLE;
        dcb.fRtsControl = RTS_CONTROL_ENABLE;
        if (!SetCommState(handle_, &dcb)) {
            error = "SetCommState failed for baud " + std::to_string(baud);
            close();
            return false;
        }

        COMMTIMEOUTS timeouts{};
        timeouts.ReadIntervalTimeout = MAXDWORD;
        timeouts.ReadTotalTimeoutMultiplier = 0;
        timeouts.ReadTotalTimeoutConstant = 0;
        timeouts.WriteTotalTimeoutMultiplier = 0;
        timeouts.WriteTotalTimeoutConstant = 100;
        SetCommTimeouts(handle_, &timeouts);
        SetupComm(handle_, 1 << 16, 1 << 16);
        PurgeComm(handle_, PURGE_RXCLEAR | PURGE_TXCLEAR);
        return true;
    }

    void close()
    {
        if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
    }

    bool write_line(const std::string& line)
    {
        if (handle_ == INVALID_HANDLE_VALUE) return false;
        std::string with_cr = line + "\r";
        DWORD written = 0;
        return WriteFile(handle_, with_cr.data(), static_cast<DWORD>(with_cr.size()), &written, nullptr) && written == with_cr.size();
    }

    bool read_char(char& c)
    {
        if (handle_ == INVALID_HANDLE_VALUE) return false;
        DWORD read = 0;
        if (!ReadFile(handle_, &c, 1, &read, nullptr)) return false;
        return read == 1;
    }

private:
    HANDLE handle_ = INVALID_HANDLE_VALUE;
};
#endif

class SlcanTransport : public ICanTransport {
public:
    SlcanTransport(std::string port, int baud) : port_(std::move(port)), baud_(baud) {}

    bool open(std::string& error) override
    {
#ifndef _WIN32
        error = "SLCAN serial transport is currently implemented for Windows only";
        return false;
#else
        if (!serial_.open(port_, baud_, error)) return false;
        serial_.write_line("C");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        serial_.write_line("S8");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        serial_.write_line("Z0");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        serial_.write_line("O");
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        return true;
#endif
    }

    void close() override
    {
#ifdef _WIN32
        serial_.write_line("C");
        serial_.close();
#endif
    }

    bool send_frame(const CanFrame& frame, std::string& error) override
    {
        try {
            const std::string line = encode_slcan(frame);
#ifdef _WIN32
            if (!serial_.write_line(line)) {
                stats_.send_errors++;
                error = "serial write failed";
                return false;
            }
#else
            (void)line;
            error = "SLCAN transport unavailable on this platform";
            return false;
#endif
            stats_.tx_frames++;
            return true;
        } catch (const std::exception& exc) {
            stats_.send_errors++;
            error = exc.what();
            return false;
        }
    }

    bool recv_frame(CanFrame& frame, int timeout_ms) override
    {
        const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
        while (Clock::now() < deadline) {
#ifdef _WIN32
            char c = '\0';
            if (!serial_.read_char(c)) {
                std::this_thread::yield();
                continue;
            }
            if (c == '\r' || c == '\n') {
                if (rx_line_.empty()) continue;
                CanFrame parsed;
                if (decode_slcan(rx_line_, parsed)) {
                    frame = parsed;
                    rx_line_.clear();
                    stats_.rx_frames++;
                    return true;
                }
                stats_.parse_errors++;
                rx_line_.clear();
                continue;
            }
            rx_line_.push_back(c);
            if (rx_line_.size() > 32) {
                stats_.parse_errors++;
                rx_line_.clear();
            }
#else
            std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
            return false;
#endif
        }
        return false;
    }

    TransportStats stats() const override { return stats_; }

private:
    std::string port_;
    int baud_ = 2000000;
    std::string rx_line_;
    TransportStats stats_;
#ifdef _WIN32
    WinSerial serial_;
#endif
};

class MocapSimulator {
public:
    void start(bool enabled)
    {
        if (!enabled || running_) return;
        running_ = true;
        thread_ = std::thread([this]() { run(); });
    }

    void stop()
    {
        running_ = false;
        if (thread_.joinable()) thread_.join();
    }

    MocapSample sample_at(double timestamp_s)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (samples_.size() < 2) return {};

        size_t upper = 0;
        while (upper < samples_.size() && samples_[upper].timestamp_s < timestamp_s) ++upper;
        if (upper == 0 || upper >= samples_.size()) {
            MocapSample latest = samples_.back();
            latest.valid = true;
            latest.stale = true;
            latest.raw_timestamp_s = latest.timestamp_s;
            latest.age_ms = std::abs(timestamp_s - latest.timestamp_s) * 1000.0;
            latest.latency_ms = latest.age_ms;
            latest.frame_rate_hz = 360.0;
            return latest;
        }

        const MocapSample& a = samples_[upper - 1];
        const MocapSample& b = samples_[upper];
        const double span = std::max(1e-9, b.timestamp_s - a.timestamp_s);
        const double alpha = (timestamp_s - a.timestamp_s) / span;
        MocapSample out;
        out.valid = true;
        out.stale = false;
        out.frame = b.frame;
        out.timestamp_s = timestamp_s;
        out.raw_timestamp_s = timestamp_s;
        out.received_s = timestamp_s;
        out.age_ms = std::min(std::abs(timestamp_s - a.timestamp_s), std::abs(b.timestamp_s - timestamp_s)) * 1000.0;
        out.latency_ms = out.age_ms;
        out.timestamp_offset_ms = 0.0;
        out.frame_rate_hz = 360.0;
        out.frame_drop_count = b.frame_drop_count;
        out.x = a.x + (b.x - a.x) * alpha;
        out.y = a.y + (b.y - a.y) * alpha;
        out.z = a.z + (b.z - a.z) * alpha;
        out.vx = (b.x - a.x) / span;
        out.vy = (b.y - a.y) / span;
        out.vz = (b.z - a.z) / span;
        out.body_count = b.body_count;
        out.body_ids = b.body_ids;
        out.body_points = b.body_points;
        out.body_poses = b.body_poses;
        out.body_mask = b.body_mask;
        for (size_t index = 0; index < kRigidBodyCount; ++index) {
            const bool has_a = (a.body_mask & (1u << index)) != 0;
            const bool has_b = (b.body_mask & (1u << index)) != 0;
            for (size_t axis = 0; axis < 3; ++axis) {
                const size_t offset = 3 * index + axis;
                out.body_centers[offset] = (has_a && has_b) ? a.body_centers[offset] + (b.body_centers[offset] - a.body_centers[offset]) * alpha
                                                            : b.body_centers[offset];
            }
            const std::array<double, 4> quat_a{a.body_quaternions[4 * index], a.body_quaternions[4 * index + 1],
                                               a.body_quaternions[4 * index + 2], a.body_quaternions[4 * index + 3]};
            const std::array<double, 4> quat_b{b.body_quaternions[4 * index], b.body_quaternions[4 * index + 1],
                                               b.body_quaternions[4 * index + 2], b.body_quaternions[4 * index + 3]};
            const std::array<double, 4> quaternion = (has_a && has_b) ? interpolate_quaternion(quat_a, quat_b, alpha) : quat_b;
            for (size_t item = 0; item < 4; ++item) out.body_quaternions[4 * index + item] = quaternion[item];
            Matrix3 rotation{};
            if (quaternion_to_matrix(quaternion, rotation)) {
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t col = 0; col < 3; ++col) out.body_rotations[9 * index + 3 * row + col] = rotation[row][col];
                }
            }
        }
        return out;
    }

    JointState joint_state_at(double timestamp_s)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return joint_state_from_kinematic_samples(kinematic_samples_, timestamp_s);
    }

private:
    std::array<RigidBodyPose, kRigidBodyCount> simulated_pose_array(double x, double y, double z, double t) const
    {
        std::array<RigidBodyPose, kRigidBodyCount> poses{};
        const Matrix3 identity = identity_matrix();
        for (size_t index = 0; index < kRigidBodyCount; ++index) {
            const double offset = (static_cast<double>(index) - 2.5) * 0.055;
            poses[index].position = Vec3{x + offset, y + 0.030 * std::sin(static_cast<double>(index) * 1.7 + t),
                                         z + 0.025 * std::cos(static_cast<double>(index) * 1.3 + t)};
            poses[index].rotation = identity;
            poses[index].quaternion = {0.0, 0.0, 0.0, 1.0};
            poses[index].valid = true;
        }
        return poses;
    }

    std::string simulated_body_points(const std::array<RigidBodyPose, kRigidBodyCount>& poses) const
    {
        constexpr int ids[] = {1000, 1001, 1002, 1003, 1004, 1005};
        std::ostringstream out;
        out << std::fixed << std::setprecision(6);
        for (size_t index = 0; index < kRigidBodyCount; ++index) {
            if (index > 0) out << '|';
            out << ids[index] << ':' << poses[index].position.x << ':' << poses[index].position.y << ':' << poses[index].position.z;
        }
        return out.str();
    }

    std::string simulated_body_poses(const std::array<RigidBodyPose, kRigidBodyCount>& poses) const
    {
        constexpr int ids[] = {1000, 1001, 1002, 1003, 1004, 1005};
        std::ostringstream out;
        out << std::fixed << std::setprecision(6);
        for (size_t index = 0; index < kRigidBodyCount; ++index) {
            if (index > 0) out << '|';
            out << ids[index] << ':' << poses[index].position.x << ':' << poses[index].position.y << ':' << poses[index].position.z
                << ":0.000000000:0.000000000:0.000000000:1.000000000";
        }
        return out.str();
    }

    void run()
    {
        const auto period = std::chrono::microseconds(2778);
        auto next = Clock::now();
        uint64_t frame = 0;
        while (running_) {
            const double t = now_seconds();
            constexpr double pi = 3.14159265358979323846;
            MocapSample sample;
            sample.valid = true;
            sample.stale = false;
            sample.frame = frame++;
            sample.timestamp_s = t;
            sample.raw_timestamp_s = t;
            sample.received_s = t;
            sample.timestamp_offset_ms = 0.0;
            sample.frame_rate_hz = 360.0;
            sample.clock_sample_count = 0;
            sample.clock_update_count = 0;
            sample.x = std::sin(2.0 * pi * 0.25 * t);
            sample.y = std::cos(2.0 * pi * 0.25 * t);
            sample.z = 0.25 * std::sin(2.0 * pi * 0.1 * t);
            sample.body_count = 6;
            sample.body_ids = "1000,1001,1002,1003,1004,1005";
            const auto poses = simulated_pose_array(sample.x, sample.y, sample.z, t);
            sample.body_points = simulated_body_points(poses);
            sample.body_poses = simulated_body_poses(poses);
            sample.body_mask = (1u << kRigidBodyCount) - 1u;
            for (size_t index = 0; index < kRigidBodyCount; ++index) {
                sample.body_centers[3 * index] = poses[index].position.x;
                sample.body_centers[3 * index + 1] = poses[index].position.y;
                sample.body_centers[3 * index + 2] = poses[index].position.z;
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t col = 0; col < 3; ++col) sample.body_rotations[9 * index + 3 * row + col] = poses[index].rotation[row][col];
                }
                for (size_t item = 0; item < 4; ++item) sample.body_quaternions[4 * index + item] = poses[index].quaternion[item];
            }
            MocapKinematicSample kinematic;
            kinematic.frame = sample.frame;
            kinematic.timestamp_s = sample.timestamp_s;
            kinematic.raw_timestamp_s = sample.raw_timestamp_s;
            kinematic.received_s = sample.received_s;
            kinematic.valid = mocap_poses_to_q_mk8(poses, (1u << kRigidBodyCount) - 1u, kinematic.theta);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                samples_.push_back(sample);
                while (samples_.size() > 2048) samples_.pop_front();
                if (kinematic.valid) {
                    kinematic_samples_.push_back(kinematic);
                    while (kinematic_samples_.size() > 2048) kinematic_samples_.pop_front();
                }
            }
            next += period;
            std::this_thread::sleep_until(next);
        }
    }

    std::atomic<bool> running_{false};
    std::thread thread_;
    std::mutex mutex_;
    std::deque<MocapSample> samples_;
    std::deque<MocapKinematicSample> kinematic_samples_;
};

std::string quote_arg(const std::string& value)
{
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += "\\\"";
        else out.push_back(c);
    }
    out += "\"";
    return out;
}

struct MocapClockResult {
    double calibrated_timestamp_s = 0.0;
    double offset_ms = 0.0;
    double frame_rate_hz = 0.0;
    uint64_t frame_drop_count = 0;
    size_t sample_count = 0;
    uint64_t update_count = 0;
};

class MocapClockCalibrator {
public:
    void reset()
    {
        samples_.clear();
        has_offset_ = false;
        has_last_frame_ = false;
        offset_s_ = 0.0;
        frame_rate_hz_ = 0.0;
        frame_drop_count_ = 0;
        update_count_ = 0;
        last_frame_ = 0;
        last_calibration_host_s_ = 0.0;
    }

    MocapClockResult update(double raw_timestamp_s, double receive_host_s, uint64_t frame)
    {
        if (!std::isfinite(raw_timestamp_s) || raw_timestamp_s <= 0.0) raw_timestamp_s = receive_host_s;
        if (!has_offset_) {
            offset_s_ = receive_host_s - raw_timestamp_s;
            has_offset_ = true;
            last_calibration_host_s_ = receive_host_s;
        }
        if (has_last_frame_ && frame > last_frame_ + 1) frame_drop_count_ += frame - last_frame_ - 1;
        has_last_frame_ = true;
        last_frame_ = frame;

        samples_.push_back(Observation{raw_timestamp_s, receive_host_s, frame});
        while (!samples_.empty() && receive_host_s - samples_.front().host_s > kWindowS) samples_.pop_front();
        update_frame_rate();

        if (samples_.size() >= kMinWindowSamples && receive_host_s - last_calibration_host_s_ >= kWindowS &&
            frame_rate_hz_ >= kMinFrameRateHz && frame_rate_hz_ <= kMaxFrameRateHz) {
            double offset_sum_s = 0.0;
            for (const Observation& sample : samples_) offset_sum_s += sample.host_s - sample.raw_s;
            const double candidate_offset_s = offset_sum_s / static_cast<double>(samples_.size());
            offset_s_ = candidate_offset_s;
            last_calibration_host_s_ = receive_host_s;
            update_count_++;
        }

        return MocapClockResult{raw_timestamp_s + offset_s_, offset_s_ * 1000.0, frame_rate_hz_, frame_drop_count_, samples_.size(), update_count_};
    }

private:
    struct Observation {
        double raw_s = 0.0;
        double host_s = 0.0;
        uint64_t frame = 0;
    };

    void update_frame_rate()
    {
        if (samples_.size() < 2) {
            frame_rate_hz_ = 0.0;
            return;
        }
        const Observation& first = samples_.front();
        const Observation& last = samples_.back();
        const double host_span_s = last.host_s - first.host_s;
        if (host_span_s <= 0.0 || last.frame <= first.frame) {
            frame_rate_hz_ = 0.0;
            return;
        }
        frame_rate_hz_ = static_cast<double>(last.frame - first.frame) / host_span_s;
    }

    static constexpr double kWindowS = 5.0;
    static constexpr size_t kMinWindowSamples = 1700;
    static constexpr double kMinFrameRateHz = 320.0;
    static constexpr double kMaxFrameRateHz = 400.0;

    std::deque<Observation> samples_;
    bool has_offset_ = false;
    bool has_last_frame_ = false;
    double offset_s_ = 0.0;
    double frame_rate_hz_ = 0.0;
    uint64_t frame_drop_count_ = 0;
    uint64_t update_count_ = 0;
    uint64_t last_frame_ = 0;
    double last_calibration_host_s_ = 0.0;
};

class MocapBridge {
public:
    explicit MocapBridge(JsonEmitter& emitter) : emitter_(emitter) {}

    void start(const Options& options)
    {
        if (!options.mocap_live || running_) return;
        clock_.reset();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            latest_ = MocapSample{};
            kinematic_samples_.clear();
        }
        running_ = true;
        emit_status("starting", options);
        thread_ = std::thread([this, options]() { run(options); });
    }

    void stop()
    {
        const bool was_running = running_;
        running_ = false;
#ifdef _WIN32
        if (process_info_.hProcess != nullptr) {
            TerminateProcess(process_info_.hProcess, 0);
        }
#endif
        if (thread_.joinable()) thread_.join();
        if (was_running) emit_status("stopped");
    }

    MocapSample sample_at(double)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        MocapSample out = latest_;
        if (!out.valid) return out;
        const double current_host_s = now_seconds();
        out.age_ms = std::max(0.0, (current_host_s - out.received_s) * 1000.0);
        out.latency_ms = std::max(0.0, (current_host_s - out.timestamp_s) * 1000.0);
        out.stale = out.age_ms > 100.0;
        return out;
    }

    JointState joint_state_at(double timestamp_s)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return joint_state_from_kinematic_samples(kinematic_samples_, timestamp_s);
    }

private:
    void emit_status(const std::string& state)
    {
        emitter_.emit("{\"type\":\"mocap_status\",\"state\":\"" + json_escape(state) + "\",\"source\":\"backend\"}");
    }

    void emit_status(const std::string& state, const Options& options)
    {
        emitter_.emit("{\"type\":\"mocap_status\",\"state\":\"" + json_escape(state) +
                      "\",\"source\":\"backend\",\"server\":\"" + json_escape(options.mocap_server) +
                      "\",\"local\":\"" + json_escape(options.mocap_local) +
                      "\",\"rigid_ids\":\"" + json_escape(options.mocap_rigid_ids) + "\"}");
    }

    void emit_log(const std::string& message)
    {
        emitter_.emit("{\"type\":\"mocap_log\",\"source\":\"backend\",\"message\":\"" + json_escape(message) + "\"}");
    }

    void emit_error(const std::string& message)
    {
        emitter_.emit("{\"type\":\"error\",\"source\":\"backend_mocap\",\"message\":\"mocap: " + json_escape(message) + "\"}");
    }

    void apply_json_line(const std::string& line)
    {
        const std::string type = json_string_value(line, "type").value_or("");
        if (type == "mocap_status") {
            emitter_.emit(line);
            return;
        }
        if (type != "mocap") {
            emit_log(line);
            return;
        }
        MocapSample sample;
        sample.valid = json_bool_value(line, "valid").value_or(false);
        sample.stale = json_bool_value(line, "stale").value_or(true);
        sample.frame = static_cast<uint64_t>(json_int_value(line, "frame").value_or(0));
        sample.raw_timestamp_s = json_double_value(line, "timestamp_s").value_or(now_seconds());
        const double receive_host_s = now_seconds();
        const MocapClockResult clock = clock_.update(sample.raw_timestamp_s, receive_host_s, sample.frame);
        sample.timestamp_s = clock.calibrated_timestamp_s;
        sample.received_s = receive_host_s;
        sample.timestamp_offset_ms = clock.offset_ms;
        sample.frame_rate_hz = clock.frame_rate_hz;
        sample.frame_drop_count = clock.frame_drop_count;
        sample.clock_sample_count = clock.sample_count;
        sample.clock_update_count = clock.update_count;
        sample.latency_ms = std::max(0.0, (receive_host_s - sample.timestamp_s) * 1000.0);
        sample.x = json_double_value(line, "x").value_or(0.0);
        sample.y = json_double_value(line, "y").value_or(0.0);
        sample.z = json_double_value(line, "z").value_or(0.0);
        sample.vx = json_double_value(line, "vx").value_or(0.0);
        sample.vy = json_double_value(line, "vy").value_or(0.0);
        sample.vz = json_double_value(line, "vz").value_or(0.0);
        sample.body_count = json_int_value(line, "body_count").value_or(0);
        sample.body_ids = json_string_value(line, "body_ids").value_or("");
        sample.body_points = json_string_value(line, "body_points").value_or("");
        sample.body_poses = json_string_value(line, "body_poses").value_or("");

        MocapKinematicSample kinematic;
        if (sample.valid) {
            std::array<RigidBodyPose, kRigidBodyCount> poses{};
            uint32_t body_mask = 0;
            if (parse_body_poses(sample.body_poses, poses, body_mask)) {
                sample.body_mask = body_mask;
                for (size_t index = 0; index < kRigidBodyCount; ++index) {
                    sample.body_centers[3 * index] = poses[index].position.x;
                    sample.body_centers[3 * index + 1] = poses[index].position.y;
                    sample.body_centers[3 * index + 2] = poses[index].position.z;
                    for (size_t row = 0; row < 3; ++row) {
                        for (size_t col = 0; col < 3; ++col) sample.body_rotations[9 * index + 3 * row + col] = poses[index].rotation[row][col];
                    }
                    for (size_t item = 0; item < 4; ++item) sample.body_quaternions[4 * index + item] = poses[index].quaternion[item];
                }
            }
            if (sample.body_mask != 0 && mocap_poses_to_q_mk8(poses, body_mask, kinematic.theta)) {
                kinematic.frame = sample.frame;
                kinematic.timestamp_s = sample.timestamp_s;
                kinematic.raw_timestamp_s = sample.raw_timestamp_s;
                kinematic.received_s = sample.received_s;
                kinematic.valid = true;
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        latest_ = sample;
        if (kinematic.valid) {
            kinematic_samples_.push_back(kinematic);
            while (kinematic_samples_.size() > 2048) kinematic_samples_.pop_front();
        }
        last_update_ = Clock::now();
    }

    std::string command_line(const Options& options) const
    {
        std::ostringstream command;
        command << quote_arg(options.mocap_python) << " "
                << quote_arg(options.mocap_script)
                << " --live --json --seconds 0"
                << " --server " << quote_arg(options.mocap_server)
                << " --local " << quote_arg(options.mocap_local)
                << " --rigid-ids " << quote_arg(options.mocap_rigid_ids)
                << (options.mocap_multicast ? " --multicast" : " --unicast");
        return command.str();
    }

    void run(const Options& options)
    {
#ifndef _WIN32
        (void)options;
    emit_error("live NatNet bridge is only implemented on Windows");
        running_ = false;
#else
        SECURITY_ATTRIBUTES security{};
        security.nLength = sizeof(security);
        security.bInheritHandle = TRUE;
        HANDLE read_pipe = nullptr;
        HANDLE write_pipe = nullptr;
        if (!CreatePipe(&read_pipe, &write_pipe, &security, 0)) {
            emit_error("failed to create mocap output pipe");
            running_ = false;
            return;
        }
        SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOA startup{};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdOutput = write_pipe;
        startup.hStdError = write_pipe;
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        std::string command = command_line(options);
        std::vector<char> mutable_command(command.begin(), command.end());
        mutable_command.push_back('\0');

        ZeroMemory(&process_info_, sizeof(process_info_));
        BOOL ok = CreateProcessA(nullptr, mutable_command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process_info_);
        CloseHandle(write_pipe);
        if (!ok) {
            emit_error("failed to start mocap process: " + command);
            CloseHandle(read_pipe);
            running_ = false;
            return;
        }

        std::string line;
        char buffer[256];
        while (running_) {
            DWORD bytes_read = 0;
            if (!ReadFile(read_pipe, buffer, sizeof(buffer), &bytes_read, nullptr) || bytes_read == 0) break;
            for (DWORD i = 0; i < bytes_read; ++i) {
                char c = buffer[i];
                if (c == '\r' || c == '\n') {
                    if (!line.empty()) {
                        apply_json_line(line);
                        line.clear();
                    }
                } else {
                    line.push_back(c);
                    if (line.size() > 4096) {
                        emit_error("mocap output line exceeded 4096 bytes");
                        line.clear();
                    }
                }
            }
        }

        const bool unexpected_exit = running_;
        CloseHandle(read_pipe);
        if (process_info_.hProcess != nullptr) {
            TerminateProcess(process_info_.hProcess, 0);
            CloseHandle(process_info_.hProcess);
            process_info_.hProcess = nullptr;
        }
        if (process_info_.hThread != nullptr) {
            CloseHandle(process_info_.hThread);
            process_info_.hThread = nullptr;
        }
        running_ = false;
        if (unexpected_exit) emit_error("mocap process exited");
#endif
    }

    JsonEmitter& emitter_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::mutex mutex_;
    MocapClockCalibrator clock_;
    MocapSample latest_;
    std::deque<MocapKinematicSample> kinematic_samples_;
    Clock::time_point last_update_{};
#ifdef _WIN32
    PROCESS_INFORMATION process_info_{};
#endif
};

class MocapRuntime {
public:
    explicit MocapRuntime(JsonEmitter& emitter) : bridge_(emitter) {}

    void start(const Options& options)
    {
        if (options.mocap_live) {
            bridge_.start(options);
        } else {
            simulator_.start(options.mocap_sim);
        }
    }

    void stop()
    {
        bridge_.stop();
        simulator_.stop();
    }

    MocapSample sample_at(double timestamp_s)
    {
        MocapSample live = bridge_.sample_at(timestamp_s);
        if (live.valid) return live;
        return simulator_.sample_at(timestamp_s);
    }

    JointState joint_state_at(double timestamp_s)
    {
        JointState live = bridge_.joint_state_at(timestamp_s);
        if (live.valid) return live;
        return simulator_.joint_state_at(timestamp_s);
    }

private:
    MocapBridge bridge_;
    MocapSimulator simulator_;
};

std::string timestamp_for_filename()
{
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return out.str();
}

void wait_until_precise(Clock::time_point deadline)
{
    while (true) {
        const auto now = Clock::now();
        if (now >= deadline) return;
        const auto remaining = deadline - now;
        if (remaining > std::chrono::milliseconds(4)) {
            std::this_thread::sleep_for(remaining - std::chrono::milliseconds(3));
        } else if (remaining > std::chrono::milliseconds(1)) {
            std::this_thread::yield();
        } else {
            std::this_thread::yield();
        }
    }
}

class BackendRuntime {
public:
    BackendRuntime(Options options, JsonEmitter& emitter) : options_(std::move(options)), emitter_(emitter), mocap_(emitter)
    {
        if (options_.ids.empty()) options_.ids = default_dt_ids();
        protocol_ = options_.runtime_protocol;
        if (protocol_ == RuntimeProtocol::Auto) {
            protocol_ = all_runtime_broadcast_ids_supported(options_.ids) ? RuntimeProtocol::Broadcast : RuntimeProtocol::Unicast;
        }
        if (protocol_ == RuntimeProtocol::Broadcast && !all_runtime_broadcast_ids_supported(options_.ids)) {
            throw std::runtime_error("broadcast CAN protocol only supports IDs 0x101..0x118");
        }
        options_.rx_window_frac = clamp_rx_window_frac(options_.rx_window_frac);
        pressure_calibration_.set_path(options_.calibration_path);
        for (uint16_t id : options_.ids) boards_[id] = BoardState{id};
        if (options_.viz_enable) {
            std::string error;
            if (!viz_writer_.start(options_.viz_shm_name, error)) {
                emit_error(error);
            }
        }
    }

    ~BackendRuntime()
    {
        request_exit_ = true;
        stop_loop();
        mocap_.stop();
    }

    bool request_exit() const { return request_exit_; }
    bool is_running() const { return running_; }

    void emit_ready()
    {
        emitter_.emit("{\"type\":\"backend\",\"state\":\"ready\",\"safe\":true,\"ids\":\"" + json_escape(ids_csv()) +
                      "\",\"simulate_can\":" + std::string(options_.simulate_can ? "true" : "false") +
                      ",\"can_protocol\":\"" + runtime_protocol_name(protocol_) + "\"" +
                      ",\"can_order\":\"" + can_order_name(options_.can_order) + "\"" +
                      ",\"rx_window_frac\":" + number(options_.rx_window_frac, 3) +
                      ",\"calibration\":\"" + json_escape(pressure_calibration_.source()) + "\"" +
                      ",\"viz_enabled\":" + std::string(viz_writer_.enabled() ? "true" : "false") +
                      ",\"viz_shm_name\":\"" + json_escape(options_.viz_shm_name) + "\"}");
    }

    void start_loop(int duration_s = 0)
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        if (running_) return;
        if (worker_.joinable()) worker_.join();
        running_ = true;
        worker_ = std::thread([this, duration_s]() { loop_worker(duration_s); });
    }

    void stop_loop()
    {
        running_ = false;
        if (worker_.joinable() && std::this_thread::get_id() != worker_.get_id()) worker_.join();
    }

    void wait_until_stopped()
    {
        while (running_) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (worker_.joinable()) worker_.join();
    }

    void process_command(const std::string& line)
    {
        auto cmd = json_string_value(line, "cmd");
        if (!cmd) {
            emit_error("command missing cmd field");
            return;
        }
        if (*cmd == "start") {
            start_loop(0);
        } else if (*cmd == "stop") {
            stop_loop();
            send_disable_once();
        } else if (*cmd == "shutdown") {
            request_exit_ = true;
            stop_loop();
            send_disable_once();
        } else if (*cmd == "start_data_collection") {
            double duration_min = json_double_value(line, "duration_min").value_or(0.0);
            if (duration_min <= 0.0) duration_min = static_cast<double>(json_int_value(line, "duration_min").value_or(0));
            if (!std::isfinite(duration_min) || duration_min <= 0.0) {
                emit_error("data collection duration_min must be positive");
                return;
            }
            const bool stop_loop_on_complete = !running_;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                collection_start_requested_ = true;
                collection_stop_requested_ = false;
                collection_requested_duration_s_ = duration_min * 60.0;
                collection_stop_loop_on_complete_ = stop_loop_on_complete;
            }
            emitter_.emit("{\"type\":\"collection\",\"state\":\"requested\",\"duration_min\":" + number(duration_min, 3) + "}");
            start_loop(0);
        } else if (*cmd == "stop_data_collection") {
            std::lock_guard<std::mutex> lock(state_mutex_);
            collection_stop_requested_ = true;
            emitter_.emit("{\"type\":\"collection\",\"state\":\"stop_requested\"}");
        } else if (*cmd == "enable_outputs") {
            const bool enable = json_bool_value(line, "enable").value_or(false);
            std::lock_guard<std::mutex> lock(state_mutex_);
            outputs_enabled_ = enable;
            for (auto& [_, board] : boards_) board.flags = enable ? kControlEnable : 0;
            emitter_.emit("{\"type\":\"backend\",\"state\":\"outputs\",\"enabled\":" + std::string(enable ? "true" : "false") + "}");
        } else if (*cmd == "set_all_targets") {
            const uint16_t target = clamp_pressure(json_int_value(line, "target").value_or(0));
            std::lock_guard<std::mutex> lock(state_mutex_);
            for (auto& [_, board] : boards_) board.target = target;
        } else if (*cmd == "set_target") {
            const uint16_t id = static_cast<uint16_t>(json_int_value(line, "id").value_or(0));
            const uint16_t target = clamp_pressure(json_int_value(line, "target").value_or(0));
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (boards_.count(id) == 0) {
                emit_error("unknown board id " + id_hex(id));
            } else {
                boards_[id].target = target;
            }
        } else if (*cmd == "set_targets") {
            // MPC fast path: one frame sets every actuator target. The targets array is
            // aligned to the sorted selected-id order, which is exactly the order reported
            // in robot_state.ids. An optional "enable" toggles control-enable atomically.
            const auto targets = json_int_array_value(line, "targets");
            if (!targets) {
                emit_error("set_targets requires an integer \"targets\" array");
                return;
            }
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (targets->size() != boards_.size()) {
                emit_error("set_targets expected " + std::to_string(boards_.size()) + " targets, got " + std::to_string(targets->size()));
                return;
            }
            const auto enable = json_bool_value(line, "enable");
            if (enable) outputs_enabled_ = *enable;
            size_t index = 0;
            for (auto& [_, board] : boards_) {
                board.target = clamp_pressure((*targets)[index++]);
                if (enable) board.flags = *enable ? kControlEnable : 0;
            }
        } else if (*cmd == "run_status_check") {
            int duration_s = json_int_value(line, "duration_s").value_or(10);
            if (duration_s < 1) duration_s = 1;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                outputs_enabled_ = false;
                for (auto& [_, board] : boards_) {
                    board.target = 0;
                    board.flags = 0;
                }
            }
            start_loop(duration_s);
        } else {
            emit_error("unknown command: " + *cmd);
        }
    }

private:
    std::string ids_csv() const
    {
        std::ostringstream out;
        bool first = true;
        for (const auto& [id, _] : boards_) {
            if (!first) out << ',';
            first = false;
            out << id_hex(id);
        }
        return out.str();
    }

    void emit_error(const std::string& message)
    {
        emitter_.emit("{\"type\":\"error\",\"message\":\"" + json_escape(message) + "\"}");
    }

    std::unique_ptr<ICanTransport> make_transport()
    {
        if (options_.simulate_can) return std::make_unique<FakeTransport>();
        return std::make_unique<SlcanTransport>(options_.port, options_.tty_baud);
    }

    std::vector<std::string> selected_id_strings_locked() const
    {
        std::vector<std::string> ids;
        ids.reserve(boards_.size());
        for (const auto& [id, _] : boards_) ids.push_back(id_hex(id));
        return ids;
    }

    std::string collection_adc_ranges_json_locked() const
    {
        std::ostringstream out;
        out << '{';
        bool first = true;
        for (const auto& [id, _] : boards_) {
            const ActuatorAdcRange range = pressure_calibration_.range_for(id);
            if (!first) out << ',';
            first = false;
            out << '"' << id_hex(id) << "\":[" << range.min_adc << ',' << range.max_adc << ']';
        }
        out << '}';
        return out.str();
    }

    std::string collection_pairs_json() const
    {
        std::ostringstream out;
        out << '[';
        const auto pairs = collection_actuator_pairs();
        for (size_t index = 0; index < pairs.size(); ++index) {
            if (index > 0) out << ',';
            out << "[\"" << id_hex(pairs[index].first) << "\",\"" << id_hex(pairs[index].second) << "\"]";
        }
        out << ']';
        return out.str();
    }

    bool selected_ids_are_complete_24_locked() const
    {
        if (boards_.size() != static_cast<size_t>(kLastRuntimeId - kFirstRuntimeId + 1)) return false;
        for (uint16_t id = kFirstRuntimeId; id <= kLastRuntimeId; ++id) {
            if (boards_.count(id) == 0) return false;
        }
        return true;
    }

    std::string collection_phase_locked() const
    {
        if (collection_active_) return "collecting";
        if (collection_deflating_) return "deflating";
        return "inactive";
    }

    void open_collection_chunk_locked(uint64_t cycle, double time_s)
    {
        if (collection_chunk_.is_open()) return;
        std::ostringstream name;
        name << "samples_chunk_" << std::setw(4) << std::setfill('0') << collection_chunk_index_ << "_open.jsonl.tmp";
        collection_chunk_temp_path_ = collection_session_dir_ / name.str();
        collection_chunk_.open(collection_chunk_temp_path_, std::ios::out | std::ios::trunc);
        collection_chunk_samples_ = 0;
        collection_chunk_start_cycle_ = cycle;
        collection_chunk_start_time_s_ = time_s;
    }

    void write_collection_manifest_locked()
    {
        if (collection_session_dir_.empty()) return;
        const std::filesystem::path manifest_path = collection_session_dir_ / "manifest.json";
        const std::filesystem::path temp_path = collection_session_dir_ / "manifest.json.tmp";
        std::ofstream manifest(temp_path);
        manifest << "{\n";
        manifest << "  \"schema_version\": 1,\n";
        manifest << "  \"session_tag\": \"" << json_escape(collection_session_tag_) << "\",\n";
        manifest << "  \"session_dir\": \"" << json_escape(collection_session_dir_.string()) << "\",\n";
        manifest << "  \"state_dimension\": 48,\n";
        manifest << "  \"input_dimension\": 24,\n";
        manifest << "  \"total_samples\": " << collection_total_samples_ << ",\n";
        manifest << "  \"checkpoint_count\": " << collection_checkpoint_count_ << ",\n";
        manifest << "  \"active\": " << (collection_active_ || collection_deflating_ ? "true" : "false") << ",\n";
        manifest << "  \"chunks\": [\n";
        for (size_t index = 0; index < collection_chunks_.size(); ++index) {
            const CollectionChunkInfo& chunk = collection_chunks_[index];
            manifest << "    {\"path\":\"" << json_escape(chunk.path) << "\",\"reason\":\"" << json_escape(chunk.reason)
                     << "\",\"samples\":" << chunk.samples << ",\"start_cycle\":" << chunk.start_cycle
                     << ",\"end_cycle\":" << chunk.end_cycle << ",\"start_time_s\":" << number(chunk.start_time_s, 6)
                     << ",\"end_time_s\":" << number(chunk.end_time_s, 6) << '}';
            if (index + 1 < collection_chunks_.size()) manifest << ',';
            manifest << "\n";
        }
        manifest << "  ]\n";
        manifest << "}\n";
        manifest.close();
        std::error_code ec;
        std::filesystem::remove(manifest_path, ec);
        ec.clear();
        std::filesystem::rename(temp_path, manifest_path, ec);
        if (ec) emit_error("failed to update collection manifest: " + ec.message());
    }

    void close_collection_chunk_locked(const std::string& reason, uint64_t end_cycle, double end_time_s)
    {
        if (!collection_chunk_.is_open()) return;
        collection_chunk_.flush();
        collection_chunk_.close();
        if (collection_chunk_samples_ == 0) {
            std::error_code remove_error;
            std::filesystem::remove(collection_chunk_temp_path_, remove_error);
            return;
        }
        std::ostringstream name;
        name << "samples_chunk_" << std::setw(4) << std::setfill('0') << collection_chunk_index_ << '_' << reason << '_'
             << timestamp_for_filename() << ".jsonl";
        const std::filesystem::path final_path = collection_session_dir_ / name.str();
        std::error_code ec;
        std::filesystem::remove(final_path, ec);
        ec.clear();
        std::filesystem::rename(collection_chunk_temp_path_, final_path, ec);
        if (ec) {
            emit_error("failed to finalize collection chunk: " + ec.message());
            return;
        }
        collection_chunks_.push_back(CollectionChunkInfo{final_path.string(), reason, collection_chunk_samples_, collection_chunk_start_cycle_, end_cycle,
                                                        collection_chunk_start_time_s_, end_time_s});
        if (reason == "checkpoint") collection_checkpoint_count_++;
        collection_chunk_index_++;
        collection_chunk_samples_ = 0;
        write_collection_manifest_locked();
        emitter_.emit("{\"type\":\"collection\",\"state\":\"chunk_closed\",\"reason\":\"" + json_escape(reason) +
                      "\",\"path\":\"" + json_escape(final_path.string()) + "\",\"samples\":" + std::to_string(collection_chunks_.back().samples) + "}");
    }

    void write_collection_metadata_locked()
    {
        std::ofstream metadata(collection_session_dir_ / "metadata.json");
        metadata << "{\n";
        metadata << "  \"schema_version\": 1,\n";
        metadata << "  \"session_tag\": \"" << json_escape(collection_session_tag_) << "\",\n";
        metadata << "  \"created_at_unix_s\": " << number(unix_seconds_now(), 6) << ",\n";
        metadata << "  \"sample_rate_hz\": " << options_.rate_hz << ",\n";
        metadata << "  \"duration_s\": " << number(collection_requested_duration_s_, 3) << ",\n";
        metadata << "  \"checkpoint_s\": " << number(kCollectionCheckpointS, 3) << ",\n";
        metadata << "  \"deflate_s\": " << number(kCollectionDeflateS, 3) << ",\n";
        metadata << "  \"sync_semantics\": \"runtime table sent before DLC-0 sync; firmware latches pressure ADC and promotes pending target on sync; compact reply pressure is latched ADC at sync\",\n";
        metadata << "  \"state_fields\": [\"q\",\"qdot\",\"pressure_adc\"],\n";
        metadata << "  \"state_dimension\": 48,\n";
        metadata << "  \"input_fields\": [\"target_adc\"],\n";
        metadata << "  \"input_dimension\": 24,\n";
        metadata << "  \"pressure_units\": \"adc_counts\",\n";
        metadata << "  \"target_units\": \"adc_counts\",\n";
        metadata << "  \"selected_ids\": " << json_string_array(selected_id_strings_locked()) << ",\n";
        metadata << "  \"adc_ranges\": " << collection_adc_ranges_json_locked() << ",\n";
        metadata << "  \"pressure_calibration_source\": \"" << json_escape(pressure_calibration_.source()) << "\",\n";
        metadata << "  \"pressure_calibration_default\": " << (pressure_calibration_.using_defaults() ? "true" : "false") << ",\n";
        metadata << "  \"actuator_pairs\": " << collection_pairs_json() << ",\n";
        metadata << "  \"target_generation_pressure_model\": \"linear ADC interpolation from each actuator's 0 psi and 40 psi calibration endpoints; calibration.json may extrapolate 40 psi from a 25 psi source mark\",\n";
        metadata << "  \"single_actuator_limit_psi\": " << number(kCollectionSingleLimitPsi, 3) << ",\n";
        metadata << "  \"pair_sum_limit_psi\": " << number(kCollectionPairSumLimitPsi, 3) << ",\n";
        metadata << "  \"single_actuator_limit_normalized\": " << number(kCollectionSingleLimitNormalized, 6) << ",\n";
        metadata << "  \"pair_sum_limit_normalized\": " << number(kCollectionPairSumLimitNormalized, 6) << ",\n";
        metadata << "  \"slew_psi_per_s_reference\": " << number(kCollectionSlewPsiPerS, 3) << ",\n";
        metadata << "  \"adc_range_reference_psi\": " << number(kCollectionAdcRangePsi, 3) << ",\n";
        metadata << "  \"can_protocol\": \"" << runtime_protocol_name(protocol_) << "\",\n";
        metadata << "  \"can_order\": \"" << can_order_name(options_.can_order) << "\",\n";
        metadata << "  \"rx_window_frac\": " << number(options_.rx_window_frac, 3) << ",\n";
        metadata << "  \"mocap_live\": " << (options_.mocap_live ? "true" : "false") << ",\n";
        metadata << "  \"mocap_sim\": " << (options_.mocap_sim ? "true" : "false") << ",\n";
        metadata << "  \"mocap_server\": \"" << json_escape(options_.mocap_server) << "\",\n";
        metadata << "  \"mocap_local\": \"" << json_escape(options_.mocap_local) << "\",\n";
        metadata << "  \"mocap_rigid_ids\": \"" << json_escape(options_.mocap_rigid_ids) << "\"\n";
        metadata << "}\n";
    }

    void begin_collection_locked(Clock::time_point now, uint64_t cycle)
    {
        collection_session_tag_ = timestamp_for_filename();
        collection_session_dir_ = std::filesystem::current_path() / "real_system_data_collection" / ("session_" + collection_session_tag_);
        std::filesystem::create_directories(collection_session_dir_);
        collection_chunks_.clear();
        collection_total_samples_ = 0;
        collection_checkpoint_count_ = 0;
        collection_chunk_index_ = 0;
        collection_norm_.clear();
        collection_velocity_.clear();
        const double time_s = time_point_seconds(now);
        for (const auto& [id, board] : boards_) {
            const ActuatorAdcRange range = pressure_calibration_.range_for(id);
            const double span = std::max(1.0, static_cast<double>(range.max_adc) - static_cast<double>(range.min_adc));
            const double normalized = std::clamp((static_cast<double>(board.target) - static_cast<double>(range.min_adc)) / span, 0.0,
                                                 kCollectionSingleLimitNormalized);
            collection_norm_[id] = normalized;
            collection_velocity_[id] = 0.0;
        }
        apply_collection_pair_projection_locked();
        collection_start_time_ = now;
        collection_deflate_start_time_ = Clock::time_point{};
        collection_active_ = true;
        collection_deflating_ = false;
        collection_start_requested_ = false;
        collection_stop_requested_ = false;
        collection_finish_reason_ = "complete";
        outputs_enabled_ = true;
        write_collection_metadata_locked();
        open_collection_chunk_locked(cycle, time_s);
        write_collection_manifest_locked();
        emitter_.emit("{\"type\":\"collection\",\"state\":\"started\",\"session_dir\":\"" + json_escape(collection_session_dir_.string()) +
                      "\",\"duration_s\":" + number(collection_requested_duration_s_, 3) + "}");
        if (!selected_ids_are_complete_24_locked()) {
            emitter_.emit("{\"type\":\"collection\",\"state\":\"warning\",\"message\":\"selected IDs are not complete 0x101-0x118; rows include masks but requested 48-state assumes 24 pressure channels\"}");
        }
    }

    void apply_collection_pair_projection_locked()
    {
        for (auto& [_, normalized] : collection_norm_) {
            normalized = std::clamp(normalized, 0.0, kCollectionSingleLimitNormalized);
        }
        for (const auto& pair : collection_actuator_pairs()) {
            auto left = collection_norm_.find(pair.first);
            auto right = collection_norm_.find(pair.second);
            if (left == collection_norm_.end() || right == collection_norm_.end()) continue;
            const double sum = left->second + right->second;
            if (sum <= kCollectionPairSumLimitNormalized || sum <= 1e-12) continue;
            const double scale_factor = kCollectionPairSumLimitNormalized / sum;
            left->second *= scale_factor;
            right->second *= scale_factor;
        }
    }

    void update_collection_targets_locked()
    {
        const double max_step = std::max(0.0001, kCollectionSlewPsiPerS / (kCollectionAdcRangePsi * static_cast<double>(std::max(1, options_.rate_hz))));
        std::uniform_real_distribution<double> acceleration_dist(-max_step * 0.18, max_step * 0.18);
        for (auto& [id, board] : boards_) {
            double velocity = collection_velocity_[id] * 0.985 + acceleration_dist(collection_rng_);
            velocity = std::max(-max_step, std::min(max_step, velocity));
            double normalized = collection_norm_[id] + velocity;
            if (normalized < 0.0) {
                normalized = -normalized;
                velocity = std::abs(velocity) * 0.35;
            } else if (normalized > kCollectionSingleLimitNormalized) {
                normalized = 2.0 * kCollectionSingleLimitNormalized - normalized;
                velocity = -std::abs(velocity) * 0.35;
            }
            collection_norm_[id] = std::clamp(normalized, 0.0, kCollectionSingleLimitNormalized);
            collection_velocity_[id] = velocity;
        }
        apply_collection_pair_projection_locked();
        for (auto& [id, board] : boards_) {
            const ActuatorAdcRange range = pressure_calibration_.range_for(id);
            const double target = static_cast<double>(range.min_adc) + collection_norm_[id] * (static_cast<double>(range.max_adc) - static_cast<double>(range.min_adc));
            board.target = clamp_pressure(static_cast<int>(std::lround(target)));
        }
        outputs_enabled_ = true;
    }

    void begin_collection_deflate_locked(Clock::time_point now, const std::string& reason)
    {
        collection_active_ = false;
        collection_deflating_ = true;
        collection_stop_requested_ = false;
        collection_finish_reason_ = reason;
        collection_deflate_start_time_ = now;
        outputs_enabled_ = true;
        for (auto& [_, board] : boards_) board.target = 0;
        emitter_.emit("{\"type\":\"collection\",\"state\":\"deflating\",\"reason\":\"" + json_escape(reason) +
                      "\",\"deflate_s\":" + number(kCollectionDeflateS, 3) + "}");
    }

    void finish_collection_locked(const std::string& reason, uint64_t cycle, double time_s)
    {
        close_collection_chunk_locked(reason, cycle, time_s);
        collection_active_ = false;
        collection_deflating_ = false;
        collection_start_requested_ = false;
        collection_stop_requested_ = false;
        outputs_enabled_ = false;
        for (auto& [_, board] : boards_) board.target = 0;
        collection_last_session_dir_ = collection_session_dir_;
        collection_last_total_samples_ = collection_total_samples_;
        collection_last_checkpoint_count_ = collection_checkpoint_count_;
        write_collection_manifest_locked();
        emitter_.emit("{\"type\":\"collection\",\"state\":\"stopped\",\"reason\":\"" + json_escape(reason) +
                      "\",\"session_dir\":\"" + json_escape(collection_session_dir_.string()) + "\",\"samples\":" +
                      std::to_string(collection_total_samples_) + ",\"checkpoints\":" + std::to_string(collection_checkpoint_count_) + "}");
    }

    bool update_collection_control_locked(Clock::time_point now, uint64_t cycle)
    {
        bool stop_loop_after_cycle = false;
        if (collection_start_requested_ && !collection_active_ && !collection_deflating_) begin_collection_locked(now, cycle);
        if (collection_active_) {
            const double elapsed_s = std::chrono::duration<double>(now - collection_start_time_).count();
            if (collection_stop_requested_) {
                begin_collection_deflate_locked(now, "stop");
            } else if (elapsed_s >= collection_requested_duration_s_) {
                begin_collection_deflate_locked(now, "complete");
            } else {
                update_collection_targets_locked();
            }
        }
        if (collection_deflating_) {
            outputs_enabled_ = true;
            for (auto& [_, board] : boards_) board.target = 0;
            const double deflate_elapsed_s = std::chrono::duration<double>(now - collection_deflate_start_time_).count();
            if (deflate_elapsed_s >= kCollectionDeflateS) {
                const bool should_stop_loop = collection_stop_loop_on_complete_;
                finish_collection_locked(collection_finish_reason_, cycle, time_point_seconds(now));
                collection_stop_loop_on_complete_ = false;
                stop_loop_after_cycle = should_stop_loop;
            }
        }
        return stop_loop_after_cycle;
    }

    std::vector<uint64_t> board_u64_values_locked(uint64_t BoardState::*field) const
    {
        std::vector<uint64_t> values;
        values.reserve(boards_.size());
        for (const auto& [_, board] : boards_) values.push_back(board.*field);
        return values;
    }

    std::vector<double> board_latency_values_locked() const
    {
        std::vector<double> values;
        values.reserve(boards_.size());
        for (const auto& [_, board] : boards_) values.push_back(board.latency_ms);
        return values;
    }

    void write_collection_sample_locked(const RobotStateSample& state, const MocapSample& mocap, double jitter_ms)
    {
        if (!collection_active_ && !collection_deflating_) return;
        open_collection_chunk_locked(state.cycle, state.can_sync_time_s);
        if (!collection_chunk_.is_open()) return;
        const std::string phase = collection_phase_locked();
        collection_chunk_ << "{\"schema_version\":1"
                          << ",\"cycle\":" << state.cycle
                          << ",\"phase\":\"" << phase << "\""
                          << ",\"timestamp_unix_s\":" << number(unix_seconds_now(), 6)
                          << ",\"timestamp_s\":" << number(state.can_sync_time_s, 6)
                          << ",\"cycle_start_time_s\":" << number(state.cycle_start_time_s, 6)
                          << ",\"can_sync_time_s\":" << number(state.can_sync_time_s, 6)
                          << ",\"jitter_ms\":" << number(jitter_ms, 3)
                          << ",\"ids\":" << json_string_array(selected_id_strings_locked())
                          << ",\"robot_state\":{\"q\":" << json_joint_array(state.joint_current_estimate.theta, 9)
                          << ",\"qdot\":" << json_joint_array(state.joint_current_estimate.theta_dot, 9)
                          << ",\"pressure_adc\":" << json_u16_array(state.pressure_adc_filtered) << '}'
                          << ",\"input\":{\"target_adc\":" << json_u16_array(state.target_next_sync) << '}'
                          << ",\"actuator_status\":" << json_u8_array(state.actuator_status)
                          << ",\"actuator_stale\":" << json_bool_array(state.actuator_stale)
                          << ",\"control_next_sync\":" << json_u8_array(state.control_next_sync)
                          << ",\"actuator_missed_total\":" << json_u64_array(board_u64_values_locked(&BoardState::missed))
                          << ",\"actuator_reply_latency_ms\":" << json_double_array(board_latency_values_locked(), 3)
                          << ",\"cycle_responded\":" << state.cycle_responded
                          << ",\"cycle_expected\":" << state.cycle_expected
                          << ",\"total_missed\":" << total_missed_
                          << ",\"unexpected_replies\":" << unexpected_replies_
                          << ",\"duplicate_replies\":" << duplicate_replies_
                          << ",\"joint_current_valid\":" << (state.joint_current_estimate.valid ? "true" : "false")
                          << ",\"joint_current_extrapolated\":" << (state.joint_current_estimate.extrapolated ? "true" : "false")
                          << ",\"joint_current_source_error_ms\":" << number(state.joint_current_estimate.source_time_error_ms, 3)
                          << ",\"joint_current_extrapolation_ms\":" << number(state.joint_current_estimate.extrapolation_ms, 3)
                          << ",\"mocap\":{\"valid\":" << (mocap.valid ? "true" : "false")
                          << ",\"stale\":" << (mocap.stale ? "true" : "false")
                          << ",\"frame\":" << mocap.frame
                          << ",\"timestamp_s\":" << number(mocap.timestamp_s, 6)
                          << ",\"raw_timestamp_s\":" << number(mocap.raw_timestamp_s, 6)
                          << ",\"received_s\":" << number(mocap.received_s, 6)
                          << ",\"latency_ms\":" << number(mocap.latency_ms, 3)
                          << ",\"age_ms\":" << number(mocap.age_ms, 3)
                          << ",\"timestamp_offset_ms\":" << number(mocap.timestamp_offset_ms, 3)
                          << ",\"frame_rate_hz\":" << number(mocap.frame_rate_hz, 3)
                          << ",\"frame_drop_count\":" << mocap.frame_drop_count
                          << ",\"clock_sample_count\":" << mocap.clock_sample_count
                          << ",\"clock_update_count\":" << mocap.clock_update_count
                          << ",\"body_count\":" << mocap.body_count
                          << ",\"body_ids\":\"" << json_escape(mocap.body_ids) << "\""
                          << ",\"body_centers\":" << json_double_array(mocap.body_centers, 9)
                          << ",\"body_rotations\":" << json_double_array(mocap.body_rotations, 9)
                          << ",\"body_quaternions\":" << json_double_array(mocap.body_quaternions, 9)
                          << ",\"body_points\":\"" << json_escape(mocap.body_points) << "\""
                          << ",\"body_poses\":\"" << json_escape(mocap.body_poses) << "\"}"
                          << "}\n";
        collection_chunk_samples_++;
        collection_total_samples_++;
        if ((collection_total_samples_ % 15u) == 0u) collection_chunk_.flush();
        if (state.can_sync_time_s - collection_chunk_start_time_s_ >= kCollectionCheckpointS) {
            close_collection_chunk_locked("checkpoint", state.cycle, state.can_sync_time_s);
            open_collection_chunk_locked(state.cycle + 1, state.can_sync_time_s);
        }
    }

    void append_joint_csv_header(const std::string& prefix)
    {
        for (size_t index = 0; index < kJointCount; ++index) csv_ << ',' << prefix << index;
    }

    void append_joint_csv_values(const std::array<double, kJointCount>& values)
    {
        for (double value : values) csv_ << ',' << std::setprecision(9) << value;
    }

    void open_csv()
    {
        std::filesystem::create_directories(options_.log_dir);
        run_name_ = "vnema_backend_" + timestamp_for_filename();
        csv_path_ = options_.log_dir / (run_name_ + ".csv");
        report_path_ = options_.log_dir / (run_name_ + ".md");
        csv_.open(csv_path_);
        csv_ << "cycle,cycle_start_time_s,can_sync_time_s,can_protocol,can_order,rx_window_frac,id,target_next_sync,control_next_sync,pressure_adc_filtered,pressure_calibrated,status,status_errors,stale,missed,latency_ms,cycle_responded,cycle_expected,jitter_ms,total_missed,unexpected_replies,duplicate_replies,mocap_valid,mocap_stale,mocap_queue_age_ms,mocap_extrapolation_ms,mocap_latency_ms,mocap_timestamp_s,mocap_raw_timestamp_s,mocap_timestamp_offset_ms,mocap_frame_rate_hz,mocap_frame_drop_count,mocap_clock_sample_count,mocap_clock_update_count,mocap_body_count,mocap_x,mocap_y,mocap_z,mocap_vx,mocap_vy,mocap_vz,joint_current_valid,joint_fixed_delay_valid,joint_current_source_error_ms,joint_fixed_delay_source_error_ms,joint_current_extrapolated,joint_fixed_delay_extrapolated,joint_current_extrapolation_ms,joint_fixed_delay_extrapolation_ms,observer_time_ms,observer_time_max_ms,observer_over_budget_count,fk_valid,fk_time_ms,fk_time_max_ms,fk_over_budget_count,viz_publish_time_ms,viz_publish_time_max_ms,viz_publish_over_budget_count";
        append_joint_csv_header("joint_current_q");
        append_joint_csv_header("joint_current_qdot");
        append_joint_csv_header("joint_fixed_delay_q");
        append_joint_csv_header("joint_fixed_delay_qdot");
        csv_ << ",calibration_default\n";
    }

    void close_csv()
    {
        if (csv_.is_open()) csv_.close();
    }

    RobotStateSample make_robot_state_locked(uint64_t cycle, double cycle_start_time_s, double can_sync_time_s, int cycle_responded,
                                             int cycle_expected, const MocapSample& mocap_current, const MocapSample& mocap_fixed_delay,
                                             const JointState& joint_current, const JointState& joint_fixed_delay,
                                             const FkResult& fk_current, bool fk_valid, double observer_time_ms, double fk_time_ms) const
    {
        (void)mocap_fixed_delay;
        RobotStateSample state;
        state.cycle = cycle;
        state.cycle_start_time_s = cycle_start_time_s;
        state.can_sync_time_s = can_sync_time_s;
        state.joint_current_estimate = joint_current;
        state.joint_fixed_delay = joint_fixed_delay;
        state.fk_current = fk_current;
        state.fk_valid = fk_valid;
        state.mocap_body_centers = mocap_current.body_centers;
        state.mocap_body_rotations = mocap_current.body_rotations;
        state.mocap_body_mask = mocap_current.body_mask;
        state.mocap_from_fk = identity_fk_transform();
        state.mocap_base_frame_valid = make_mocap_from_fk_transform(fk_current, mocap_current, state.mocap_from_fk);
        state.mocap_frame = mocap_current.frame;
        state.mocap_timestamp_s = mocap_current.timestamp_s;
        state.mocap_raw_timestamp_s = mocap_current.raw_timestamp_s;
        state.mocap_timestamp_offset_ms = mocap_current.timestamp_offset_ms;
        state.mocap_frame_rate_hz = mocap_current.frame_rate_hz;
        state.mocap_frame_drop_count = mocap_current.frame_drop_count;
        state.mocap_clock_sample_count = mocap_current.clock_sample_count;
        state.mocap_clock_update_count = mocap_current.clock_update_count;
        state.mocap_queue_age_ms = mocap_current.age_ms;
        state.mocap_extrapolation_ms = (mocap_current.valid && mocap_current.stale) ? mocap_current.age_ms : 0.0;
        state.mocap_stale = mocap_current.stale;
        state.calibration_default = pressure_calibration_.using_defaults();
        state.cycle_responded = cycle_responded;
        state.cycle_expected = cycle_expected;
        state.observer_time_ms = observer_time_ms;
        state.observer_time_max_ms = observer_time_max_ms_;
        state.observer_over_budget_count = observer_over_budget_count_;
        state.fk_time_ms = fk_time_ms;
        state.fk_time_max_ms = fk_time_max_ms_;
        state.fk_over_budget_count = fk_over_budget_count_;
        state.viz_publish_time_ms = viz_publish_time_last_ms_;
        state.viz_publish_time_max_ms = viz_publish_time_max_ms_;
        state.viz_publish_over_budget_count = viz_publish_over_budget_count_;

        state.ids.reserve(boards_.size());
        state.pressure_adc_filtered.reserve(boards_.size());
        state.pressure_calibrated.reserve(boards_.size());
        state.actuator_status.reserve(boards_.size());
        state.actuator_stale.reserve(boards_.size());
        state.target_next_sync.reserve(boards_.size());
        state.control_next_sync.reserve(boards_.size());
        for (const auto& [id, board] : boards_) {
            state.ids.push_back(id);
            state.pressure_adc_filtered.push_back(board.pressure_filtered);
            state.pressure_calibrated.push_back(board.pressure_calibrated);
            state.actuator_status.push_back(board.status);
            state.actuator_stale.push_back(board.stale);
            state.target_next_sync.push_back(board.target);
            state.control_next_sync.push_back(board.flags);
        }
        return state;
    }

    VizSharedSample make_viz_sample(const RobotStateSample& state) const
    {
        VizSharedSample sample;
        sample.cycle = static_cast<int64_t>(state.cycle);
        sample.mocap_frame = static_cast<int64_t>(state.mocap_frame);
        sample.mocap_valid = state.mocap_body_mask != 0 ? 1 : 0;
        sample.mocap_stale = state.mocap_stale ? 1 : 0;
        sample.joint_current_valid = state.joint_current_estimate.valid ? 1 : 0;
        sample.joint_fixed_delay_valid = state.joint_fixed_delay.valid ? 1 : 0;
        sample.fk_valid = state.fk_valid ? 1 : 0;
        sample.base_frame_valid = state.mocap_base_frame_valid ? 1 : 0;
        sample.mocap_body_mask = static_cast<int64_t>(state.mocap_body_mask);
        sample.observer_over_budget_count = static_cast<int64_t>(state.observer_over_budget_count);
        sample.fk_over_budget_count = static_cast<int64_t>(state.fk_over_budget_count);
        sample.publish_over_budget_count = static_cast<int64_t>(state.viz_publish_over_budget_count);
        sample.cycle_start_time_s = state.cycle_start_time_s;
        sample.can_sync_time_s = state.can_sync_time_s;
        sample.mocap_timestamp_s = state.mocap_timestamp_s;
        sample.mocap_latency_ms = mocap_latency_last_ms_;
        sample.mocap_frame_rate_hz = state.mocap_frame_rate_hz;
        sample.observer_time_ms = state.observer_time_ms;
        sample.observer_time_max_ms = state.observer_time_max_ms;
        sample.observer_budget_ms = kObserverBudgetMs;
        sample.fk_time_ms = state.fk_time_ms;
        sample.fk_time_max_ms = state.fk_time_max_ms;
        sample.fk_budget_ms = kFkBudgetMs;
        sample.publish_time_ms = state.viz_publish_time_ms;
        sample.publish_time_max_ms = state.viz_publish_time_max_ms;
        sample.publish_budget_ms = kVizPublishBudgetMs;
        sample.q = state.joint_current_estimate.theta;
        sample.qdot = state.joint_current_estimate.theta_dot;
        sample.mocap_from_fk = state.mocap_from_fk;
        for (size_t index = 0; index < kFkUJointCount; ++index) {
            FkVec3 center = state.fk_current.ujoint_centers[index];
            if (state.mocap_base_frame_valid) center = transform_point(state.mocap_from_fk, center);
            sample.fk_centers[3 * index] = center.x;
            sample.fk_centers[3 * index + 1] = center.y;
            sample.fk_centers[3 * index + 2] = center.z;
        }
        FkVec3 tip = state.fk_current.tip_position;
        if (state.mocap_base_frame_valid) tip = transform_point(state.mocap_from_fk, tip);
        sample.fk_tip = {tip.x, tip.y, tip.z};
        sample.mocap_centers = state.mocap_body_centers;
        for (size_t index = 0; index < state.ids.size(); ++index) {
            size_t slot = index;
            if (runtime_broadcast_id_supported(state.ids[index])) slot = static_cast<size_t>(state.ids[index] - kFirstRuntimeId);
            if (slot >= kVizActuatorCount) continue;
            if (index < state.pressure_calibrated.size()) sample.pressure[slot] = state.pressure_calibrated[index];
            if (index < state.target_next_sync.size()) sample.target[slot] = static_cast<double>(state.target_next_sync[index]);
        }
        return sample;
    }

    void publish_visualization_sample(const RobotStateSample& state)
    {
        if (!viz_writer_.enabled()) return;
        VizSharedSample sample = make_viz_sample(state);
        const auto publish_start = Clock::now();
        std::string error;
        if (!viz_writer_.publish(sample, error)) {
            emit_error(error);
            return;
        }
        const double publish_time_ms = std::chrono::duration<double, std::milli>(Clock::now() - publish_start).count();
        viz_publish_time_last_ms_ = publish_time_ms;
        viz_publish_time_max_ms_ = std::max(viz_publish_time_max_ms_, publish_time_ms);
        if (publish_time_ms > kVizPublishBudgetMs) viz_publish_over_budget_count_++;
    }

    void emit_viz_status()
    {
        if (!options_.viz_enable) return;
        emitter_.emit("{\"type\":\"viz_status\",\"enabled\":" + std::string(viz_writer_.enabled() ? "true" : "false") +
                      ",\"shm_name\":\"" + json_escape(options_.viz_shm_name) + "\"" +
                      ",\"publish_count\":" + std::to_string(viz_writer_.publish_count()) +
                      ",\"publish_time_ms\":" + number(viz_publish_time_last_ms_, 3) +
                      ",\"publish_time_max_ms\":" + number(viz_publish_time_max_ms_, 3) +
                      ",\"publish_budget_ms\":" + number(kVizPublishBudgetMs, 3) +
                      ",\"publish_over_budget_count\":" + std::to_string(viz_publish_over_budget_count_) +
                      ",\"fk_time_ms\":" + number(fk_time_last_ms_, 3) +
                      ",\"fk_time_max_ms\":" + number(fk_time_max_ms_, 3) +
                      ",\"fk_budget_ms\":" + number(kFkBudgetMs, 3) +
                      ",\"fk_over_budget_count\":" + std::to_string(fk_over_budget_count_) + "}");
    }

    void loop_worker(int duration_s)
    {
#ifdef _WIN32
        timeBeginPeriod(1);
#endif
        cycle_count_ = 0;
        total_missed_ = 0;
        total_expected_ = 0;
        total_responded_ = 0;
        max_jitter_ms_ = 0.0;
        mocap_latency_samples_ = 0;
        mocap_latency_sum_ms_ = 0.0;
        mocap_latency_max_ms_ = 0.0;
        mocap_latency_last_ms_ = 0.0;
        mocap_timestamp_offset_last_ms_ = 0.0;
        mocap_frame_rate_last_hz_ = 0.0;
        mocap_frame_drop_count_last_ = 0;
        observer_time_last_ms_ = 0.0;
        observer_time_max_ms_ = 0.0;
        observer_over_budget_count_ = 0;
        fk_time_last_ms_ = 0.0;
        fk_time_max_ms_ = 0.0;
        fk_over_budget_count_ = 0;
        viz_publish_time_last_ms_ = 0.0;
        viz_publish_time_max_ms_ = 0.0;
        viz_publish_over_budget_count_ = 0;
        last_robot_state_ = RobotStateSample{};

        open_csv();
        mocap_.start(options_);
        transport_ = make_transport();
        std::string error;
        if (!transport_->open(error)) {
            emit_error(error);
            running_ = false;
            close_csv();
#ifdef _WIN32
            timeEndPeriod(1);
#endif
            return;
        }

        send_disable_once();
        drain_transport_for(std::chrono::milliseconds(20));

        emitter_.emit("{\"type\":\"backend\",\"state\":\"running\",\"safe\":" + std::string(outputs_enabled_ ? "false" : "true") +
              ",\"can_protocol\":\"" + runtime_protocol_name(protocol_) + "\"" +
                  ",\"log\":\"" + json_escape(csv_path_.string()) +
                  "\",\"calibration\":\"" + json_escape(pressure_calibration_.source()) + "\"}");

        const auto period = std::chrono::duration<double>(1.0 / std::max(1, options_.rate_hz));
        const uint64_t telemetry_interval_cycles = static_cast<uint64_t>(std::max(1, options_.rate_hz));
        const auto start_time = Clock::now();
        auto next_cycle = start_time;
        auto previous_cycle = start_time;
        uint64_t cycle = 0;

        while (running_) {
            const auto cycle_start = Clock::now();
            const double cycle_time_s = time_point_seconds(cycle_start);
            const double jitter_ms = std::abs(std::chrono::duration<double, std::milli>(cycle_start - previous_cycle).count() - std::chrono::duration<double, std::milli>(period).count());
            if (cycle > 0) max_jitter_ms_ = std::max(max_jitter_ms_, jitter_ms);
            previous_cycle = cycle_start;

            bool stop_loop_after_collection = false;
            std::vector<BoardCommand> commands;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                stop_loop_after_collection = update_collection_control_locked(cycle_start, cycle);
                for (auto& [id, board] : boards_) {
                    board.flags = outputs_enabled_ ? kControlEnable : 0;
                    commands.push_back(BoardCommand{id, board.target, board.flags});
                }
            }
            apply_can_order(commands, options_.can_order, cycle);

            std::vector<CanFrame> command_frames;
            if (protocol_ == RuntimeProtocol::Broadcast) {
                command_frames = compact_broadcast_command_frames(commands);
                if (options_.can_order == CanOrder::Reverse) {
                    std::reverse(command_frames.begin(), command_frames.end());
                } else if (options_.can_order == CanOrder::Rotate && !command_frames.empty()) {
                    const auto offset = static_cast<std::vector<CanFrame>::difference_type>(cycle % command_frames.size());
                    std::rotate(command_frames.begin(), command_frames.begin() + offset, command_frames.end());
                }
            } else {
                for (const BoardCommand& command : commands) {
                    command_frames.push_back(compact_command_frame(command.id, command.target, command.flags));
                }
            }

            Clock::time_point sync_send_time;
            double can_sync_time_s = 0.0;
            if (protocol_ == RuntimeProtocol::Broadcast) {
                for (const auto& frame : command_frames) {
                    std::string send_error;
                    if (!transport_->send_frame(frame, send_error)) emit_error(send_error);
                }

                std::string sync_error;
                sync_send_time = Clock::now();
                can_sync_time_s = time_point_seconds(sync_send_time);
                if (!transport_->send_frame(CanFrame{kBroadcastSyncId, {}}, sync_error)) emit_error(sync_error);
            } else {
                std::string sync_error;
                sync_send_time = Clock::now();
                can_sync_time_s = time_point_seconds(sync_send_time);
                if (!transport_->send_frame(CanFrame{kBroadcastSyncId, {}}, sync_error)) emit_error(sync_error);

                for (const auto& frame : command_frames) {
                    std::string send_error;
                    if (!transport_->send_frame(frame, send_error)) emit_error(send_error);
                }
            }

            std::map<uint16_t, bool> responded;
            for (const auto& [id, _] : boards_) responded[id] = false;
            const size_t expected_reply_count = boards_.size();
            size_t responded_count = 0;
            const auto rx_deadline = cycle_start + std::chrono::duration_cast<Clock::duration>(period * options_.rx_window_frac);
            while (Clock::now() < rx_deadline) {
                CanFrame reply;
                if (!transport_->recv_frame(reply, 1)) continue;
                uint16_t pressure = 0;
                uint8_t status = 0;
                if (!compact_response_from_frame(reply, pressure, status)) continue;
                const auto rx_time = Clock::now();
                std::lock_guard<std::mutex> lock(state_mutex_);
                auto it = boards_.find(reply.id);
                if (it == boards_.end()) {
                    unexpected_replies_++;
                    continue;
                }
                if (responded[reply.id]) {
                    duplicate_replies_++;
                    continue;
                }
                BoardState& board = it->second;
                board.pressure_filtered = pressure;
                board.pressure = pressure;
                board.pressure_calibrated = pressure_calibration_.apply(reply.id, pressure);
                board.status = status;
                if ((status & kStatusError) != 0) board.status_errors++;
                board.stale = false;
                board.last_reply = rx_time;
                board.replies++;
                board.latency_ms = std::chrono::duration<double, std::milli>(rx_time - sync_send_time).count();
                board.latency_sum_ms += board.latency_ms;
                board.latency_max_ms = std::max(board.latency_max_ms, board.latency_ms);
                responded[reply.id] = true;
                responded_count++;
                if (responded_count >= expected_reply_count) break;
            }

            int cycle_responded = 0;
            MocapSample mocap = mocap_.sample_at(cycle_time_s);
            MocapSample mocap_fixed_delay = mocap_.sample_at(cycle_time_s - kFixedDelayStateS);
            const auto observer_start = Clock::now();
            JointState joint_current = mocap_.joint_state_at(cycle_time_s);
            JointState joint_fixed_delay = mocap_.joint_state_at(cycle_time_s - kFixedDelayStateS);
            const double observer_time_ms = std::chrono::duration<double, std::milli>(Clock::now() - observer_start).count();
            observer_time_last_ms_ = observer_time_ms;
            observer_time_max_ms_ = std::max(observer_time_max_ms_, observer_time_ms);
            if (observer_time_ms > kObserverBudgetMs) observer_over_budget_count_++;

            const auto fk_start = Clock::now();
            FkResult fk_current;
            const bool fk_valid = joint_current.valid && umarm_forward_kinematics(joint_current.theta, fk_current);
            const double fk_time_ms = std::chrono::duration<double, std::milli>(Clock::now() - fk_start).count();
            fk_time_last_ms_ = fk_time_ms;
            fk_time_max_ms_ = std::max(fk_time_max_ms_, fk_time_ms);
            if (fk_time_ms > kFkBudgetMs) fk_over_budget_count_++;
            if (mocap.valid && std::isfinite(mocap.latency_ms)) {
                mocap_latency_samples_++;
                mocap_latency_sum_ms_ += mocap.latency_ms;
                mocap_latency_max_ms_ = std::max(mocap_latency_max_ms_, mocap.latency_ms);
                mocap_latency_last_ms_ = mocap.latency_ms;
                mocap_timestamp_offset_last_ms_ = mocap.timestamp_offset_ms;
                mocap_frame_rate_last_hz_ = mocap.frame_rate_hz;
                mocap_frame_drop_count_last_ = mocap.frame_drop_count;
            }
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                for (auto& [id, board] : boards_) {
                    if (responded[id]) {
                        cycle_responded++;
                    } else {
                        board.stale = true;
                        board.missed++;
                        total_missed_++;
                    }
                }

                RobotStateSample robot_state = make_robot_state_locked(cycle, cycle_time_s, can_sync_time_s, cycle_responded,
                                                                       static_cast<int>(boards_.size()), mocap, mocap_fixed_delay,
                                                                       joint_current, joint_fixed_delay, fk_current, fk_valid,
                                                                       observer_time_ms, fk_time_ms);

                for (auto& [id, board] : boards_) {
                    if (csv_.is_open()) {
                        csv_ << cycle << ',' << std::fixed << std::setprecision(6) << cycle_time_s << ',' << can_sync_time_s << ','
                             << runtime_protocol_name(protocol_) << ',' << can_order_name(options_.can_order) << ',' << std::setprecision(3) << options_.rx_window_frac << ','
                             << id_hex(id) << ','
                             << board.target << ',' << static_cast<int>(board.flags) << ',' << board.pressure_filtered << ','
                             << board.pressure_calibrated << ','
                             << static_cast<int>(board.status) << ',' << board.status_errors << ',' << (board.stale ? 1 : 0) << ',' << board.missed << ','
                             << std::setprecision(3) << board.latency_ms << ',' << cycle_responded << ',' << boards_.size() << ','
                             << jitter_ms << ',' << total_missed_ << ',' << unexpected_replies_ << ',' << duplicate_replies_ << ','
                             << (mocap.valid ? 1 : 0) << ',' << (mocap.stale ? 1 : 0) << ',' << robot_state.mocap_queue_age_ms << ','
                             << robot_state.mocap_extrapolation_ms << ','
                             << mocap.latency_ms << ',' << mocap.timestamp_s << ',' << mocap.raw_timestamp_s << ','
                             << mocap.timestamp_offset_ms << ',' << mocap.frame_rate_hz << ',' << mocap.frame_drop_count << ','
                             << mocap.clock_sample_count << ',' << mocap.clock_update_count << ','
                             << mocap.body_count << ','
                             << mocap.x << ',' << mocap.y << ',' << mocap.z << ',' << mocap.vx << ',' << mocap.vy << ',' << mocap.vz << ','
                             << (robot_state.joint_current_estimate.valid ? 1 : 0) << ',' << (robot_state.joint_fixed_delay.valid ? 1 : 0) << ','
                                << robot_state.joint_current_estimate.source_time_error_ms << ',' << robot_state.joint_fixed_delay.source_time_error_ms << ','
                                << (robot_state.joint_current_estimate.extrapolated ? 1 : 0) << ',' << (robot_state.joint_fixed_delay.extrapolated ? 1 : 0) << ','
                                << robot_state.joint_current_estimate.extrapolation_ms << ',' << robot_state.joint_fixed_delay.extrapolation_ms << ','
                                << robot_state.observer_time_ms << ',' << robot_state.observer_time_max_ms << ',' << robot_state.observer_over_budget_count << ','
                                << (robot_state.fk_valid ? 1 : 0) << ',' << robot_state.fk_time_ms << ',' << robot_state.fk_time_max_ms << ',' << robot_state.fk_over_budget_count << ','
                                << robot_state.viz_publish_time_ms << ',' << robot_state.viz_publish_time_max_ms << ',' << robot_state.viz_publish_over_budget_count;
                            append_joint_csv_values(robot_state.joint_current_estimate.theta);
                            append_joint_csv_values(robot_state.joint_current_estimate.theta_dot);
                            append_joint_csv_values(robot_state.joint_fixed_delay.theta);
                            append_joint_csv_values(robot_state.joint_fixed_delay.theta_dot);
                            csv_ << ',' << (robot_state.calibration_default ? 1 : 0) << '\n';
                    }
                }

                write_collection_sample_locked(robot_state, mocap, jitter_ms);

                last_robot_state_ = robot_state;
                publish_visualization_sample(robot_state);

                total_expected_ += boards_.size();
                total_responded_ += static_cast<uint64_t>(cycle_responded);
                cycle_count_++;

                // MPC consumers want fresh state every 150 Hz cycle; dashboards only need a
                // periodic heartbeat. robot_state streams per-cycle when requested, otherwise
                // it rides along with the once-per-second board/cycle/viz telemetry.
                const bool telemetry_tick = (cycle % telemetry_interval_cycles == 0);
                if (options_.stream_state_every_cycle || telemetry_tick) {
                    emit_robot_state(robot_state);
                }
                if (telemetry_tick) {
                    for (const auto& [id, board] : boards_) emit_board(board);
                    emit_cycle(cycle, cycle_responded, static_cast<int>(boards_.size()), jitter_ms, mocap);
                    emit_viz_status();
                }
            }

            if (stop_loop_after_collection) running_ = false;

            ++cycle;
            if (duration_s > 0 && Clock::now() - start_time >= std::chrono::seconds(duration_s)) break;
            next_cycle += std::chrono::duration_cast<Clock::duration>(period);
            wait_until_precise(next_cycle);
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (collection_active_ || collection_deflating_ || collection_chunk_.is_open()) {
                finish_collection_locked("shutdown", cycle, time_point_seconds(Clock::now()));
            }
        }
        send_disable_once();
        write_report();
        transport_->close();
        close_csv();
        emitter_.emit("{\"type\":\"backend\",\"state\":\"stopped\",\"report\":\"" + json_escape(report_path_.string()) + "\"}");
        running_ = false;
    #ifdef _WIN32
        timeEndPeriod(1);
    #endif
    }

    void emit_board(const BoardState& board)
    {
        emitter_.emit("{\"type\":\"board\",\"id\":\"" + id_hex(board.id) + "\",\"pressure\":" + std::to_string(board.pressure) +
                      ",\"pressure_adc_filtered\":" + std::to_string(board.pressure_filtered) +
                      ",\"pressure_calibrated\":" + number(board.pressure_calibrated, 3) +
                      ",\"pressure_psi\":" + number(board.pressure_calibrated, 3) +
                      ",\"target\":" + std::to_string(board.target) + ",\"status\":" + std::to_string(board.status) +
                      ",\"enabled\":" + std::string((board.status & kStatusEnabled) ? "true" : "false") +
                      ",\"ota\":" + std::string((board.status & kStatusOtaActive) ? "true" : "false") +
                      ",\"command_seen\":" + std::string((board.status & kStatusCommandSeen) ? "true" : "false") +
                      ",\"error\":" + std::string((board.status & kStatusError) ? "true" : "false") +
                      ",\"status_errors\":" + std::to_string(board.status_errors) +
                      ",\"stale\":" + std::string(board.stale ? "true" : "false") + ",\"missed\":" + std::to_string(board.missed) +
                      ",\"latency_ms\":" + number(board.latency_ms, 3) + "}");
    }

    void emit_cycle(uint64_t cycle, int responded, int expected, double jitter_ms, const MocapSample& mocap)
    {
        emitter_.emit("{\"type\":\"cycle\",\"cycle\":" + std::to_string(cycle) + ",\"rate_hz\":" + std::to_string(options_.rate_hz) +
                      ",\"can_protocol\":\"" + runtime_protocol_name(protocol_) + "\"" +
                      ",\"can_order\":\"" + can_order_name(options_.can_order) + "\"" +
                      ",\"rx_window_frac\":" + number(options_.rx_window_frac, 3) +
                      ",\"responded\":" + std::to_string(responded) + ",\"expected\":" + std::to_string(expected) +
                      ",\"jitter_ms\":" + number(jitter_ms, 3) +
                      ",\"total_missed\":" + std::to_string(total_missed_) +
                      ",\"unexpected_replies\":" + std::to_string(unexpected_replies_) +
                      ",\"duplicate_replies\":" + std::to_string(duplicate_replies_) +
                      ",\"mocap_valid\":" + std::string(mocap.valid ? "true" : "false") +
                      ",\"mocap_stale\":" + std::string(mocap.stale ? "true" : "false") +
                      ",\"mocap_frame\":" + std::to_string(mocap.frame) +
                      ",\"mocap_timestamp_s\":" + number(mocap.timestamp_s, 6) +
                      ",\"mocap_raw_timestamp_s\":" + number(mocap.raw_timestamp_s, 6) +
                      ",\"mocap_timestamp_offset_ms\":" + number(mocap.timestamp_offset_ms, 3) +
                      ",\"mocap_frame_rate_hz\":" + number(mocap.frame_rate_hz, 3) +
                      ",\"mocap_frame_drop_count\":" + std::to_string(mocap.frame_drop_count) +
                      ",\"mocap_clock_sample_count\":" + std::to_string(mocap.clock_sample_count) +
                      ",\"mocap_clock_update_count\":" + std::to_string(mocap.clock_update_count) +
                      ",\"mocap_age_ms\":" + number(mocap.age_ms, 3) +
                      ",\"mocap_latency_ms\":" + number(mocap.latency_ms, 3) +
                      ",\"mocap_body_count\":" + std::to_string(mocap.body_count) +
                      ",\"mocap_body_ids\":\"" + json_escape(mocap.body_ids) +
                      "\",\"mocap_body_points\":\"" + json_escape(mocap.body_points) + "\",\"mocap_x\":" + number(mocap.x, 4) +
                      ",\"mocap_y\":" + number(mocap.y, 4) + ",\"mocap_z\":" + number(mocap.z, 4) +
                      ",\"mocap_vx\":" + number(mocap.vx, 4) + ",\"mocap_vy\":" + number(mocap.vy, 4) +
                      ",\"mocap_vz\":" + number(mocap.vz, 4) + "}");
    }

    void emit_robot_state(const RobotStateSample& state)
    {
        emitter_.emit("{\"type\":\"robot_state\",\"cycle\":" + std::to_string(state.cycle) +
                      ",\"cycle_start_time_s\":" + number(state.cycle_start_time_s, 6) +
                      ",\"can_sync_time_s\":" + number(state.can_sync_time_s, 6) +
                      ",\"ids\":" + json_u16_array(state.ids) +
                      ",\"pressure_adc_filtered\":" + json_u16_array(state.pressure_adc_filtered) +
                      ",\"pressure_calibrated\":" + json_double_array(state.pressure_calibrated, 3) +
                      ",\"actuator_status\":" + json_u8_array(state.actuator_status) +
                      ",\"actuator_stale\":" + json_bool_array(state.actuator_stale) +
                      ",\"target_next_sync\":" + json_u16_array(state.target_next_sync) +
                      ",\"control_next_sync\":" + json_u8_array(state.control_next_sync) +
                      ",\"joint_current_valid\":" + std::string(state.joint_current_estimate.valid ? "true" : "false") +
                      ",\"joint_current_theta\":" + json_joint_array(state.joint_current_estimate.theta, 6) +
                      ",\"joint_current_theta_dot\":" + json_joint_array(state.joint_current_estimate.theta_dot, 6) +
                      ",\"joint_current_extrapolated\":" + std::string(state.joint_current_estimate.extrapolated ? "true" : "false") +
                      ",\"joint_current_extrapolation_ms\":" + number(state.joint_current_estimate.extrapolation_ms, 3) +
                      ",\"joint_current_source_error_ms\":" + number(state.joint_current_estimate.source_time_error_ms, 3) +
                      ",\"joint_fixed_delay_valid\":" + std::string(state.joint_fixed_delay.valid ? "true" : "false") +
                      ",\"joint_fixed_delay_s\":" + number(kFixedDelayStateS, 3) +
                      ",\"joint_fixed_delay_theta\":" + json_joint_array(state.joint_fixed_delay.theta, 6) +
                      ",\"joint_fixed_delay_theta_dot\":" + json_joint_array(state.joint_fixed_delay.theta_dot, 6) +
                      ",\"joint_fixed_delay_extrapolated\":" + std::string(state.joint_fixed_delay.extrapolated ? "true" : "false") +
                      ",\"joint_fixed_delay_extrapolation_ms\":" + number(state.joint_fixed_delay.extrapolation_ms, 3) +
                      ",\"joint_fixed_delay_source_error_ms\":" + number(state.joint_fixed_delay.source_time_error_ms, 3) +
                      ",\"observer_time_ms\":" + number(state.observer_time_ms, 3) +
                      ",\"observer_time_max_ms\":" + number(state.observer_time_max_ms, 3) +
                      ",\"observer_budget_ms\":" + number(kObserverBudgetMs, 3) +
                      ",\"observer_over_budget_count\":" + std::to_string(state.observer_over_budget_count) +
                      ",\"fk_valid\":" + std::string(state.fk_valid ? "true" : "false") +
                      ",\"fk_tip\":[" + number(state.fk_current.tip_position.x, 6) + "," + number(state.fk_current.tip_position.y, 6) + "," + number(state.fk_current.tip_position.z, 6) + "]" +
                      ",\"fk_time_ms\":" + number(state.fk_time_ms, 3) +
                      ",\"fk_time_max_ms\":" + number(state.fk_time_max_ms, 3) +
                      ",\"fk_budget_ms\":" + number(kFkBudgetMs, 3) +
                      ",\"fk_over_budget_count\":" + std::to_string(state.fk_over_budget_count) +
                      ",\"viz_publish_time_ms\":" + number(state.viz_publish_time_ms, 3) +
                      ",\"viz_publish_time_max_ms\":" + number(state.viz_publish_time_max_ms, 3) +
                      ",\"viz_publish_budget_ms\":" + number(kVizPublishBudgetMs, 3) +
                      ",\"viz_publish_over_budget_count\":" + std::to_string(state.viz_publish_over_budget_count) +
                      ",\"mocap_frame\":" + std::to_string(state.mocap_frame) +
                      ",\"mocap_timestamp_s\":" + number(state.mocap_timestamp_s, 6) +
                      ",\"mocap_raw_timestamp_s\":" + number(state.mocap_raw_timestamp_s, 6) +
                      ",\"mocap_timestamp_offset_ms\":" + number(state.mocap_timestamp_offset_ms, 3) +
                      ",\"mocap_frame_rate_hz\":" + number(state.mocap_frame_rate_hz, 3) +
                      ",\"mocap_frame_drop_count\":" + std::to_string(state.mocap_frame_drop_count) +
                      ",\"mocap_clock_sample_count\":" + std::to_string(state.mocap_clock_sample_count) +
                      ",\"mocap_clock_update_count\":" + std::to_string(state.mocap_clock_update_count) +
                      ",\"mocap_queue_age_ms\":" + number(state.mocap_queue_age_ms, 3) +
                      ",\"mocap_extrapolation_ms\":" + number(state.mocap_extrapolation_ms, 3) +
                      ",\"mocap_stale\":" + std::string(state.mocap_stale ? "true" : "false") +
                      ",\"cycle_responded\":" + std::to_string(state.cycle_responded) +
                      ",\"cycle_expected\":" + std::to_string(state.cycle_expected) +
                      ",\"can_order\":\"" + can_order_name(options_.can_order) + "\"" +
                      ",\"rx_window_frac\":" + number(options_.rx_window_frac, 3) +
                      ",\"total_missed\":" + std::to_string(total_missed_) +
                      ",\"unexpected_replies\":" + std::to_string(unexpected_replies_) +
                      ",\"duplicate_replies\":" + std::to_string(duplicate_replies_) +
                      ",\"calibration_default\":" + std::string(state.calibration_default ? "true" : "false") + "}");
    }

    std::string number(double value, int precision) const
    {
        std::ostringstream out;
        out << std::fixed << std::setprecision(precision) << value;
        return out.str();
    }

    void send_disable_once()
    {
        if (!transport_) return;
        std::string error;
        for (const auto& [id, _] : boards_) transport_->send_frame(compact_command_frame(id, 0, 0), error);
        transport_->send_frame(CanFrame{kBroadcastSyncId, {}}, error);
    }

    void drain_transport_for(std::chrono::milliseconds duration)
    {
        if (!transport_) return;
        const auto deadline = Clock::now() + duration;
        while (Clock::now() < deadline) {
            CanFrame ignored;
            if (!transport_->recv_frame(ignored, 1)) std::this_thread::yield();
        }
    }

    void write_report()
    {
        std::filesystem::create_directories(options_.log_dir);
        std::ofstream report(report_path_);
        const TransportStats stats = transport_ ? transport_->stats() : TransportStats{};
        report << "# VNEMA Backend Status Report\n\n";
        report << "- Mode: " << (options_.simulate_can ? "simulated CAN" : "live SLCAN") << "\n";
        report << "- Mocap: " << (options_.mocap_live ? "live NatNet" : (options_.mocap_sim ? "simulated" : "disabled")) << "\n";
        report << "- Mocap rigid IDs: " << options_.mocap_rigid_ids << "\n";
        report << "- Pressure calibration: " << pressure_calibration_.source() << "\n";
        report << "- Port: " << options_.port << "\n";
        report << "- CAN protocol: " << runtime_protocol_name(protocol_) << "\n";
        report << "- CAN order: " << can_order_name(options_.can_order) << "\n";
        report << "- RX window fraction: " << number(options_.rx_window_frac, 3) << "\n";
        report << "- Selected IDs: " << ids_csv() << "\n";
        report << "- Outputs enabled during run: " << (outputs_enabled_ ? "yes" : "no") << "\n";
        report << "- Cycles: " << cycle_count_ << "\n";
        report << "- Expected replies: " << total_expected_ << "\n";
        report << "- Received replies: " << total_responded_ << "\n";
        report << "- Missed replies: " << total_missed_ << "\n";
        report << "- Unexpected replies: " << unexpected_replies_ << "\n";
        report << "- Duplicate replies: " << duplicate_replies_ << "\n";
        report << "- Max cycle jitter ms: " << number(max_jitter_ms_, 3) << "\n";
        report << "- Observer budget ms: " << number(kObserverBudgetMs, 3) << "\n";
        report << "- Last observer time ms: " << number(observer_time_last_ms_, 3) << "\n";
        report << "- Max observer time ms: " << number(observer_time_max_ms_, 3) << "\n";
        report << "- Observer over-budget count: " << observer_over_budget_count_ << "\n";
        report << "- FK budget ms: " << number(kFkBudgetMs, 3) << "\n";
        report << "- Last FK time ms: " << number(fk_time_last_ms_, 3) << "\n";
        report << "- Max FK time ms: " << number(fk_time_max_ms_, 3) << "\n";
        report << "- FK over-budget count: " << fk_over_budget_count_ << "\n";
        report << "- Visualization shared memory: " << (viz_writer_.enabled() ? options_.viz_shm_name : "disabled") << "\n";
        report << "- Visualization publishes: " << viz_writer_.publish_count() << "\n";
        report << "- Visualization publish budget ms: " << number(kVizPublishBudgetMs, 3) << "\n";
        report << "- Last visualization publish time ms: " << number(viz_publish_time_last_ms_, 3) << "\n";
        report << "- Max visualization publish time ms: " << number(viz_publish_time_max_ms_, 3) << "\n";
        report << "- Visualization publish over-budget count: " << viz_publish_over_budget_count_ << "\n";
        if (mocap_latency_samples_ > 0) {
            report << "- Mocap latency samples: " << mocap_latency_samples_ << "\n";
            report << "- Last mocap latency ms: " << number(mocap_latency_last_ms_, 3) << "\n";
            report << "- Avg mocap latency ms: " << number(mocap_latency_sum_ms_ / static_cast<double>(mocap_latency_samples_), 3) << "\n";
            report << "- Max mocap latency ms: " << number(mocap_latency_max_ms_, 3) << "\n";
            report << "- Last mocap timestamp offset ms: " << number(mocap_timestamp_offset_last_ms_, 3) << "\n";
            report << "- Last mocap frame rate Hz: " << number(mocap_frame_rate_last_hz_, 3) << "\n";
            report << "- Mocap frame drops: " << mocap_frame_drop_count_last_ << "\n";
            report << "- Mocap clock sample count: " << last_robot_state_.mocap_clock_sample_count << "\n";
            report << "- Mocap clock 5s updates: " << last_robot_state_.mocap_clock_update_count << "\n";
        }
        report << "- Transport TX frames: " << stats.tx_frames << "\n";
        report << "- Transport RX frames: " << stats.rx_frames << "\n";
        report << "- Transport parse errors: " << stats.parse_errors << "\n";
        report << "- CSV log: " << csv_path_.string() << "\n\n";
        if (!collection_last_session_dir_.empty()) {
            report << "## Real System Data Collection\n\n";
            report << "- Session directory: " << collection_last_session_dir_.string() << "\n";
            report << "- Logged samples: " << collection_last_total_samples_ << "\n";
            report << "- Checkpoints: " << collection_last_checkpoint_count_ << "\n";
            report << "- Data format: JSON Lines, one row per 150 Hz sync cycle\n";
            report << "- Pressure/input units: raw ADC counts\n\n";
        }
        report << "| ID | Replies | Missed | Status Errors | Avg Latency ms | Max Latency ms | Last Filtered ADC | Last Calibrated | Status |\n";
        report << "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
        std::lock_guard<std::mutex> lock(state_mutex_);
        for (const auto& [id, board] : boards_) {
            const double avg = board.replies == 0 ? 0.0 : board.latency_sum_ms / static_cast<double>(board.replies);
                 report << "| " << id_hex(id) << " | " << board.replies << " | " << board.missed << " | " << board.status_errors << " | "
                     << number(avg, 3) << " | " << number(board.latency_max_ms, 3) << " | " << board.pressure_filtered << " | "
                   << number(board.pressure_calibrated, 3) << " | " << static_cast<int>(board.status) << " |\n";
        }
    }

    Options options_;
    JsonEmitter& emitter_;
    std::map<uint16_t, BoardState> boards_;
    RuntimeProtocol protocol_ = RuntimeProtocol::Unicast;
    std::unique_ptr<ICanTransport> transport_;
    MocapRuntime mocap_;
    VizSharedMemoryWriter viz_writer_;
    PressureCalibration pressure_calibration_;
    RobotStateSample last_robot_state_;
    std::atomic<bool> running_{false};
    std::atomic<bool> request_exit_{false};
    std::thread worker_;
    std::mutex lifecycle_mutex_;
    std::mutex state_mutex_;
    bool outputs_enabled_ = false;
    std::ofstream csv_;
    std::filesystem::path csv_path_;
    std::filesystem::path report_path_;
    std::string run_name_;
    bool collection_start_requested_ = false;
    bool collection_stop_requested_ = false;
    bool collection_active_ = false;
    bool collection_deflating_ = false;
    bool collection_stop_loop_on_complete_ = false;
    double collection_requested_duration_s_ = 0.0;
    std::string collection_finish_reason_ = "complete";
    Clock::time_point collection_start_time_{};
    Clock::time_point collection_deflate_start_time_{};
    std::filesystem::path collection_session_dir_;
    std::filesystem::path collection_last_session_dir_;
    std::string collection_session_tag_;
    std::ofstream collection_chunk_;
    std::filesystem::path collection_chunk_temp_path_;
    uint64_t collection_chunk_index_ = 0;
    uint64_t collection_chunk_samples_ = 0;
    uint64_t collection_chunk_start_cycle_ = 0;
    double collection_chunk_start_time_s_ = 0.0;
    uint64_t collection_total_samples_ = 0;
    uint64_t collection_last_total_samples_ = 0;
    uint64_t collection_checkpoint_count_ = 0;
    uint64_t collection_last_checkpoint_count_ = 0;
    std::vector<CollectionChunkInfo> collection_chunks_;
    std::map<uint16_t, double> collection_norm_;
    std::map<uint16_t, double> collection_velocity_;
    std::mt19937 collection_rng_{std::random_device{}()};
    uint64_t cycle_count_ = 0;
    uint64_t total_expected_ = 0;
    uint64_t total_responded_ = 0;
    uint64_t total_missed_ = 0;
    uint64_t unexpected_replies_ = 0;
    uint64_t duplicate_replies_ = 0;
    double max_jitter_ms_ = 0.0;
    uint64_t mocap_latency_samples_ = 0;
    double mocap_latency_sum_ms_ = 0.0;
    double mocap_latency_max_ms_ = 0.0;
    double mocap_latency_last_ms_ = 0.0;
    double mocap_timestamp_offset_last_ms_ = 0.0;
    double mocap_frame_rate_last_hz_ = 0.0;
    uint64_t mocap_frame_drop_count_last_ = 0;
    double observer_time_last_ms_ = 0.0;
    double observer_time_max_ms_ = 0.0;
    uint64_t observer_over_budget_count_ = 0;
    double fk_time_last_ms_ = 0.0;
    double fk_time_max_ms_ = 0.0;
    uint64_t fk_over_budget_count_ = 0;
    double viz_publish_time_last_ms_ = 0.0;
    double viz_publish_time_max_ms_ = 0.0;
    uint64_t viz_publish_over_budget_count_ = 0;
};

bool run_self_test()
{
    bool ok = true;
    const uint16_t packed = pack_compact(0x1234, 0x0A);
    uint16_t pressure = 0;
    uint8_t flags = 0;
    unpack_compact(packed, pressure, flags);
    ok = ok && pressure == 0x0234 && flags == 0x0A;

    const CanFrame frame = compact_command_frame(0x101, 1234, kControlEnable);
    const std::string encoded = encode_slcan(frame);
    CanFrame decoded;
    ok = ok && decode_slcan(encoded, decoded) && decoded.id == frame.id && decoded.data == frame.data;

    FakeTransport fake;
    std::string error;
    ok = ok && fake.open(error);
    ok = ok && fake.send_frame(frame, error);
    CanFrame reply;
    ok = ok && fake.recv_frame(reply, 5) && reply.id == 0x101;
    ok = ok && compact_response_from_frame(reply, pressure, flags) && (flags & kStatusCommandSeen);

    std::vector<BoardCommand> order_test{{0x101, 0, 0}, {0x102, 0, 0}, {0x103, 0, 0}};
    apply_can_order(order_test, CanOrder::Reverse, 0);
    ok = ok && order_test[0].id == 0x103 && order_test[2].id == 0x101;
    apply_can_order(order_test, CanOrder::Rotate, 1);
    ok = ok && order_test[0].id == 0x102;

    const auto parsed_targets = json_int_array_value("{\"cmd\":\"set_targets\",\"targets\":[1500, 800,0 ,4095]}", "targets");
    ok = ok && parsed_targets && parsed_targets->size() == 4 && (*parsed_targets)[0] == 1500 && (*parsed_targets)[2] == 0 && (*parsed_targets)[3] == 4095;
    ok = ok && !json_int_array_value("{\"targets\":\"oops\"}", "targets").has_value();
    ok = ok && !json_int_array_value("{\"other\":[1,2]}", "targets").has_value();

    MocapClockCalibrator clock;
    MocapClockResult clock_result;
    double host_s = 1000.0;
    double raw_s = 10.0;
    constexpr double host_period_s = 1.0 / 360.0;
    constexpr double raw_period_s = host_period_s * 1.000025;
    for (uint64_t frame_index = 0; frame_index < 7200; ++frame_index) {
        host_s += host_period_s;
        raw_s += raw_period_s;
        clock_result = clock.update(raw_s, host_s, frame_index);
    }
    ok = ok && std::abs(clock_result.calibrated_timestamp_s - host_s) < 0.010;
    ok = ok && clock_result.frame_rate_hz > 350.0 && clock_result.frame_rate_hz < 370.0;
    ok = ok && clock_result.update_count >= 3;

    auto close_enough = [](double left, double right, double tolerance) { return std::abs(left - right) <= tolerance; };

    Matrix3 identity_rotation;
    ok = ok && quaternion_to_matrix(0.0, 0.0, 0.0, 1.0, identity_rotation);
    ok = ok && close_enough(identity_rotation[0][0], 1.0, 1e-12) && close_enough(identity_rotation[1][1], 1.0, 1e-12) && close_enough(identity_rotation[2][2], 1.0, 1e-12);

    Matrix3 align_x_to_z;
    ok = ok && rotation_from_to(Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 0.0, 1.0}, align_x_to_z);
    const Vec3 aligned = matrix_vector(align_x_to_z, Vec3{1.0, 0.0, 0.0});
    ok = ok && close_enough(aligned.x, 0.0, 1e-12) && close_enough(aligned.y, 0.0, 1e-12) && close_enough(aligned.z, 1.0, 1e-12);

    double theta1 = 1.0;
    double theta2 = 1.0;
    ok = ok && angle_pair(Vec3{0.0, 0.0, 1.0}, theta1, theta2) && close_enough(theta1, 0.0, 1e-12) && close_enough(theta2, 0.0, 1e-12);

    std::array<RigidBodyPose, kRigidBodyCount> straight_poses{};
    const Matrix3 identity = identity_matrix();
    for (size_t index = 0; index < kRigidBodyCount; ++index) {
        straight_poses[index].position = Vec3{0.0, 0.0, -static_cast<double>(index)};
        straight_poses[index].rotation = identity;
        straight_poses[index].valid = true;
    }
    std::array<double, kJointCount> straight_q{};
    ok = ok && mocap_poses_to_q_mk8(straight_poses, (1u << kRigidBodyCount) - 1u, straight_q);
    ok = ok && std::all_of(straight_q.begin(), straight_q.end(), [&](double value) { return close_enough(value, 0.0, 1e-12); });

    std::deque<MocapKinematicSample> q_samples;
    MocapKinematicSample q0;
    MocapKinematicSample q1;
    q0.valid = true;
    q1.valid = true;
    q0.timestamp_s = 10.0;
    q1.timestamp_s = 11.0;
    q0.theta.fill(0.0);
    q1.theta.fill(0.2);
    q_samples.push_back(q0);
    q_samples.push_back(q1);
    JointState interpolated = joint_state_from_kinematic_samples(q_samples, 10.5);
    ok = ok && interpolated.valid && close_enough(interpolated.theta[0], 0.1, 1e-12) && close_enough(interpolated.theta_dot[0], 0.2, 1e-12);

    q_samples.clear();
    q0.theta.fill(0.0);
    q1.theta.fill(0.0);
    q0.theta[0] = 3.0;
    q1.theta[0] = -3.0;
    q_samples.push_back(q0);
    q_samples.push_back(q1);
    JointState wrapped = joint_state_from_kinematic_samples(q_samples, 10.5);
    ok = ok && wrapped.valid && wrapped.theta_dot[0] > 0.0 && wrapped.theta_dot[0] < 0.4;

    FkResult fk_zero;
    std::array<double, kJointCount> q_zero{};
    ok = ok && umarm_forward_kinematics(q_zero, fk_zero) && fk_zero.valid;
    ok = ok && close_enough(fk_zero.ujoint_centers[0].x, 0.0, 1e-12) && close_enough(fk_zero.ujoint_centers[0].y, 0.0, 1e-12) && close_enough(fk_zero.ujoint_centers[0].z, 1.2, 1e-12);
    ok = ok && close_enough(fk_zero.ujoint_centers[1].z, 0.93495, 1e-12);
    ok = ok && close_enough(fk_zero.ujoint_centers[2].z, 0.86246, 1e-12);
    ok = ok && close_enough(fk_zero.ujoint_centers[3].z, 0.62832, 1e-12);
    ok = ok && close_enough(fk_zero.ujoint_centers[4].z, 0.55506, 1e-12);
    ok = ok && close_enough(fk_zero.ujoint_centers[5].z, 0.3234, 1e-12);
    ok = ok && close_enough(fk_zero.tip_position.z, 0.19352015, 1e-12);

    MocapSample base_frame_sample;
    base_frame_sample.body_mask = 1u;
    base_frame_sample.body_centers[0] = 1.0;
    base_frame_sample.body_centers[1] = 2.0;
    base_frame_sample.body_centers[2] = 3.0;
    base_frame_sample.body_rotations[0] = 0.0;
    base_frame_sample.body_rotations[1] = -1.0;
    base_frame_sample.body_rotations[2] = 0.0;
    base_frame_sample.body_rotations[3] = 1.0;
    base_frame_sample.body_rotations[4] = 0.0;
    base_frame_sample.body_rotations[5] = 0.0;
    base_frame_sample.body_rotations[6] = 0.0;
    base_frame_sample.body_rotations[7] = 0.0;
    base_frame_sample.body_rotations[8] = 1.0;
    FkTransform mocap_from_fk;
    ok = ok && make_mocap_from_fk_transform(fk_zero, base_frame_sample, mocap_from_fk);
    const FkVec3 mapped_base = transform_point(mocap_from_fk, fk_zero.ujoint_centers[0]);
    const FkVec3 mapped_x_axis = transform_point(mocap_from_fk, FkVec3{fk_zero.ujoint_centers[0].x + 1.0, fk_zero.ujoint_centers[0].y, fk_zero.ujoint_centers[0].z});
    ok = ok && close_enough(mapped_base.x, 1.0, 1e-12) && close_enough(mapped_base.y, 2.0, 1e-12) && close_enough(mapped_base.z, 3.0, 1e-12);
    ok = ok && close_enough(mapped_x_axis.x, 1.0, 1e-12) && close_enough(mapped_x_axis.y, 3.0, 1e-12) && close_enough(mapped_x_axis.z, 3.0, 1e-12);

    std::array<double, kJointCount> q_nonzero{};
    for (size_t index = 0; index < q_nonzero.size(); ++index) q_nonzero[index] = 0.01 * static_cast<double>(index + 1);
    FkResult fk_nonzero;
    ok = ok && umarm_forward_kinematics(q_nonzero, fk_nonzero) && fk_nonzero.valid;
    ok = ok && std::isfinite(fk_nonzero.tip_position.x) && std::isfinite(fk_nonzero.tip_position.y) && std::isfinite(fk_nonzero.tip_position.z);

    if (ok) {
        std::cout << "self-test OK" << std::endl;
        return true;
    }
    std::cerr << "self-test FAILED" << std::endl;
    return false;
}

Options parse_args(int argc, char** argv)
{
    Options options;
    options.ids = default_dt_ids();
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
            return argv[++i];
        };
        if (arg == "--self-test") options.self_test = true;
        else if (arg == "--simulate-can") options.simulate_can = true;
        else if (arg == "--mocap-sim") options.mocap_sim = true;
        else if (arg == "--mocap-live") options.mocap_live = true;
        else if (arg == "--mocap-server") options.mocap_server = require_value(arg);
        else if (arg == "--mocap-local") options.mocap_local = require_value(arg);
        else if (arg == "--mocap-rigid-ids") options.mocap_rigid_ids = require_value(arg);
        else if (arg == "--mocap-python") options.mocap_python = require_value(arg);
        else if (arg == "--mocap-script") options.mocap_script = require_value(arg);
        else if (arg == "--mocap-multicast") options.mocap_multicast = true;
        else if (arg == "--mocap-unicast") options.mocap_multicast = false;
        else if (arg == "--viz-enable") options.viz_enable = true;
        else if (arg == "--viz-shm-name") options.viz_shm_name = require_value(arg);
        else if (arg == "--status-only") options.status_only = true;
        else if (arg == "--stream-state") options.stream_state_every_cycle = true;
        else if (arg == "--can-protocol") options.runtime_protocol = parse_runtime_protocol(require_value(arg));
        else if (arg == "--can-order") options.can_order = parse_can_order(require_value(arg));
        else if (arg == "--rx-window-frac") options.rx_window_frac = clamp_rx_window_frac(std::strtod(require_value(arg).c_str(), nullptr));
        else if (arg == "--port") options.port = require_value(arg);
        else if (arg == "--tty-baud") options.tty_baud = parse_int_auto(require_value(arg));
        else if (arg == "--rate") options.rate_hz = parse_int_auto(require_value(arg));
        else if (arg == "--duration") options.duration_s = parse_int_auto(require_value(arg));
        else if (arg == "--ids") options.ids = parse_id_list(require_value(arg));
        else if (arg == "--calibration") options.calibration_path = require_value(arg);
        else if (arg == "--log-dir") options.log_dir = require_value(arg);
        else if (arg == "--disabled-safe-start") {}
        else throw std::runtime_error("unknown argument: " + arg);
    }
    return options;
}

} // namespace vnema

int main(int argc, char** argv)
{
    try {
        vnema::Options options = vnema::parse_args(argc, argv);
        if (options.self_test) return vnema::run_self_test() ? 0 : 1;

        vnema::JsonEmitter emitter;
        vnema::BackendRuntime runtime(options, emitter);

        if (options.status_only || options.duration_s > 0) {
            runtime.start_loop(options.duration_s > 0 ? options.duration_s : 10);
            runtime.wait_until_stopped();
            return 0;
        }

        runtime.emit_ready();
        std::thread input_thread([&runtime]() {
            std::string line;
            while (std::getline(std::cin, line)) {
                if (!line.empty()) runtime.process_command(line);
                if (runtime.request_exit()) break;
            }
            if (!runtime.request_exit()) runtime.process_command("{\"cmd\":\"shutdown\"}");
        });

        while (!runtime.request_exit()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (input_thread.joinable()) input_thread.join();
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "backend error: " << exc.what() << std::endl;
        return 2;
    }
}
