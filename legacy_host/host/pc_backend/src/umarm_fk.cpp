#include "umarm_fk.h"

#include <algorithm>
#include <cmath>

namespace vnema {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kBasePosition[3] = {0.0, 0.0, 1.2};
constexpr double kBaseEulerDeg[3] = {0.0, 0.0, 0.0};
constexpr double kSegmentRodLengthM[kFkSegmentCount] = {0.26505, 0.23414, 0.23166};
constexpr double kSegmentTwistDeg[kFkSegmentCount] = {45.0, 45.0, 45.0};
constexpr double kInterSegmentOffsetM[kFkSegmentCount - 1][3] = {
    {0.0, 0.0, 0.07249},
    {0.0, 0.0, 0.07326},
};
constexpr double kInterSegmentEulerDeg[kFkSegmentCount - 1][3] = {
    {0.0, 0.0, -45.0},
    {0.0, 0.0, -45.0},
};
constexpr double kEndEffectorRodLengthM = 0.12987985;
constexpr const char* kRobotConfigSignature = "original_umarm_3seg_fk_20260522";

bool finite(double value)
{
    return std::isfinite(value);
}

bool finite(const FkTransform& transform)
{
    return std::all_of(transform.begin(), transform.end(), [](double value) { return finite(value); });
}

FkTransform identity_transform()
{
    FkTransform out{};
    out[0] = 1.0;
    out[5] = 1.0;
    out[10] = 1.0;
    out[15] = 1.0;
    return out;
}

FkTransform multiply_transform(const FkTransform& left, const FkTransform& right)
{
    FkTransform out{};
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
            double value = 0.0;
            for (std::size_t inner = 0; inner < 4; ++inner) {
                value += left[4 * row + inner] * right[4 * inner + col];
            }
            out[4 * row + col] = value;
        }
    }
    return out;
}

FkTransform translate(double x_m, double y_m, double z_m)
{
    FkTransform out = identity_transform();
    out[3] = x_m;
    out[7] = y_m;
    out[11] = z_m;
    return out;
}

FkTransform rotate_x(double theta_rad)
{
    const double c = std::cos(theta_rad);
    const double s = std::sin(theta_rad);
    FkTransform out = identity_transform();
    out[5] = c;
    out[6] = -s;
    out[9] = s;
    out[10] = c;
    return out;
}

FkTransform rotate_y(double theta_rad)
{
    const double c = std::cos(theta_rad);
    const double s = std::sin(theta_rad);
    FkTransform out = identity_transform();
    out[0] = c;
    out[2] = s;
    out[8] = -s;
    out[10] = c;
    return out;
}

FkTransform rotate_z(double theta_rad)
{
    const double c = std::cos(theta_rad);
    const double s = std::sin(theta_rad);
    FkTransform out = identity_transform();
    out[0] = c;
    out[1] = -s;
    out[4] = s;
    out[5] = c;
    return out;
}

FkTransform euler_xyz(const double euler_deg[3])
{
    FkTransform out = multiply_transform(rotate_x(euler_deg[0] * kPi / 180.0), rotate_y(euler_deg[1] * kPi / 180.0));
    out = multiply_transform(out, rotate_z(euler_deg[2] * kPi / 180.0));
    return out;
}

FkVec3 translation_of(const FkTransform& transform)
{
    return FkVec3{transform[3], transform[7], transform[11]};
}

} // namespace

bool umarm_forward_kinematics(const std::array<double, kFkJointCount>& q_rad, FkResult& result)
{
    result = FkResult{};
    if (!std::all_of(q_rad.begin(), q_rad.end(), [](double value) { return finite(value); })) return false;

    FkTransform transform = multiply_transform(translate(kBasePosition[0], kBasePosition[1], kBasePosition[2]), euler_xyz(kBaseEulerDeg));
    std::size_t ujoint_index = 0;
    for (std::size_t segment_index = 0; segment_index < kFkSegmentCount; ++segment_index) {
        if (segment_index > 0) {
            const double* offset = kInterSegmentOffsetM[segment_index - 1];
            transform = multiply_transform(transform, translate(offset[0], offset[1], -offset[2]));
            transform = multiply_transform(transform, euler_xyz(kInterSegmentEulerDeg[segment_index - 1]));
        }

        result.ujoint_centers[ujoint_index] = translation_of(transform);
        result.ujoint_transforms[ujoint_index] = transform;
        ++ujoint_index;

        const std::size_t q_index = 4 * segment_index;
        transform = multiply_transform(transform, rotate_x(q_rad[q_index]));
        transform = multiply_transform(transform, rotate_y(q_rad[q_index + 1]));
        transform = multiply_transform(transform, translate(0.0, 0.0, -kSegmentRodLengthM[segment_index]));
        transform = multiply_transform(transform, rotate_z(kSegmentTwistDeg[segment_index] * kPi / 180.0));

        result.ujoint_centers[ujoint_index] = translation_of(transform);
        result.ujoint_transforms[ujoint_index] = transform;
        ++ujoint_index;

        transform = multiply_transform(transform, rotate_x(q_rad[q_index + 2]));
        transform = multiply_transform(transform, rotate_y(q_rad[q_index + 3]));
    }

    result.tip_transform = multiply_transform(transform, translate(0.0, 0.0, -kEndEffectorRodLengthM));
    result.tip_position = translation_of(result.tip_transform);
    result.valid = finite(result.tip_transform);
    return result.valid;
}

const char* umarm_fk_robot_config_signature()
{
    return kRobotConfigSignature;
}

} // namespace vnema
