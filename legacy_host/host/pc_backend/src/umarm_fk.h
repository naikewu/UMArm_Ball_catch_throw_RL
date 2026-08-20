#pragma once

#include <array>
#include <cstddef>

namespace vnema {

constexpr std::size_t kFkJointCount = 12;
constexpr std::size_t kFkUJointCount = 6;
constexpr std::size_t kFkSegmentCount = 3;

struct FkVec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

using FkTransform = std::array<double, 16>;

struct FkResult {
    std::array<FkVec3, kFkUJointCount> ujoint_centers{};
    std::array<FkTransform, kFkUJointCount> ujoint_transforms{};
    FkTransform tip_transform{};
    FkVec3 tip_position{};
    bool valid = false;
};

bool umarm_forward_kinematics(const std::array<double, kFkJointCount>& q_rad, FkResult& result);
const char* umarm_fk_robot_config_signature();

} // namespace vnema
