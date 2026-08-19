#ifndef FAST_LIO_TIME_UTILS_H
#define FAST_LIO_TIME_UTILS_H

#include <builtin_interfaces/msg/time.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

inline bool try_get_ros_time(double timestamp, builtin_interfaces::msg::Time &time)
{
    if (!std::isfinite(timestamp) || timestamp < 0.0)
        return false;

    double integral_seconds = 0.0;
    const double fractional_seconds = std::modf(timestamp, &integral_seconds);
    if (integral_seconds > std::numeric_limits<int32_t>::max())
        return false;

    auto nanoseconds = static_cast<uint64_t>(std::llround(fractional_seconds * 1e9));
    if (nanoseconds >= 1000000000ULL)
    {
        integral_seconds += 1.0;
        nanoseconds -= 1000000000ULL;
    }
    if (integral_seconds > std::numeric_limits<int32_t>::max())
        return false;

    time.sec = static_cast<int32_t>(integral_seconds);
    time.nanosec = static_cast<uint32_t>(nanoseconds);
    return true;
}

inline bool try_apply_time_offset(double timestamp, double offset, double &adjusted_timestamp)
{
    if (!std::isfinite(timestamp) || !std::isfinite(offset))
        return false;

    adjusted_timestamp = timestamp + offset;
    return std::isfinite(adjusted_timestamp) && adjusted_timestamp >= 0.0;
}

#endif
