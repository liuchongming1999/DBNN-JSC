#include <cmath>
#include <cstdint>

// Original hard-reset recurrence, for CPU validation/calibration only.
extern "C" std::int64_t reset_voltage(const double* drive, double* voltage,
                                      std::int64_t* events, std::int64_t length,
                                      double threshold, double amp,
                                      double tau_samples) {
    const double alpha = std::exp(-1.0 / tau_samples);
    double recovery = 0.0;
    std::int64_t count = 0;
    for (std::int64_t i = 0; i < length; ++i) {
        voltage[i] = drive[i] + recovery;
        const bool fired = voltage[i] >= threshold;
        recovery = alpha * recovery - (fired ? amp : 0.0);
        if (fired) events[count++] = i;
    }
    return count;
}
