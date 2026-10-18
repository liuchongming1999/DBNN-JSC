#include <math.h>
#include <stddef.h>

/* Same causal reset equation, evaluated in O(n) for training diagnostics. */
size_t dbnn_reset(const double *drive, double *voltage, double *events,
                  size_t n, double theta, double amplitude, double tau) {
    const double alpha = exp(-1.0 / tau);
    double r = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        voltage[i] = drive[i] + r;
        const int fire = voltage[i] >= theta;
        if (fire) events[count++] = (double)i;
        r = alpha * r - (fire ? amplitude : 0.0);
    }
    return count;
}
