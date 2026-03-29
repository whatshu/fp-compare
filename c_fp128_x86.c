#include <math.h>
#include <quadmath.h>
#include <stddef.h>

// Compute mean absolute error of truncated exp Taylor series in float64.
double mean_abs_error_exp_taylor_f64(int degree, int points) {
    if (degree < 0 || points <= 0) return -1.0;

    double sum_err = 0.0;
    for (int i = 0; i <= points; ++i) {
        double x = -1.0 + 2.0 * (double)i / (double)points;

        double acc = 1.0;
        double term = 1.0;
        for (int k = 1; k <= degree; ++k) {
            term *= x / (double)k;
            acc += term;
        }

        double truth = exp(x);
        sum_err += fabs(acc - truth);
    }
    return sum_err / (double)(points + 1);
}

// Compute mean absolute error of truncated exp Taylor series in __float128.
double mean_abs_error_exp_taylor_f128(int degree, int points) {
    if (degree < 0 || points <= 0) return -1.0;

    __float128 sum_err = 0.0Q;
    for (int i = 0; i <= points; ++i) {
        __float128 x = -1.0Q + 2.0Q * (__float128)i / (__float128)points;

        __float128 acc = 1.0Q;
        __float128 term = 1.0Q;
        for (int k = 1; k <= degree; ++k) {
            term *= x / (__float128)k;
            acc += term;
        }

        __float128 truth = expq(x);
        __float128 err = fabsq(acc - truth);
        sum_err += err;
    }

    __float128 mean_err = sum_err / (__float128)(points + 1);
    return (double)mean_err;
}
