#include "math_utils.h"

namespace math_utils {

int calculate_dynamic_d_out(const std::vector<float>& eigenvalues, float target_variance) {
    float total_sum = 0.0f;
    for (float ev : eigenvalues) {
        total_sum += ev;
    }

    float target_sum = total_sum * target_variance;
    float running_sum = 0.0f;
    
    int d_out = 0;
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        running_sum += eigenvalues[i];
        d_out++;
        if (running_sum >= target_sum) {
            break;
        }
    }
    
    return d_out;
}

std::vector<float> convert_f16_to_f32(const std::vector<half_float::half>& f16_data) {
    std::vector<float> f32_data(f16_data.size());
    for (size_t i = 0; i < f16_data.size(); ++i) {
        f32_data[i] = static_cast<float>(f16_data[i]);
    }
    return f32_data;
}

} // namespace math_utils
