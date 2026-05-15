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

std::vector<int8_t> convert_f16_to_int8(const std::vector<half_float::half>& f16_data) {
    std::vector<int8_t> int8_data(f16_data.size());
    for (size_t i = 0; i < f16_data.size(); ++i) {
        float val = static_cast<float>(f16_data[i]);
        int8_data[i] = static_cast<int8_t>(val)*127.0f;
    }
    return int8_data;
}

#include <immintrin.h>

int32_t dot_product_int8(const int8_t* a, const int8_t* b, size_t d) {
    int32_t sum = 0;
    size_t i = 0;
#ifdef __AVX2__
    __m256i vsum = _mm256_setzero_si256();
    for (; i + 31 < d; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        
        __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 0));
        __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 0));
        __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
        
        __m256i vprod_lo = _mm256_madd_epi16(va_lo, vb_lo);
        __m256i vprod_hi = _mm256_madd_epi16(va_hi, vb_hi);
        
        vsum = _mm256_add_epi32(vsum, vprod_lo);
        vsum = _mm256_add_epi32(vsum, vprod_hi);
    }
    int32_t sums[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(sums), vsum);
    for (int j = 0; j < 8; ++j) sum += sums[j];
#endif
    for (; i < d; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

} // namespace math_utils
