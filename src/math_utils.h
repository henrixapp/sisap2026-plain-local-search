#pragma once
#include <highfive/highfive.hpp>

//
#include <cstdint>
#include <highfive/half_float.hpp>
#include <vector>

namespace math_utils {

/**
 * @brief Calculate dynamic dimensionality to explain `target_variance` of data.
 *
 * @param eigenvalues std::vector of eigenvalues sorted from largest to
 * smallest.
 * @param target_variance A float like 0.95.
 * @return The number of dimensions needed to retain `target_variance`.
 */
int calculate_dynamic_d_out(const std::vector<float> &eigenvalues,
                            float target_variance);

/**
 * @brief Converts an array of 16-bit floats to 32-bit floats.
 *
 * @param f16_data Pointer to the int16_t (fp16) dataset.
 * @param num_elements Number of elements to convert.
 * @return std::vector<float> An allocated array of fp32 elements.
 */
std::vector<float>
convert_f16_to_f32(const std::vector<half_float::half> &f16_data);

std::vector<int8_t>
convert_f16_to_int8(const std::vector<half_float::half> &f16_data);

int32_t dot_product_int8(const int8_t* a, const int8_t* b, size_t d);

} // namespace math_utils
