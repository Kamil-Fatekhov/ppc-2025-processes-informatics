#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace fatehov_k_matrix_crs {

using InType = std::tuple<size_t, size_t, std::vector<double>, std::vector<double>, std::vector<size_t>,
                          std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>;
using OutType = std::vector<double>;
using TestType = std::tuple<int, size_t, size_t, std::vector<double>, std::vector<double>, std::vector<size_t>,
                            std::vector<size_t>, std::vector<size_t>, std::vector<size_t>, std::vector<double>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace fatehov_k_matrix_crs
