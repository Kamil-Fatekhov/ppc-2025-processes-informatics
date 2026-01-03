#include "fatehov_k_matrix_crs/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "fatehov_k_matrix_crs/common/include/common.hpp"

namespace fatehov_k_matrix_crs {

FatehovKMatrixCRSSEQ::FatehovKMatrixCRSSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool FatehovKMatrixCRSSEQ::ValidationImpl() {
  auto &data = GetInput();
  size_t rows = std::get<0>(data);
  size_t cols = std::get<1>(data);
  auto &values = std::get<2>(data);
  auto &values2 = std::get<3>(data);
  auto &col_indices = std::get<4>(data);
  auto &col_indices2 = std::get<5>(data);
  auto &row_ptr = std::get<6>(data);
  auto &row_ptr2 = std::get<7>(data);

  return (rows > 0 && rows <= kMaxRows) && (cols > 0 && cols <= kMaxCols) && (values.size() == col_indices.size()) &&
         (values2.size() == col_indices2.size()) && (values.size() <= kMaxNonZero) && (values2.size() <= kMaxNonZero) &&
         (row_ptr.size() == rows + 1) && (row_ptr2.size() == rows + 1) && (!values.empty()) && (!values2.empty());
}

bool FatehovKMatrixCRSSEQ::PreProcessingImpl() {
  return true;
}

bool FatehovKMatrixCRSSEQ::RunImpl() {
  auto &data = GetInput();
  size_t rows = std::get<0>(data);
  size_t cols = std::get<1>(data);
  auto &values = std::get<2>(data);
  auto &values2 = std::get<3>(data);
  auto &col_indices = std::get<4>(data);
  auto &col_indices2 = std::get<5>(data);
  auto &row_ptr = std::get<6>(data);
  auto &row_ptr2 = std::get<7>(data);

  std::vector<double> result(rows * cols, 0.0);

  for (size_t i = 0; i < rows; ++i) {
    for (size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
      size_t colA = col_indices[k];
      double valA = values[k];

      for (size_t j = row_ptr2[colA]; j < row_ptr2[colA + 1]; ++j) {
        size_t colB = col_indices2[j];
        double valB = values2[j];

        result[i * cols + colB] += valA * valB;
      }
    }
  }

  GetOutput() = result;
  return true;
}

bool FatehovKMatrixCRSSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace fatehov_k_matrix_crs
