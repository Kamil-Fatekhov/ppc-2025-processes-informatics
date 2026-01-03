#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "fatehov_k_matrix_crs/common/include/common.hpp"
#include "fatehov_k_matrix_crs/mpi/include/ops_mpi.hpp"
#include "fatehov_k_matrix_crs/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace fatehov_k_matrix_crs {

class FatehovKRunPerfTestsMatrixCRS : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_ = std::make_tuple(0, 0, std::vector<double>{}, std::vector<double>{}, std::vector<size_t>{},
                                       std::vector<size_t>{}, std::vector<size_t>{}, std::vector<size_t>{});
  OutType expected_result_;

  void SetUp() override {
    const size_t rows = 1000;
    const size_t cols = 1000;
    const double sparsity = 0.02;

    std::vector<double> values{};
    std::vector<double> values2{};
    std::vector<size_t> col_indices{};
    std::vector<size_t> col_indices2{};
    std::vector<size_t> row_ptr(rows + 1, 0);
    std::vector<size_t> row_ptr2(rows + 1, 0);

    uint64_t state = 42;
    const uint64_t a = 1664525ULL;
    const uint64_t c = 1013904223ULL;
    const uint64_t m = (1ULL << 22);

    for (size_t i = 0; i < rows; ++i) {
      size_t nnz_in_row = 0;
      for (size_t j = 0; j < cols; ++j) {
        state = (a * state + c) % m;
        if (static_cast<double>(state % 1000) / 1000.0 < sparsity) {
          double value = ((static_cast<double>(state) / m) * 20.0) - 10.0;
          values.push_back(value);
          col_indices.push_back(j);
          nnz_in_row++;
        }
      }
      row_ptr[i + 1] = row_ptr[i] + nnz_in_row;
    }

    state = 123;
    for (size_t i = 0; i < rows; ++i) {
      size_t nnz_in_row = 0;
      for (size_t j = 0; j < cols; ++j) {
        state = (a * state + c) % m;
        if (static_cast<double>(state % 1000) / 1000.0 < sparsity) {
          double value = ((static_cast<double>(state) / m) * 20.0) - 10.0;
          values2.push_back(value);
          col_indices2.push_back(j);
          nnz_in_row++;
        }
      }
      row_ptr2[i + 1] = row_ptr2[i] + nnz_in_row;
    }

    std::vector<double> computed_result(rows * cols, 0.0);

    for (size_t i = 0; i < rows; ++i) {
      for (size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
        size_t col_a = col_indices[k];
        double val_a = values[k];

        for (size_t j = row_ptr2[col_a]; j < row_ptr2[col_a + 1]; ++j) {
          size_t col_b = col_indices2[j];
          double val_b = values2[j];

          size_t index = (i * cols) + col_b;
          computed_result[index] += val_a * val_b;
        }
      }
    }

    expected_result_ = computed_result;
    input_data_ = std::make_tuple(rows, cols, values, values2, col_indices, col_indices2, row_ptr, row_ptr2);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_result_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::fabs(expected_result_[i] - output_data[i]) > 1e-10) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(FatehovKRunPerfTestsMatrixCRS, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, FatehovKMatrixCRSMPI, FatehovKMatrixCRSSEQ>(PPC_SETTINGS_fatehov_k_matrix_crs);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = FatehovKRunPerfTestsMatrixCRS::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunPerfTest, FatehovKRunPerfTestsMatrixCRS, kGtestValues, kPerfTestName);

}  // namespace fatehov_k_matrix_crs
