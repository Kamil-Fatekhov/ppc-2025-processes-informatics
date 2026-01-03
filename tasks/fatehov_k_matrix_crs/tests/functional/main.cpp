#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "fatehov_k_matrix_crs/common/include/common.hpp"
#include "fatehov_k_matrix_crs/mpi/include/ops_mpi.hpp"
#include "fatehov_k_matrix_crs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace fatehov_k_matrix_crs {

class FatehovKRunFuncTestsMatrixCRS : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string out = std::to_string(std::get<0>(test_param)) + "_matrix_" + std::to_string(std::get<1>(test_param)) +
                      "x" + std::to_string(std::get<2>(test_param));
    std::ranges::replace(out, '-', 'm');
    std::ranges::replace(out, '.', '_');
    return out;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    size_t rows = std::get<1>(params);
    size_t cols = std::get<2>(params);
    std::vector<double> values = std::get<3>(params);
    std::vector<double> values2 = std::get<4>(params);
    std::vector<size_t> col_indices = std::get<5>(params);
    std::vector<size_t> col_indices2 = std::get<6>(params);
    std::vector<size_t> row_ptr = std::get<7>(params);
    std::vector<size_t> row_ptr2 = std::get<8>(params);
    std::vector<double> expected = std::get<9>(params);

    input_data_ = std::make_tuple(rows, cols, values, values2, col_indices, col_indices2, row_ptr, row_ptr2);
    expected_result_ = expected;
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

 private:
  InType input_data_ = std::make_tuple(0, 0, std::vector<double>{}, std::vector<double>{}, std::vector<size_t>{},
                                       std::vector<size_t>{}, std::vector<size_t>{}, std::vector<size_t>{});
  OutType expected_result_;
};

namespace {

TEST_P(FatehovKRunFuncTestsMatrixCRS, MatrixMultiplicationCRS) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(1, 2, 2, std::vector<double>{1.0, 0.0, 0.0, 1.0}, std::vector<double>{2.0, 3.0, 4.0, 5.0},
                    std::vector<size_t>{0, 1, 0, 1}, std::vector<size_t>{0, 1, 0, 1}, std::vector<size_t>{0, 2, 4},
                    std::vector<size_t>{0, 2, 4}, std::vector<double>{2.0, 3.0, 4.0, 5.0}),

    std::make_tuple(2, 2, 2, std::vector<double>{2.0, 0.0, 0.0, 2.0}, std::vector<double>{1.0, 2.0, 3.0, 4.0},
                    std::vector<size_t>{0, 1, 0, 1}, std::vector<size_t>{0, 1, 0, 1}, std::vector<size_t>{0, 2, 4},
                    std::vector<size_t>{0, 2, 4}, std::vector<double>{2.0, 4.0, 6.0, 8.0}),

    std::make_tuple(3, 2, 2, std::vector<double>{1.0, 0.0, 0.0, 2.0}, std::vector<double>{1.0, 2.0, 3.0, 4.0},
                    std::vector<size_t>{0, 1, 0, 1}, std::vector<size_t>{0, 1, 0, 1}, std::vector<size_t>{0, 2, 4},
                    std::vector<size_t>{0, 2, 4}, std::vector<double>{1.0, 2.0, 6.0, 8.0})};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<FatehovKMatrixCRSMPI, InType>(kTestParam, PPC_SETTINGS_fatehov_k_matrix_crs),
                   ppc::util::AddFuncTask<FatehovKMatrixCRSSEQ, InType>(kTestParam, PPC_SETTINGS_fatehov_k_matrix_crs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = FatehovKRunFuncTestsMatrixCRS::PrintFuncTestName<FatehovKRunFuncTestsMatrixCRS>;

INSTANTIATE_TEST_SUITE_P(TestMatrixCRS, FatehovKRunFuncTestsMatrixCRS, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace fatehov_k_matrix_crs
