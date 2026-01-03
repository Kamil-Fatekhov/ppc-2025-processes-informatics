#pragma once

#include "fatehov_k_matrix_crs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace fatehov_k_matrix_crs {

class FatehovKMatrixCRSMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit FatehovKMatrixCRSMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static const int kMaxRows = 10000;
  static const int kMaxCols = 10000;
  static const int kMaxNonZero = 10000000;
};

}  // namespace fatehov_k_matrix_crs
