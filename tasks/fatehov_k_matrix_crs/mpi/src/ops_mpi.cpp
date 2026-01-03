#include "fatehov_k_matrix_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "fatehov_k_matrix_crs/common/include/common.hpp"

namespace fatehov_k_matrix_crs {

namespace {

void BroadcastMatrixSizes(size_t &rows, size_t &cols, int rank, const InType &input) {
  if (rank == 0) {
    rows = std::get<0>(input);
    cols = std::get<1>(input);
  }
  MPI_Bcast(&rows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
}

void BroadcastMatrixB(std::vector<double> &val_b, std::vector<size_t> &col_b, std::vector<size_t> &ptr_b, size_t &nnz_b,
                      size_t rows, int rank, const InType &input) {
  if (rank == 0) {
    val_b = std::get<3>(input);
    col_b = std::get<5>(input);
    ptr_b = std::get<7>(input);
    nnz_b = val_b.size();
  }
  MPI_Bcast(&nnz_b, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    val_b.resize(nnz_b);
    col_b.resize(nnz_b);
  }
  if (nnz_b > 0) {
    MPI_Bcast(val_b.data(), static_cast<int>(nnz_b), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_b.data(), static_cast<int>(nnz_b), MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  }
  ptr_b.resize(rows + 1);
  MPI_Bcast(ptr_b.data(), static_cast<int>(rows) + 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
}

void BroadcastMatrixAStructure(std::vector<size_t> &ptr_a, size_t rows, int rank, const InType &input) {
  ptr_a.resize(rows + 1);
  if (rank == 0) {
    ptr_a = std::get<6>(input);
  }
  MPI_Bcast(ptr_a.data(), static_cast<int>(rows) + 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
}

void DistributeLocalWork(int &local_rows, int &start_row, int &end_row, size_t rows, int size, int rank) {
  int rows_per_proc = static_cast<int>(rows) / size;
  int rem = static_cast<int>(rows) % size;
  local_rows = rows_per_proc + (rank < rem ? 1 : 0);
  start_row = (rank * rows_per_proc) + std::min(rank, rem);
  end_row = start_row + local_rows;
}

void ScatterMatrixA(std::vector<double> &val_a_loc, std::vector<size_t> &col_a_loc, const std::vector<size_t> &ptr_a,
                    int start_row, int end_row, int rank, int size, const InType &input) {
  size_t local_nnz = ptr_a[end_row] - ptr_a[start_row];
  val_a_loc.resize(local_nnz);
  col_a_loc.resize(local_nnz);

  if (rank == 0) {
    const auto &values_a = std::get<2>(input);
    const auto &cols_a = std::get<4>(input);

    for (int i = 1; i < size; ++i) {
      int i_rows_per_proc = static_cast<int>(ptr_a.size() - 1) / size;
      int i_rem = static_cast<int>(ptr_a.size() - 1) % size;
      int i_local_rows = i_rows_per_proc + (i < i_rem ? 1 : 0);
      int i_start_row = (i * i_rows_per_proc) + std::min(i, i_rem);
      int i_end_row = i_start_row + i_local_rows;
      size_t sz = ptr_a[i_end_row] - ptr_a[i_start_row];
      if (sz > 0) {
        MPI_Send(&values_a[ptr_a[i_start_row]], static_cast<int>(sz), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        MPI_Send(&cols_a[ptr_a[i_start_row]], static_cast<int>(sz), MPI_UNSIGNED_LONG, i, 1, MPI_COMM_WORLD);
      }
    }

    std::copy(values_a.begin() + static_cast<ptrdiff_t>(ptr_a[start_row]),
              values_a.begin() + static_cast<ptrdiff_t>(ptr_a[end_row]), val_a_loc.begin());
    std::copy(cols_a.begin() + static_cast<ptrdiff_t>(ptr_a[start_row]),
              cols_a.begin() + static_cast<ptrdiff_t>(ptr_a[end_row]), col_a_loc.begin());
  } else if (local_nnz > 0) {
    MPI_Recv(val_a_loc.data(), static_cast<int>(local_nnz), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(col_a_loc.data(), static_cast<int>(local_nnz), MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void ComputeLocalResult(const std::vector<double> &val_a_loc, const std::vector<size_t> &col_a_loc,
                        const std::vector<double> &val_b, const std::vector<size_t> &col_b,
                        const std::vector<size_t> &ptr_b, std::vector<double> &res_loc, int local_rows, size_t cols,
                        const std::vector<size_t> &ptr_a, int start_row) {
  for (int i = 0; i < local_rows; ++i) {
    size_t row_start_offset = ptr_a[start_row + i] - ptr_a[start_row];
    size_t row_end_offset = ptr_a[start_row + i + 1] - ptr_a[start_row];

    for (size_t k = row_start_offset; k < row_end_offset; ++k) {
      double a_val = val_a_loc[k];
      size_t a_col = col_a_loc[k];

      for (size_t j = ptr_b[a_col]; j < ptr_b[a_col + 1]; ++j) {
        size_t index = static_cast<size_t>(i) * cols + col_b[j];
        res_loc[index] += a_val * val_b[j];
      }
    }
  }
}

void GatherResults(std::vector<double> &full_res, const std::vector<double> &res_loc, int local_rows, size_t rows,
                   size_t cols, int rank, int size, OutType &output) {
  std::vector<int> counts(size);
  std::vector<int> displs(size);
  int send_cnt = local_rows * static_cast<int>(cols);
  MPI_Gather(&send_cnt, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    full_res.resize(rows * cols);
    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + counts[i - 1];
    }
  }

  MPI_Gatherv(res_loc.data(), send_cnt, MPI_DOUBLE, full_res.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

  if (rank == 0) {
    output = std::move(full_res);
  } else {
    output.resize(rows * cols);
  }

  MPI_Bcast(output.data(), static_cast<int>(rows * cols), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

}  // namespace

FatehovKMatrixCRSMPI::FatehovKMatrixCRSMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool FatehovKMatrixCRSMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int is_valid = 0;
  if (rank == 0) {
    auto &data = GetInput();
    size_t rows = std::get<0>(data);
    size_t cols = std::get<1>(data);
    if (rows > 0 && cols > 0 && rows <= kMaxRows && cols <= kMaxCols) {
      is_valid = 1;
    }
  }
  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return is_valid == 1;
}

bool FatehovKMatrixCRSMPI::PreProcessingImpl() {
  return true;
}

bool FatehovKMatrixCRSMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();

  size_t rows = 0;
  size_t cols = 0;
  BroadcastMatrixSizes(rows, cols, rank, input);

  std::vector<double> val_b{};
  std::vector<size_t> col_b{};
  std::vector<size_t> ptr_b{};
  size_t nnz_b = 0;
  BroadcastMatrixB(val_b, col_b, ptr_b, nnz_b, rows, rank, input);

  std::vector<size_t> ptr_a{};
  BroadcastMatrixAStructure(ptr_a, rows, rank, input);

  int local_rows = 0;
  int start_row = 0;
  int end_row = 0;
  DistributeLocalWork(local_rows, start_row, end_row, rows, size, rank);

  std::vector<double> val_a_loc{};
  std::vector<size_t> col_a_loc{};
  ScatterMatrixA(val_a_loc, col_a_loc, ptr_a, start_row, end_row, rank, size, input);

  std::vector<double> res_loc(static_cast<size_t>(local_rows) * cols, 0.0);
  ComputeLocalResult(val_a_loc, col_a_loc, val_b, col_b, ptr_b, res_loc, local_rows, cols, ptr_a, start_row);

  std::vector<double> full_res{};
  auto &output = GetOutput();
  GatherResults(full_res, res_loc, local_rows, rows, cols, rank, size, output);

  return true;
}

bool FatehovKMatrixCRSMPI::PostProcessingImpl() {
  return true;
}

}  // namespace fatehov_k_matrix_crs
