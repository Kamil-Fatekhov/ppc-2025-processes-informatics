#include "fatehov_k_matrix_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "fatehov_k_matrix_crs/common/include/common.hpp"

namespace fatehov_k_matrix_crs {

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
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t rows = 0;
  size_t cols = 0;
  if (rank == 0) {
    rows = std::get<0>(GetInput());
    cols = std::get<1>(GetInput());
  }
  MPI_Bcast(&rows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  std::vector<double> valB = {};
  std::vector<size_t> colB{};
  std::vector<size_t> ptrB(rows + 1, 0);
  size_t nnzB = 0;

  if (rank == 0) {
    valB = std::get<3>(GetInput());
    colB = std::get<5>(GetInput());
    ptrB = std::get<7>(GetInput());
    nnzB = valB.size();
  }
  MPI_Bcast(&nnzB, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    valB.resize(nnzB);
    colB.resize(nnzB);
  }

  if (nnzB > 0) {
    MPI_Bcast(valB.data(), (int)nnzB, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(colB.data(), (int)nnzB, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  }
  MPI_Bcast(ptrB.data(), (int)rows + 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  std::vector<size_t> ptrA(rows + 1);
  if (rank == 0) {
    ptrA = std::get<6>(GetInput());
  }
  MPI_Bcast(ptrA.data(), (int)rows + 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  int rows_per_proc = (int)rows / size;
  int rem = (int)rows % size;
  int local_rows = rows_per_proc + (rank < rem ? 1 : 0);
  int start_row = rank * rows_per_proc + std::min(rank, rem);
  int end_row = start_row + local_rows;

  size_t local_nnz = ptrA[end_row] - ptrA[start_row];
  std::vector<double> valA_loc(local_nnz);
  std::vector<size_t> colA_loc(local_nnz);

  if (rank == 0) {
    const auto &vA = std::get<2>(GetInput());
    const auto &cA = std::get<4>(GetInput());
    for (int i = 1; i < size; ++i) {
      int s = (i * rows_per_proc) + std::min(i, rem);
      int e = s + rows_per_proc + (i < rem ? 1 : 0);
      size_t sz = ptrA[e] - ptrA[s];
      if (sz > 0) {
        MPI_Send(&vA[ptrA[s]], (int)sz, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        MPI_Send(&cA[ptrA[s]], (int)sz, MPI_UNSIGNED_LONG, i, 1, MPI_COMM_WORLD);
      }
    }
    std::copy(vA.begin() + ptrA[start_row], vA.begin() + ptrA[end_row], valA_loc.begin());
    std::copy(cA.begin() + ptrA[start_row], cA.begin() + ptrA[end_row], colA_loc.begin());
  } else if (local_nnz > 0) {
    MPI_Recv(valA_loc.data(), (int)local_nnz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(colA_loc.data(), (int)local_nnz, MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  std::vector<double> res_loc(local_rows * cols, 0.0);
  for (int i = 0; i < local_rows; ++i) {
    size_t row_start_offset = ptrA[start_row + i] - ptrA[start_row];
    size_t row_end_offset = ptrA[start_row + i + 1] - ptrA[start_row];

    for (size_t k = row_start_offset; k < row_end_offset; ++k) {
      double a_val = valA_loc[k];
      size_t a_col = colA_loc[k];

      for (size_t j = ptrB[a_col]; j < ptrB[a_col + 1]; ++j) {
        res_loc[i * cols + colB[j]] += a_val * valB[j];
      }
    }
  }

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  int send_cnt = local_rows * (int)cols;
  MPI_Gather(&send_cnt, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<double> full_res = {};
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
    GetOutput() = std::move(full_res);
  } else {
    GetOutput().resize(rows * cols);
  }

  MPI_Bcast(GetOutput().data(), (int)(rows * cols), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool FatehovKMatrixCRSMPI::PostProcessingImpl() {
  return true;
}

}  // namespace fatehov_k_matrix_crs
