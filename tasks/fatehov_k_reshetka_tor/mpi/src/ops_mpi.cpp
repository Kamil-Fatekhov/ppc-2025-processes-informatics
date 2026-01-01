#include "fatehov_k_reshetka_tor/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

namespace fatehov_k_reshetka_tor {

FatehovKReshetkaTorMPI::FatehovKReshetkaTorMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool FatehovKReshetkaTorMPI::ValidationImpl() {
  auto &data = GetInput();
  size_t rows = std::get<0>(data);
  size_t cols = std::get<1>(data);
  auto &vec = std::get<2>(data);

  return (rows > 0 && rows <= kMaxRows) && (cols > 0 && cols <= kMaxCols) && (rows * cols <= kMaxMatrixSize) &&
         (vec.size() == rows * cols) && (!vec.empty());
}

bool FatehovKReshetkaTorMPI::PreProcessingImpl() {
  return true;
}

bool FatehovKReshetkaTorMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  size_t total_rows = 0;
  size_t total_cols = 0;
  std::vector<double> global_matrix;

  if (world_rank == 0) {
    auto &data = GetInput();
    total_rows = std::get<0>(data);
    total_cols = std::get<1>(data);
    global_matrix = std::get<2>(data);
  }

  MPI_Bcast(&total_rows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&total_cols, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  int dims[2] = {0, 0};
  MPI_Dims_create(world_size, 2, dims);

  if (dims[0] * dims[1] != world_size) {
    if (world_rank == 0) {
      std::cerr << "Error: Cannot create 2D grid with " << world_size << " processes" << std::endl;
    }
    return false;
  }

  int periods[2] = {1, 1};
  int reorder = 0;
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

  int cart_rank;
  int coords[2];
  MPI_Comm_rank(cart_comm, &cart_rank);
  MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

  size_t rows_per_proc = total_rows / dims[0];
  size_t rem_rows = total_rows % dims[0];
  size_t cols_per_proc = total_cols / dims[1];
  size_t rem_cols = total_cols % dims[1];

  size_t proc_row = static_cast<size_t>(coords[0]);
  size_t proc_col = static_cast<size_t>(coords[1]);

  size_t my_rows = rows_per_proc + (proc_row < rem_rows ? 1 : 0);
  size_t my_cols = cols_per_proc + (proc_col < rem_cols ? 1 : 0);

  std::vector<double> local_matrix(my_rows * my_cols);

  if (world_rank == 0) {
    for (int proc = 0; proc < world_size; ++proc) {
      int proc_coords[2];
      MPI_Cart_coords(cart_comm, proc, 2, proc_coords);

      size_t target_row = static_cast<size_t>(proc_coords[0]);
      size_t target_col = static_cast<size_t>(proc_coords[1]);

      size_t proc_rows = rows_per_proc + (target_row < rem_rows ? 1 : 0);
      size_t proc_cols = cols_per_proc + (target_col < rem_cols ? 1 : 0);

      size_t proc_start_row = target_row * rows_per_proc + std::min<size_t>(target_row, rem_rows);
      size_t proc_start_col = target_col * cols_per_proc + std::min<size_t>(target_col, rem_cols);

      std::vector<double> buffer(proc_rows * proc_cols);
      for (size_t i = 0; i < proc_rows; ++i) {
        for (size_t j = 0; j < proc_cols; ++j) {
          size_t global_i = proc_start_row + i;
          size_t global_j = proc_start_col + j;
          buffer[i * proc_cols + j] = global_matrix[global_i * total_cols + global_j];
        }
      }

      if (proc == 0) {
        local_matrix = buffer;
      } else {
        int buffer_size = static_cast<int>(buffer.size());
        MPI_Send(buffer.data(), buffer_size, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    int local_size = static_cast<int>(local_matrix.size());
    MPI_Recv(local_matrix.data(), local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  double local_max = -1e18;
  if (!local_matrix.empty()) {
    for (double val : local_matrix) {
      double heavy_val = val;
      for (int k = 0; k < 100; ++k) {
        heavy_val = std::sin(heavy_val) * std::cos(heavy_val) + std::exp(std::complex<double>(0, heavy_val).real()) +
                    std::sqrt(std::abs(heavy_val) + 1.0);
        if (std::isinf(heavy_val)) {
          heavy_val = val;
        }
      }

      if (heavy_val > local_max) {
        local_max = heavy_val;
      }
    }
  }

  double global_max;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, cart_comm);

  GetOutput() = global_max;

  MPI_Bcast(&global_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Comm_free(&cart_comm);
  return true;
}

bool FatehovKReshetkaTorMPI::PostProcessingImpl() {
  return true;
}

}  // namespace fatehov_k_reshetka_tor
