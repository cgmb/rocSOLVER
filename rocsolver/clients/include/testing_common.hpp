/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

enum rocsolver_device : int {
  rsHost   = 1 << 0,
  rsDevice = 1 << 1,
};

enum rocsolver_blas_vector_init : bool {
  rsSeedReset = true,
  rsNoSeedReset = false
};
