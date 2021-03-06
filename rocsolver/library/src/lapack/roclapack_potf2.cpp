/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_potf2.hpp"

template <typename T, typename U>
rocblas_status rocsolver_potf2_impl(rocblas_handle handle,
                                    const rocblas_fill uplo,
                                    const rocblas_int n, U A,
                                    const rocblas_int lda, rocblas_int *info) {
  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st = rocsolver_potf2_potrf_argCheck(uplo, n, lda, A, info);
  if (st != rocblas_status_continue)
    return st;

  rocblas_stride strideA = 0;
  rocblas_int batch_count = 1;

  // memory managment
  size_t size_1; // size of constants
  size_t size_2; // size of workspace
  size_t size_3;
  rocsolver_potf2_getMemorySize<T>(n, batch_count, &size_1, &size_2, &size_3);

  // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
  void *scalars, *work, *pivotGPU;
  hipMalloc(&scalars, size_1);
  hipMalloc(&work, size_2);
  hipMalloc(&pivotGPU, size_3);
  if (!scalars || (size_2 && !work) || (size_3 && !pivotGPU))
    return rocblas_status_memory_error;

  // scalar constants for rocblas functions calls
  // (to standarize and enable re-use, size_1 always equals 3*sizeof(T))
  T sca[] = {-1, 0, 1};
  RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca, size_1, hipMemcpyHostToDevice));

  // execution
  rocblas_status status = rocsolver_potf2_template<T>(
      handle, uplo, n, A,
      0, // the matrix is shifted 0 entries (will work on the entire matrix)
      lda, strideA, info, batch_count, (T *)scalars, (T *)work, (T *)pivotGPU);

  hipFree(scalars);
  hipFree(work);
  hipFree(pivotGPU);
  return status;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_spotf2(rocblas_handle handle, const rocblas_fill uplo,
                                const rocblas_int n, float *A,
                                const rocblas_int lda, rocblas_int *info) {
  return rocsolver_potf2_impl<float>(handle, uplo, n, A, lda, info);
}

rocblas_status rocsolver_dpotf2(rocblas_handle handle, const rocblas_fill uplo,
                                const rocblas_int n, double *A,
                                const rocblas_int lda, rocblas_int *info) {
  return rocsolver_potf2_impl<double>(handle, uplo, n, A, lda, info);
}

rocblas_status rocsolver_cpotf2(rocblas_handle handle, const rocblas_fill uplo,
                                const rocblas_int n, rocblas_float_complex *A,
                                const rocblas_int lda, rocblas_int *info) {
  return rocsolver_potf2_impl<rocblas_float_complex>(handle, uplo, n, A, lda,
                                                     info);
}

rocblas_status rocsolver_zpotf2(rocblas_handle handle, const rocblas_fill uplo,
                                const rocblas_int n, rocblas_double_complex *A,
                                const rocblas_int lda, rocblas_int *info) {
  return rocsolver_potf2_impl<rocblas_double_complex>(handle, uplo, n, A, lda,
                                                      info);
}
}
