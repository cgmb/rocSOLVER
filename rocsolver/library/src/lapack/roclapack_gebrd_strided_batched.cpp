/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gebrd.hpp"

template <typename S, typename T, typename U>
rocblas_status rocsolver_gebrd_strided_batched_impl(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n, U A,
    const rocblas_int lda, const rocblas_stride strideA, S *D,
    const rocblas_stride strideD, S *E, const rocblas_stride strideE, T *tauq,
    const rocblas_stride strideQ, T *taup, const rocblas_stride strideP,
    const rocblas_int batch_count) {
  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st = rocsolver_gebd2_gebrd_argCheck(m, n, lda, A, D, E, tauq,
                                                     taup, batch_count);
  if (st != rocblas_status_continue)
    return st;

  rocblas_stride strideX = m * GEBRD_GEBD2_SWITCHSIZE;
  rocblas_stride strideY = n * GEBRD_GEBD2_SWITCHSIZE;

  // memory managment
  size_t size_1; // size of constants
  size_t size_2; // size of workspace
  size_t size_3; // size of array of pointers to workspace
  size_t size_4; // size of cache for norms and diag elements
  size_t size_5; // size of matrix X
  size_t size_6; // size of matrix Y
  rocsolver_gebrd_getMemorySize<T, false>(m, n, batch_count, &size_1, &size_2,
                                          &size_3, &size_4, &size_5, &size_6);

  // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
  void *scalars, *work, *workArr, *diag, *X, *Y;
  hipMalloc(&scalars, size_1);
  hipMalloc(&work, size_2);
  hipMalloc(&workArr, size_3);
  hipMalloc(&diag, size_4);
  hipMalloc(&X, size_5);
  hipMalloc(&Y, size_6);
  if (!scalars || (size_2 && !work) || (size_3 && !workArr) ||
      (size_4 && !diag) || (size_5 && !X) || (size_6 && !Y))
    return rocblas_status_memory_error;

  // scalar constants for rocblas functions calls
  // (to standarize and enable re-use, size_1 always equals 3*sizeof(T))
  T sca[] = {-1, 0, 1};
  RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca, size_1, hipMemcpyHostToDevice));

  // execution
  rocblas_status status = rocsolver_gebrd_template<false, true, S, T>(
      handle, m, n, A,
      0, // the matrix is shifted 0 entries (will work on the entire matrix)
      lda, strideA, D, strideD, E, strideE, tauq, strideQ, taup, strideP, (U)X,
      0, // the matrix is shifted 0 entries (will work on the entire matrix)
      m, strideX, (U)Y,
      0, // the matrix is shifted 0 entries (will work on the entire matrix)
      n, strideY, batch_count, (T *)scalars, (T *)work, (T **)workArr,
      (T *)diag);

  hipFree(scalars);
  hipFree(work);
  hipFree(workArr);
  hipFree(diag);
  hipFree(X);
  hipFree(Y);
  return status;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgebrd_strided_batched(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n, float *A,
    const rocblas_int lda, const rocblas_stride strideA, float *D,
    const rocblas_stride strideD, float *E, const rocblas_stride strideE,
    float *tauq, const rocblas_stride strideQ, float *taup,
    const rocblas_stride strideP, const rocblas_int batch_count) {
  return rocsolver_gebrd_strided_batched_impl<float, float>(
      handle, m, n, A, lda, strideA, D, strideD, E, strideE, tauq, strideQ,
      taup, strideP, batch_count);
}

rocblas_status rocsolver_dgebrd_strided_batched(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n, double *A,
    const rocblas_int lda, const rocblas_stride strideA, double *D,
    const rocblas_stride strideD, double *E, const rocblas_stride strideE,
    double *tauq, const rocblas_stride strideQ, double *taup,
    const rocblas_stride strideP, const rocblas_int batch_count) {
  return rocsolver_gebrd_strided_batched_impl<double, double>(
      handle, m, n, A, lda, strideA, D, strideD, E, strideE, tauq, strideQ,
      taup, strideP, batch_count);
}

rocblas_status rocsolver_cgebrd_strided_batched(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n,
    rocblas_float_complex *A, const rocblas_int lda,
    const rocblas_stride strideA, float *D, const rocblas_stride strideD,
    float *E, const rocblas_stride strideE, rocblas_float_complex *tauq,
    const rocblas_stride strideQ, rocblas_float_complex *taup,
    const rocblas_stride strideP, const rocblas_int batch_count) {
  return rocsolver_gebrd_strided_batched_impl<float, rocblas_float_complex>(
      handle, m, n, A, lda, strideA, D, strideD, E, strideE, tauq, strideQ,
      taup, strideP, batch_count);
}

rocblas_status rocsolver_zgebrd_strided_batched(
    rocblas_handle handle, const rocblas_int m, const rocblas_int n,
    rocblas_double_complex *A, const rocblas_int lda,
    const rocblas_stride strideA, double *D, const rocblas_stride strideD,
    double *E, const rocblas_stride strideE, rocblas_double_complex *tauq,
    const rocblas_stride strideQ, rocblas_double_complex *taup,
    const rocblas_stride strideP, const rocblas_int batch_count) {
  return rocsolver_gebrd_strided_batched_impl<double, rocblas_double_complex>(
      handle, m, n, A, lda, strideA, D, strideD, E, strideE, tauq, strideQ,
      taup, strideP, batch_count);
}

} // extern C
