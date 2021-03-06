/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getri.hpp"

/*
 * ===========================================================================
 *    getri_outofplace_batched is not intended for inclusion in the public API.
 *    It exists to provide a getri_batched method with a signature identical to
 *    the cuBLAS implementation, for use exclusively in hipBLAS.
 * ===========================================================================
 */

template <typename T, typename U>
rocblas_status rocsolver_getri_outofplace_batched_impl(
    rocblas_handle handle, const rocblas_int n, U A, const rocblas_int lda,
    rocblas_int *ipiv, const rocblas_stride strideP, U C, const rocblas_int ldc,
    rocblas_int *info, const rocblas_int batch_count) {
  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st =
      rocsolver_getri_argCheck(n, lda, ldc, A, C, ipiv, info, batch_count);
  if (st != rocblas_status_continue)
    return st;

  rocblas_stride strideA = 0;
  rocblas_stride strideC = 0;

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  // memory managment
  size_t size_1; // size of constants
  size_t size_2; // for workspace and TRSM invA
  size_t size_3; // for array of pointers to workspace
  size_t size_4; // for TRSM x_temp and TRTRI c_temp
  size_t size_5; // for TRSM x_temp_arr
  size_t size_6; // for TRSM invA_arr
  rocsolver_getri_getMemorySize<true, false, T>(
      n, batch_count, &size_1, &size_2, &size_3, &size_4, &size_5, &size_6);

  // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
  void *scalars, *work, *workArr, *x_temp, *x_temp_arr, *invA_arr;
  bool optim_mem = true;

  hipMalloc(&scalars, size_1);
  hipMalloc(&work, size_2);
  hipMalloc(&workArr, size_3);
  hipMalloc(&x_temp, size_4);
  hipMalloc(&x_temp_arr, size_5);
  hipMalloc(&invA_arr, size_6);
  if (!scalars || (size_2 && !work) || (size_3 && !workArr) ||
      (size_4 && !x_temp) || (size_5 && !x_temp_arr) || (size_6 && !invA_arr))
    return rocblas_status_memory_error;

  // scalar constants for rocblas functions calls
  // (to standarize and enable re-use, size_1 always equals 3*sizeof(T))
  T sca[] = {-1, 0, 1};
  RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca, size_1, hipMemcpyHostToDevice));

  // execution
  rocblas_status status = rocsolver_getri_template<true, false, T>(
      handle, n, A, 0, lda, strideA, C,
      0, // the matrix is shifted 0 entries (will work on the entire matrix)
      ldc, strideC, ipiv,
      0, // the vector is shifted 0 entries (will work on the entire vector)
      strideP, info, batch_count, (T *)scalars, (T *)work, (T **)workArr,
      x_temp, x_temp_arr, invA_arr, optim_mem);

  hipFree(scalars);
  hipFree(work);
  hipFree(workArr);
  hipFree(x_temp);
  hipFree(x_temp_arr);
  hipFree(invA_arr);
  return status;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetri_outofplace_batched(
    rocblas_handle handle, const rocblas_int n, float *const A[],
    const rocblas_int lda, rocblas_int *ipiv, const rocblas_stride strideP,
    float *const C[], const rocblas_int ldc, rocblas_int *info,
    const rocblas_int batch_count) {
  return rocsolver_getri_outofplace_batched_impl<float>(
      handle, n, A, lda, ipiv, strideP, C, ldc, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetri_outofplace_batched(
    rocblas_handle handle, const rocblas_int n, double *const A[],
    const rocblas_int lda, rocblas_int *ipiv, const rocblas_stride strideP,
    double *const C[], const rocblas_int ldc, rocblas_int *info,
    const rocblas_int batch_count) {
  return rocsolver_getri_outofplace_batched_impl<double>(
      handle, n, A, lda, ipiv, strideP, C, ldc, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetri_outofplace_batched(
    rocblas_handle handle, const rocblas_int n,
    rocblas_float_complex *const A[], const rocblas_int lda, rocblas_int *ipiv,
    const rocblas_stride strideP, rocblas_float_complex *const C[],
    const rocblas_int ldc, rocblas_int *info, const rocblas_int batch_count) {
  return rocsolver_getri_outofplace_batched_impl<rocblas_float_complex>(
      handle, n, A, lda, ipiv, strideP, C, ldc, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetri_outofplace_batched(
    rocblas_handle handle, const rocblas_int n,
    rocblas_double_complex *const A[], const rocblas_int lda, rocblas_int *ipiv,
    const rocblas_stride strideP, rocblas_double_complex *const C[],
    const rocblas_int ldc, rocblas_int *info, const rocblas_int batch_count) {
  return rocsolver_getri_outofplace_batched_impl<rocblas_double_complex>(
      handle, n, A, lda, ipiv, strideP, C, ldc, info, batch_count);
}

} // extern C
