/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gels.hpp"

// was geqrf

template <typename T, typename U>
rocblas_status
rocsolver_gels_batched_impl(rocblas_handle handle, const rocblas_int m,
                            const rocblas_int n, U A, const rocblas_int lda,
                            T *ipiv, const rocblas_stride stridep,
                            const rocblas_int batch_count) {
  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st =
      rocsolver_geqr2_gels_argCheck(m, n, lda, A, ipiv, batch_count);
  if (st != rocblas_status_continue)
    return st;

  rocblas_stride strideA = 0;

  // memory managment
  size_t size_1; // size of constants
  size_t size_2; // size of workspace
  size_t size_3; // size of array of pointers to workspace
  size_t size_4; // size of diagonal entry cache and TRMM calls workspace
  size_t size_5; // size of triangular factor for block reflector
  rocsolver_gels_getMemorySize<T, true>(m, n, batch_count, &size_1, &size_2,
                                        &size_3, &size_4, &size_5);

  // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
  void *scalars, *work, *workArr, *diag, *trfact;
  hipMalloc(&scalars, size_1);
  hipMalloc(&work, size_2);
  hipMalloc(&workArr, size_3);
  hipMalloc(&diag, size_4);
  hipMalloc(&trfact, size_5);
  if (!scalars || (size_2 && !work) || (size_3 && !workArr) ||
      (size_4 && !diag) || (size_5 && !trfact))
    return rocblas_status_memory_error;

  // scalar constants for rocblas functions calls
  // (to standarize and enable re-use, size_1 always equals 3*sizeof(T))
  T sca[] = {-1, 0, 1};
  RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca, size_1, hipMemcpyHostToDevice));

  // execution
  rocblas_status status = rocsolver_gels_template<true, false, T>(
      handle, m, n, A,
      0, // the matrix is shifted 0 entries (will work on the entire matrix)
      lda, strideA, ipiv, stridep, batch_count, (T *)scalars, (T *)work,
      (T **)workArr, (T *)diag, (T *)trfact);

  hipFree(scalars);
  hipFree(work);
  hipFree(workArr);
  hipFree(diag);
  hipFree(trfact);
  return status;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgels_batched(rocblas_handle handle,
                                       rocblas_operation_t trans,
                                       const rocblas_int m, const rocblas_int n,
                                       const rocblas_int nrhs, float *const A[],
                                       const rocblas_int lda, float *const C[],
                                       const rocblas_int ldc, rocblas_int *info,
                                       rocblas_int solution_info[],
                                       const rocblas_int batch_count) {
  return rocsolver_gels_batched_impl<float>(handle, trans, m, n, nrhs, A, lda,
                                            C, ldc, info, solution_info,
                                            batch_count);
}

rocblas_status rocsolver_dgels_batched(
    rocblas_handle handle, rocblas_operation_t trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs, double *const A[],
    const rocblas_int lda, double *const C[], const rocblas_int ldc,
    rocblas_int *info, rocblas_int solution_info[],
    const rocblas_int batch_count) {
  return rocsolver_gels_batched_impl<double>(handle, trans, m, n, nrhs, A, lda,
                                             C, ldc, info, solution_info,
                                             batch_count);
}

rocblas_status rocsolver_cgels_batched(
    rocblas_handle handle, rocblas_operation_t trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs,
    rocblas_float_complex *const A[], const rocblas_int lda,
    rocblas_float_complex *const C[], const rocblas_int ldc, rocblas_int *info,
    rocblas_int solution_info[], const rocblas_int batch_count) {
  return rocsolver_gels_batched_impl<rocblas_float_complex>(
      handle, trans, m, n, nrhs, A, lda, C, ldc, info, solution_info,
      batch_count);
}

rocblas_status rocsolver_zgels_batched(
    rocblas_handle handle, rocblas_operation_t trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs,
    rocblas_double_complex *const A[], const rocblas_int lda,
    rocblas_double_complex *const C[], const rocblas_int ldc, rocblas_int *info,
    rocblas_int solution_info[],
    const rocblas_int
        batch_count) return rocsolver_gels_batched_impl<rocblas_float_complex>(handle,
                                                                               trans,
                                                                               m,
                                                                               n,
                                                                               nrhs,
                                                                               A,
                                                                               lda,
                                                                               C,
                                                                               ldc,
                                                                               info,
                                                                               solution_info,
                                                                               batch_count);
}

rocblas_status rocsolver_sgels_batched(rocblas_handle handle,
                                       const rocblas_int m, const rocblas_int n,
                                       float *const A[], const rocblas_int lda,
                                       float *ipiv,
                                       const rocblas_stride stridep,
                                       const rocblas_int batch_count) {
  return rocsolver_gels_batched_impl<float>(handle, m, n, A, lda, ipiv, stridep,
                                            batch_count);
}

rocblas_status rocsolver_dgels_batched(rocblas_handle handle,
                                       const rocblas_int m, const rocblas_int n,
                                       double *const A[], const rocblas_int lda,
                                       double *ipiv,
                                       const rocblas_stride stridep,
                                       const rocblas_int batch_count) {
  return rocsolver_gels_batched_impl<double>(handle, m, n, A, lda, ipiv,
                                             stridep, batch_count);
}

rocblas_status rocsolver_cgels_batched(rocblas_handle handle,
                                       const rocblas_int m, const rocblas_int n,
                                       rocblas_float_complex *const A[],
                                       const rocblas_int lda,
                                       rocblas_float_complex *ipiv,
                                       const rocblas_stride stridep,
                                       const rocblas_int batch_count) {
  return rocsolver_gels_batched_impl<rocblas_float_complex>(
      handle, m, n, A, lda, ipiv, stridep, batch_count);
}

rocblas_status rocsolver_zgels_batched(rocblas_handle handle,
                                       const rocblas_int m, const rocblas_int n,
                                       rocblas_double_complex *const A[],
                                       const rocblas_int lda,
                                       rocblas_double_complex *ipiv,
                                       const rocblas_stride stridep,
                                       const rocblas_int batch_count) {
  return rocsolver_gels_batched_impl<rocblas_double_complex>(
      handle, m, n, A, lda, ipiv, stridep, batch_count);
}

} // extern C
