/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gels.hpp"
#include "roclapack_geqrf.hpp"
#include "rocblas.hpp"
#include "../auxiliary/rocauxiliary_ormqr_unmqr.hpp"

//          = 'S' or 's ,   SLAMCH('S') := sfmin
//          = 'B' or 'b',   SLAMCH := base
//          = 'P' or 'p',   SLAMCH('P') := eps*base

template <typename T>
rocblas_status rocsolver_gels_batched_impl(
    rocblas_handle handle, rocblas_operation trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs, T *const *A,
    const rocblas_int lda, T *const *C, const rocblas_int ldc,
    rocblas_int *info, rocblas_int* solution_info,
    const rocblas_int batch_count) {
  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st =
      rocsolver_gels_argCheck(trans, m, n, nrhs, A, lda, C, ldc, batch_count);
  if (st != rocblas_status_continue)
    return st;

  constexpr bool BATCHED = true;

  size_t size_scalars, size_2, size_3, size_4, size_5;
  rocsolver_gels_getMemorySize<BATCHED,T>(m, n, nrhs, batch_count,
    &size_scalars, &size_2, &size_3, &size_4, &size_5);

  if (rocblas_is_device_memory_size_query(handle))
    return rocblas_set_optimal_device_memory_size(handle, size_scalars,
                                                  size_2, size_3,
                                                  size_4, size_5);
  // memory workspace allocation
  rocblas_device_malloc mem(handle, size_scalars, size_2, size_3, size_4, size_5);
  if (!mem)
    return rocblas_status_memory_error;

  T *scalars = (T *)mem[0];
  T sca[] = {-1, 0, 1};
  RETURN_IF_HIP_ERROR(
      hipMemcpy(scalars, sca, size_scalars, hipMemcpyHostToDevice));

  return rocsolver_gels_template<BATCHED>(handle, trans, m,
    n, nrhs, A, lda, C, ldc,
    info, solution_info,
    batch_count, scalars, mem[1], mem[2], mem[3], mem[4]);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgels_batched(rocblas_handle handle,
                                       rocblas_operation trans,
                                       const rocblas_int m, const rocblas_int n,
                                       const rocblas_int nrhs, float *const A[],
                                       const rocblas_int lda, float *const C[],
                                       const rocblas_int ldc, rocblas_int *info,
                                       rocblas_int* solution_info,
                                       const rocblas_int batch_count) {
  return rocsolver_gels_batched_impl(handle, trans, m, n, nrhs, A, lda,
                                            C, ldc, info, solution_info,
                                            batch_count);
}

rocblas_status rocsolver_dgels_batched(
    rocblas_handle handle, rocblas_operation trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs, double *const A[],
    const rocblas_int lda, double *const C[], const rocblas_int ldc,
    rocblas_int *info, rocblas_int* solution_info,
    const rocblas_int batch_count) {
  return rocsolver_gels_batched_impl(handle, trans, m, n, nrhs, A, lda,
                                             C, ldc, info, solution_info,
                                             batch_count);
}

rocblas_status rocsolver_cgels_batched(
    rocblas_handle handle, rocblas_operation trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs,
    rocblas_float_complex *const A[], const rocblas_int lda,
    rocblas_float_complex *const C[], const rocblas_int ldc, rocblas_int *info,
    rocblas_int* solution_info, const rocblas_int batch_count) {
  return rocsolver_gels_batched_impl(
      handle, trans, m, n, nrhs, A, lda, C, ldc, info, solution_info,
      batch_count);
}

rocblas_status rocsolver_zgels_batched(
    rocblas_handle handle, rocblas_operation trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs,
    rocblas_double_complex *const A[], const rocblas_int lda,
    rocblas_double_complex *const C[], const rocblas_int ldc, rocblas_int *info,
    rocblas_int* solution_info, const rocblas_int batch_count) {
  return rocsolver_gels_batched_impl(
      handle, trans, m, n, nrhs, A, lda, C, ldc, info, solution_info,
      batch_count);
}

} // extern C
