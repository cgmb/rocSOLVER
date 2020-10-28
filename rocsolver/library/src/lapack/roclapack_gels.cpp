/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gels.hpp"

template <typename T>
rocblas_status rocsolver_gels_impl(
    rocblas_handle handle, rocblas_operation trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs, T * A,
    const rocblas_int lda, T * C, const rocblas_int ldc,
    rocblas_int *info) {
  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st = rocsolver_gels_argCheck(trans, m, n, nrhs, A, lda, C, ldc);
  if (st != rocblas_status_continue)
    return st;

  constexpr rocblas_int batch_count = 1;
  constexpr rocblas_int shiftA = 0;
  constexpr rocblas_stride strideA = 0;
  constexpr rocblas_int shiftC = 0;
  constexpr rocblas_stride strideC = 0;

  size_t size_scalars, size_2, size_3, size_4, size_5;
  rocsolver_gels_getMemorySize<false,false,T>(m, n, nrhs, batch_count,
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

  return rocsolver_gels_template<false,false>(handle, trans, m,
    n, nrhs, A, shiftA, lda, strideA, C, shiftC, ldc, strideC,
    info, batch_count, scalars, mem[1], mem[2], mem[3], mem[4]);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgels(rocblas_handle handle,
                                       rocblas_operation trans,
                                       const rocblas_int m, const rocblas_int n,
                                       const rocblas_int nrhs, float * A,
                                       const rocblas_int lda, float * C,
                                       const rocblas_int ldc, rocblas_int *info) {
  return rocsolver_gels_impl(handle, trans, m, n, nrhs, A, lda, C, ldc, info);
}

rocblas_status rocsolver_dgels(
    rocblas_handle handle, rocblas_operation trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs, double * A,
    const rocblas_int lda, double * C, const rocblas_int ldc,
    rocblas_int *info) {
  return rocsolver_gels_impl(handle, trans, m, n, nrhs, A, lda, C, ldc, info);
}

rocblas_status rocsolver_cgels(
    rocblas_handle handle, rocblas_operation trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs,
    rocblas_float_complex * A, const rocblas_int lda,
    rocblas_float_complex * C, const rocblas_int ldc, rocblas_int *info) {
  return rocsolver_gels_impl(handle, trans, m, n, nrhs, A, lda, C, ldc, info);
}

rocblas_status rocsolver_zgels(
    rocblas_handle handle, rocblas_operation trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs,
    rocblas_double_complex * A, const rocblas_int lda,
    rocblas_double_complex * C, const rocblas_int ldc, rocblas_int *info) {
  return rocsolver_gels_impl(handle, trans, m, n, nrhs, A, lda, C, ldc, info);
}

} // extern C
