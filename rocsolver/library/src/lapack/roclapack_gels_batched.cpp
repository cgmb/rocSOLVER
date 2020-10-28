/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gels.hpp"

template <typename T>
rocblas_status rocsolver_gels_batched_impl(
    rocblas_handle handle, rocblas_operation trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs, T *const *A,
    const rocblas_int lda, T *const *C, const rocblas_int ldc,
    rocblas_int *info, const rocblas_int batch_count) {
  if (!handle)
    return rocblas_status_invalid_handle;

  // logging is missing ???

  // argument checking
  rocblas_status st =
      rocsolver_gels_argCheck(trans, m, n, nrhs, A, lda, C, ldc, batch_count);
  if (st != rocblas_status_continue)
    return st;

  // working with unshifted arrays
  const rocblas_int shiftA = 0;
  const rocblas_stride strideA = 0;
  const rocblas_int shiftC = 0;
  const rocblas_stride strideC = 0;

  size_t size_scalars, size_2, size_3, size_4, size_5;
  rocsolver_gels_getMemorySize<true, false, T>(m, n, nrhs, batch_count,
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

  return rocsolver_gels_template<true, false, T>(handle, trans, m,
    n, nrhs, A, shiftA, lda, strideA, C, shiftC, ldc, strideC,
    info, batch_count, scalars, mem[1], mem[2], mem[3], mem[4]);
}
/*
    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // batched execution
    rocblas_stride strideA = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr;
    // extra requirements for calling larf and larfg
    size_t size_Abyx_norms;
    rocsolver_gebd2_getMemorySize<T, true>(m, n, batch_count, &size_scalars, &size_work_workArr,
                                           &size_Abyx_norms);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_workArr,
                                                      size_Abyx_norms);

    // memory workspace allocation
    void *scalars, *work_workArr, *Abyx_norms;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_Abyx_norms);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms = mem[2];
    T sca[] = {-1, 0, 1};
    RETURN_IF_HIP_ERROR(hipMemcpy((T*)scalars, sca, size_scalars, hipMemcpyHostToDevice));
*/
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
                                       const rocblas_int batch_count) {
  return rocsolver_gels_batched_impl<float>(handle, trans, m, n, nrhs, A, lda,
                                            C, ldc, info, batch_count);
}

rocblas_status rocsolver_dgels_batched(
    rocblas_handle handle, rocblas_operation trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs, double *const A[],
    const rocblas_int lda, double *const C[], const rocblas_int ldc,
    rocblas_int *info, const rocblas_int batch_count) {
  return rocsolver_gels_batched_impl<double>(handle, trans, m, n, nrhs, A, lda,
                                             C, ldc, info, batch_count);
}

rocblas_status rocsolver_cgels_batched(
    rocblas_handle handle, rocblas_operation trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs,
    rocblas_float_complex *const A[], const rocblas_int lda,
    rocblas_float_complex *const C[], const rocblas_int ldc, rocblas_int *info,
    const rocblas_int batch_count) {
  return rocsolver_gels_batched_impl<rocblas_float_complex>(
      handle, trans, m, n, nrhs, A, lda, C, ldc, info, batch_count);
}

rocblas_status rocsolver_zgels_batched(
    rocblas_handle handle, rocblas_operation trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs,
    rocblas_double_complex *const A[], const rocblas_int lda,
    rocblas_double_complex *const C[], const rocblas_int ldc, rocblas_int *info,
    const rocblas_int batch_count) {
  return rocsolver_gels_batched_impl<rocblas_double_complex>(
      handle, trans, m, n, nrhs, A, lda, C, ldc, info, batch_count);
}

} // extern C
