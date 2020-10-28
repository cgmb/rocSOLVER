/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gels.hpp"

template <typename T, typename U>
rocblas_status rocsolver_gels_strided_batched_impl(rocblas_handle handle,
                                                   const rocblas_int m,
                                                   const rocblas_int n,
                                                   U A,
                                                   const rocblas_int lda,
                                                   const rocblas_stride strideA,
                                                   rocblas_int* ipiv,
                                                   const rocblas_stride strideP,
                                                   rocblas_int* info,
                                                   const rocblas_int batch_count,
                                                   const int pivot)
{
    using S = decltype(std::real(T{}));

    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_getf2_gels_argCheck(m, n, lda, A, ipiv, info, pivot, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
  const rocblas_int shiftA = 0;
  const rocblas_int shiftC = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
//    size_t size_scalars;
    // size of reusable workspace (and for calling TRSM)
//    size_t size_work, size_work1, size_work2, size_work3, size_work4;
    // extra requirements for calling GETF2
//    size_t size_pivotval, size_pivotidx;
    // size to store info about singularity of each subblock
//    size_t size_iinfo;

    size_t size_scalars, size_2, size_3, size_4, size_5;
    rocsolver_gels_getMemorySize<false,true,T>(m, n, nrhs, batch_count,
      &size_scalars, &size_2, &size_3, &size_4, &size_5);

  if (rocblas_is_device_memory_size_query(handle))
    return rocblas_set_optimal_device_memory_size(handle, size_scalars,
                                                  size_2, size_3,
                                                  size_4, size_5);
  // memory workspace allocation
  rocblas_device_malloc mem(handle, size_scalars, size_2, size_3, size_4, size_5);
  if (!mem)
    return rocblas_status_memory_error;

    // always allocate all required memory for TRSM optimal performance
    const bool optimial_memory = true;

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4;
    rocblas_device_malloc mem(handle, size_scalars, size_2, size_3, size_4, size_5);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    T sca[] = {-1, 0, 1};
    RETURN_IF_HIP_ERROR(hipMemcpy((T*)scalars, sca, size_scalars, hipMemcpyHostToDevice));

  return rocsolver_gels_template<false, true, T>(handle, trans, m,
    n, nrhs, A, shiftA, lda, strideA, C, shiftC, ldc, strideC,
    info, batch_count, scalars, work1, work2, work3, work4);
/*
    return rocsolver_gels_template<false, true, T>(
        handle, m, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info, batch_count, pivot,
        (T*)scalars, work1, work2, work3, work4);
*/
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgels_strided_batched(rocblas_handle handle,
                                       rocblas_operation trans,
                                       const rocblas_int m, const rocblas_int n,
                                       const rocblas_int nrhs, float* A,
                                       const rocblas_int lda, const rocblas_stride strideA, float* C,
                                       const rocblas_int ldc, const rocblas_stride strideC, rocblas_int* info,
                                       const rocblas_int batch_count)
{
    return rocsolver_gels_strided_batched_impl<float>(handle, trans, m, n, nrhs, A, lda, strideA, C, ldc, strideC, info, batch_count);
}

rocblas_status rocsolver_dgels_strided_batched(rocblas_handle handle,
                                       rocblas_operation trans,
                                       const rocblas_int m, const rocblas_int n,
                                       const rocblas_int nrhs, double* A,
                                       const rocblas_int lda, const rocblas_stride strideA, double* C,
                                       const rocblas_int ldc, const rocblas_stride strideC, rocblas_int* info,
                                       const rocblas_int batch_count)
{
    return rocsolver_gels_strided_batched_impl<double>(handle, trans, m, n, nrhs, A, lda, strideA, C, ldc, strideC, info, batch_count);
}

rocblas_status rocsolver_cgels_strided_batched(rocblas_handle handle,
                                       rocblas_operation trans,
                                       const rocblas_int m, const rocblas_int n,
                                       const rocblas_int nrhs, rocblas_float_complex* A,
                                       const rocblas_int lda, const rocblas_stride strideA, rocblas_float_complex* C,
                                       const rocblas_int ldc, const rocblas_stride strideC, rocblas_int* info,
                                       const rocblas_int batch_count)
{
    return rocsolver_gels_strided_batched_impl<rocblas_float_complex>(handle, trans, m, n, nrhs, A, lda, strideA, C, ldc, strideC, info, batch_count);
}

rocblas_status rocsolver_zgels_strided_batched(rocblas_handle handle,
                                       rocblas_operation trans,
                                       const rocblas_int m, const rocblas_int n,
                                       const rocblas_int nrhs, rocblas_double_complex* A,
                                       const rocblas_int lda, const rocblas_stride strideA, rocblas_double_complex* C,
                                       const rocblas_int ldc, const rocblas_stride strideC, rocblas_int* info,
                                       const rocblas_int batch_count)
{
    return rocsolver_gels_strided_batched_impl<rocblas_double_complex>(handle, trans, m, n, nrhs, A, lda, strideA, C, ldc, strideC, info, batch_count);
}

} // extern C
