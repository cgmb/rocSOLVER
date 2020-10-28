/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GELS_HPP
#define ROCLAPACK_GELS_HPP

#include "rocblas.hpp"
#include "rocsolver.h"
#include "roclapack_geqrf.hpp"
#include "../auxiliary/rocauxiliary_ormqr_unmqr.hpp"

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_gels_getMemorySize(const rocblas_int m, const rocblas_int n, const rocblas_int nrhs,
                                   const rocblas_int batch_count,
                                   size_t *size_scalars, size_t *size_work_x_temp,
                                   size_t *size_workArr_temp_arr, size_t *size_diag_trfac_invA, size_t *size_trfact_workTrmm_invA_arr, size_t* size_ipiv) {
  // if quick return no workspace needed
  if (m == 0 || n == 0 || batch_count == 0) {
    *size_scalars = 0;
    *size_work_x_temp = 0;
    *size_workArr_temp_arr = 0;
    *size_diag_trfac_invA = 0;
    *size_trfact_workTrmm_invA_arr = 0;
    *size_ipiv = 0;
    return;
  }

  size_t geqrf_scalars, geqrf_work, geqrf_workArr, geqrf_diag, geqrf_trfact;
  rocsolver_geqrf_getMemorySize<T,BATCHED>(m, n, batch_count,
    &geqrf_scalars, &geqrf_work, &geqrf_workArr, &geqrf_diag, &geqrf_trfact);

  size_t ormqr_scalars, ormqr_work, ormqr_workArr, ormqr_trfact, ormqr_workTrmm;
  rocsolver_ormqr_unmqr_getMemorySize<T, BATCHED>(
      rocblas_side_left, m, nrhs, n, batch_count, &ormqr_scalars, &ormqr_work,
      &ormqr_workArr, &ormqr_trfact, &ormqr_workTrmm);

  size_t trsm_x_temp, trsm_x_temp_arr, trsm_invA, trsm_invA_arr;
  rocblasCall_trsm_mem<BATCHED,T>(rocblas_side_left, n, nrhs, batch_count,
                       &trsm_x_temp, &trsm_x_temp_arr, &trsm_invA, &trsm_invA_arr);

  // TODO: rearrange to minimize total size
  *size_scalars = std::max(geqrf_scalars, ormqr_scalars);
  *size_work_x_temp = std::max({geqrf_work, ormqr_work, trsm_x_temp});
  *size_workArr_temp_arr = std::max({geqrf_workArr, ormqr_workArr, trsm_x_temp_arr});
  *size_diag_trfac_invA = std::max({geqrf_diag, ormqr_trfact, trsm_invA});
  *size_trfact_workTrmm_invA_arr = std::max({geqrf_trfact, ormqr_workTrmm, trsm_invA_arr});

  const rocblas_int pivot_count_per_batch = std::min(m, n);
  *size_ipiv = sizeof(T) * pivot_count_per_batch * batch_count;
}

template <typename T>
rocblas_status
rocsolver_gels_argCheck(rocblas_operation trans, const rocblas_int m,
                        const rocblas_int n, const rocblas_int nrhs, T A,
                        const rocblas_int lda, T C, const rocblas_int ldc,
                        const rocblas_int batch_count = 1) {
  // order is important for unit tests:
  // 1. invalid/non-supported values
  if (trans != rocblas_operation_none)
    return rocblas_status_invalid_value;

/*
  // 2. invalid size
  if (n < 0 || nrhs < 0 || lda < m || ldc < n || batch_count < 0)
    return rocblas_status_invalid_size;

  // 3. invalid pointers
  if ((n && !A) || (n && !C))
    return rocblas_status_invalid_pointer;
*/
  return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_gels_template(
    rocblas_handle handle, rocblas_operation trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs, U A, const rocblas_int shiftA,
    const rocblas_int lda, const rocblas_stride strideA, U C, const rocblas_int shiftC, const rocblas_int ldc, const rocblas_stride strideC,
    T* ipiv, const rocblas_stride strideP,
    rocblas_int *info, const rocblas_int batch_count, T* scalars, void* void_work, void* void_workArr, void* void_diag_trfact, void* void_trfact_workTrmm, bool optim_mem) {

  // quick return
  if (n == 0 || nrhs == 0 || batch_count == 0) {
    return rocblas_status_success;
  }

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  // everything must be executed with scalars on the host
  rocblas_pointer_mode old_mode;
  rocblas_get_pointer_mode(handle, &old_mode);
  rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

  {
    // note: m > n
    // compute QR factorization of A
    T *work = (T *)void_work;
    T *workArr = (T *)void_workArr;
    T *diag = (T *)void_diag_trfact;
    T **trfact = (T **)void_trfact_workTrmm;
    rocsolver_geqrf_template<BATCHED, STRIDED>(
        handle, m, n, A, shiftA, lda, strideA, ipiv, strideP, batch_count,
        scalars, work, workArr, diag, trfact);
  }
  {
      T *work = (T *)void_work;
      T *workArr = (T *)void_workArr;
      T *trfact = (T *)void_diag_trfact;
      T **workTrmm = (T **)void_trfact_workTrmm;
      rocsolver_ormqr_unmqr_template<BATCHED,STRIDED>(
          handle, rocblas_side_left, rocblas_operation_transpose, m, nrhs, n, A,
          shiftA, lda, strideA, ipiv, strideP, C, shiftC, ldc, strideC,
          batch_count, scalars, work, workArr, trfact, workTrmm);
  }
  {
      // do the equivalent of strtrs

      // TODO: singularity check
      void *x_temp = void_work;
      void *x_temp_arr = void_workArr;
      void *invA =  void_diag_trfact;
      void *invA_arr = void_trfact_workTrmm;
      const T one = 1; // constant 1 in host
      // solve U*X = B, overwriting B with X
      rocblasCall_trsm<BATCHED, T>(handle, rocblas_side_left, rocblas_fill_upper,
                                rocblas_operation_none, rocblas_diagonal_non_unit, n, nrhs,
                                &one, A, shiftA, lda, strideA, C, shiftC, ldc,
                                strideC, batch_count, optim_mem, x_temp,
                                x_temp_arr, invA, invA_arr/*, workArr ? */);
  }

  rocblas_set_pointer_mode(handle, old_mode);
  return rocblas_status_success;
}

#endif /* ROCLAPACK_GELS_HPP */
