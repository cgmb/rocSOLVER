/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_GELS_HPP
#define ROCLAPACK_GELS_HPP

#include "rocblas.hpp"
#include "rocsolver.h"
#include "roclapack_geqrf.hpp"
#include "../auxiliary/rocauxiliary_ormqr_unmqr.hpp"

template <typename T>
rocblas_status
rocsolver_gels_argCheck(rocblas_operation trans, const rocblas_int m,
                        const rocblas_int n, const rocblas_int nrhs, T A,
                        const rocblas_int lda, T C, const rocblas_int ldc,
                        const rocblas_int batch_count) {
  // order is important for unit tests:

  // 1. invalid/non-supported values
  if (trans != rocblas_operation_none)
    return rocblas_status_invalid_value;

  // 2. invalid size
  if (n < 0 || nrhs < 0 || lda < m || ldc < n || batch_count < 0)
    return rocblas_status_invalid_size;

  // 3. invalid pointers
  if ((n && !A) || (n && !C))
    return rocblas_status_invalid_pointer;

  return rocblas_status_continue;
}

template <bool BATCHED, typename T>
void rocsolver_gels_getMemorySize(const rocblas_int m, const rocblas_int n, const rocblas_int nrhs,
                                   const rocblas_int batch_count,
                                   size_t *size_scalars, size_t *size_2,
                                   size_t *size_3, size_t *size_4, size_t *size_5) {
  // if quick return no workspace needed
  if (m == 0 || n == 0 || batch_count == 0) {
    *size_scalars = 0;
    *size_2 = 0;
    *size_3 = 0;
    *size_4 = 0;
    *size_5 = 0;
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

  // TODO: adjust to minimize total size
  *size_scalars = std::max(geqrf_scalars, ormqr_scalars);
  *size_2 = std::max({geqrf_work, ormqr_work, trsm_x_temp});
  *size_3 = std::max({geqrf_workArr, ormqr_workArr, trsm_x_temp_arr});
  *size_4 = std::max({geqrf_diag, ormqr_trfact, trsm_invA});
  *size_5 = std::max({geqrf_trfact, ormqr_workTrmm, trsm_invA_arr});
}

template <bool BATCHED, typename T>
rocblas_status rocsolver_gels_template(
    rocblas_handle handle, rocblas_operation trans, const rocblas_int m,
    const rocblas_int n, const rocblas_int nrhs, T *const *A,
    const rocblas_int lda, T *const *C, const rocblas_int ldc,
    rocblas_int *info, rocblas_int* solution_info,
    const rocblas_int batch_count, T* scalars, void* void_work, void* void_workArr, void* void_diag_trfact, void* void_trfact_workTrmm) {
  static_assert(BATCHED, "only the batched version is implemented");
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

  // TODO: fixup this allocation
  const rocblas_int pivot_count_per_batch = std::min(m, n);
  T *dIpiv;
  hipMalloc(&dIpiv, sizeof(T) * pivot_count_per_batch * batch_count);

  constexpr rocblas_int shiftA = 0;
  constexpr rocblas_int shiftC = 0;
  constexpr rocblas_stride strideA = 0;
  constexpr rocblas_stride strideC = 0;
  constexpr rocblas_stride strideP = 0;
  constexpr bool STRIDED = false;
  {
    // note: m > n
    // compute QR factorization of A
    T *work = (T *)void_work;
    T *workArr = (T *)void_workArr;
    T *diag = (T *)void_diag_trfact;
    T **trfact = (T **)void_trfact_workTrmm;
    rocsolver_geqrf_template<BATCHED, STRIDED>(
        handle, m, n, A, shiftA, lda, strideA, dIpiv, strideP, batch_count,
        scalars, work, workArr, diag, trfact);
    // what if the result is not success?
  }
  {
      T *work = (T *)void_work;
      T *workArr = (T *)void_workArr;
      T *trfact = (T *)void_diag_trfact;
      T **workTrmm = (T **)void_trfact_workTrmm;
      rocsolver_ormqr_unmqr_template<BATCHED,STRIDED>(
          handle, rocblas_side_left, rocblas_operation_transpose, m, nrhs, n, A,
          shiftA, lda, strideA, dIpiv, strideP, C, shiftC, ldc, strideC,
          batch_count, scalars, work, workArr, trfact, workTrmm);
      // again, what about failure?
  }
  {
      // this is implementing strtrs inline
      // TODO: singularity check
      void *x_temp = void_work;
      void *x_temp_arr = void_workArr;
      void *invA =  void_diag_trfact;
      void *invA_arr = void_trfact_workTrmm;
      constexpr bool optimal_memory = true;
      const T one = 1; // constant 1 in host
      // solve U*X = B, overwriting B with X
      rocblasCall_trsm<BATCHED>(handle, rocblas_side_left, rocblas_fill_upper,
                                rocblas_operation_none, rocblas_diagonal_non_unit, n, nrhs,
                                &one, A, shiftA, lda, strideA, C, shiftC, ldc,
                                strideC, batch_count, optimal_memory, x_temp,
                                x_temp_arr, invA, invA_arr/*, workArr ? */);
      // failure?
  }

  rocblas_set_pointer_mode(handle, old_mode);
  return rocblas_status_success;
}

#endif /* ROCLAPACK_GELS_HPP */
