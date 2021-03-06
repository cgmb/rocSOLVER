/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#ifndef ROCLAPACK_ORMQR_UNMQR_HPP
#define ROCLAPACK_ORMQR_UNMQR_HPP

#include "rocauxiliary_larfb.hpp"
#include "rocauxiliary_larft.hpp"
#include "rocauxiliary_orm2r_unm2r.hpp"
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, bool BATCHED>
void rocsolver_ormqr_unmqr_getMemorySize(
    const rocblas_side side, const rocblas_int m, const rocblas_int n,
    const rocblas_int k, const rocblas_int batch_count, size_t *size_1,
    size_t *size_2, size_t *size_3, size_t *size_4, size_t *size_5) {
  size_t s1, s2, unused;
  rocsolver_orm2r_unm2r_getMemorySize<T, BATCHED>(
      side, m, n, batch_count, size_1, size_2, size_3, size_4);

  if (k > ORMQR_ORM2R_BLOCKSIZE) {
    // size of workspace
    // maximum of what is needed by larft and larfb
    rocblas_int jb = ORMQR_ORM2R_BLOCKSIZE;
    rocsolver_larft_getMemorySize<T>(min(jb, k), batch_count, &s1);
    rocsolver_larfb_getMemorySize<T, BATCHED>(
        side, m, n, min(jb, k), batch_count, &s2, &unused, size_5);

    *size_2 = max(s1, s2);

    // size of temporary array for triangular factor
    *size_4 = sizeof(T) * jb * jb * batch_count;
  } else
    *size_5 = 0;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_ormqr_unmqr_template(
    rocblas_handle handle, const rocblas_side side,
    const rocblas_operation trans, const rocblas_int m, const rocblas_int n,
    const rocblas_int k, U A, const rocblas_int shiftA, const rocblas_int lda,
    const rocblas_stride strideA, T *ipiv, const rocblas_stride strideP, U C,
    const rocblas_int shiftC, const rocblas_int ldc,
    const rocblas_stride strideC, const rocblas_int batch_count, T *scalars,
    T *work, T **workArr, T *trfact, T *workTrmm) {
  // quick return
  if (!n || !m || !k || !batch_count)
    return rocblas_status_success;

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  // if the matrix is small, use the unblocked variant of the algorithm
  if (k <= ORMQR_ORM2R_BLOCKSIZE)
    return rocsolver_orm2r_unm2r_template<T>(
        handle, side, trans, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, C,
        shiftC, ldc, strideC, batch_count, scalars, work, workArr, trfact);

  rocblas_int ldw = ORMQR_ORM2R_BLOCKSIZE;
  rocblas_stride strideW = rocblas_stride(ldw) * ldw;

  // determine limits and indices
  bool left = (side == rocblas_side_left);
  bool transpose = (trans != rocblas_operation_none);
  rocblas_int start, step, ncol, nrow, ic, jc, order;
  if (left) {
    ncol = n;
    order = m;
    jc = 0;
    if (transpose) {
      start = 0;
      step = 1;
    } else {
      start = (k - 1) / ldw * ldw;
      step = -1;
    }
  } else {
    nrow = m;
    order = n;
    ic = 0;
    if (transpose) {
      start = (k - 1) / ldw * ldw;
      step = -1;
    } else {
      start = 0;
      step = 1;
    }
  }

  rocblas_int i;
  for (rocblas_int j = 0; j < k; j += ldw) {
    i = start + step * j; // current householder block
    if (left) {
      nrow = m - i;
      ic = i;
    } else {
      ncol = n - i;
      jc = i;
    }

    // generate triangular factor of current block reflector
    rocsolver_larft_template<T>(
        handle, rocblas_forward_direction, rocblas_column_wise, order - i,
        min(ldw, k - i), A, shiftA + idx2D(i, i, lda), lda, strideA, ipiv + i,
        strideP, trfact, ldw, strideW, batch_count, scalars, work, workArr);

    // apply current block reflector
    rocsolver_larfb_template<BATCHED, STRIDED, T>(
        handle, side, trans, rocblas_forward_direction, rocblas_column_wise,
        nrow, ncol, min(ldw, k - i), A, shiftA + idx2D(i, i, lda), lda, strideA,
        trfact, 0, ldw, strideW, C, shiftC + idx2D(ic, jc, ldc), ldc, strideC,
        batch_count, work, workArr, workTrmm);
  }

  return rocblas_status_success;
}

#endif
