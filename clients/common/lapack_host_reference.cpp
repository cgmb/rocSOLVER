/* ************************************************************************
 * Copyright (c) 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************/

#include "lapack_host_reference.hpp"

#include <cblas.h>
#include <rocblas.h>

#include "FortranCInterface.h"

/*!\file
 * \brief provide template functions interfaces to BLAS and LAPACK interfaces, it is
 * only used for testing, not part of the GPU library
 */

/*************************************************************************/
// These are C wrapper calls to CBLAS and fortran LAPACK

#ifdef __cplusplus
extern "C" {
#endif

void ssymv(char* uplo,
           int* n,
           float* alpha,
           float* A,
           int* lda,
           float* x,
           int* incx,
           float* beta,
           float* y,
           int* incy);
void dsymv(char* uplo,
           int* n,
           double* alpha,
           double* A,
           int* lda,
           double* x,
           int* incx,
           double* beta,
           double* y,
           int* incy);
void chemv(char* uplo,
           int* n,
           rocblas_float_complex* alpha,
           rocblas_float_complex* A,
           int* lda,
           rocblas_float_complex* x,
           int* incx,
           rocblas_float_complex* beta,
           rocblas_float_complex* y,
           int* incy);
void zhemv(char* uplo,
           int* n,
           rocblas_double_complex* alpha,
           rocblas_double_complex* A,
           int* lda,
           rocblas_double_complex* x,
           int* incx,
           rocblas_double_complex* beta,
           rocblas_double_complex* y,
           int* incy);

void ssymm(char* side,
           char* uplo,
           int* m,
           int* n,
           float* alpha,
           float* A,
           int* lda,
           float* B,
           int* ldb,
           float* beta,
           float* C,
           int* ldc);
void dsymm(char* side,
           char* uplo,
           int* m,
           int* n,
           double* alpha,
           double* A,
           int* lda,
           double* B,
           int* ldb,
           double* beta,
           double* C,
           int* ldc);
void chemm(char* side,
           char* uplo,
           int* m,
           int* n,
           rocblas_float_complex* alpha,
           rocblas_float_complex* A,
           int* lda,
           rocblas_float_complex* B,
           int* ldb,
           rocblas_float_complex* beta,
           rocblas_float_complex* C,
           int* ldc);
void zhemm(char* side,
           char* uplo,
           int* m,
           int* n,
           rocblas_double_complex* alpha,
           rocblas_double_complex* A,
           int* lda,
           rocblas_double_complex* B,
           int* ldb,
           rocblas_double_complex* beta,
           rocblas_double_complex* C,
           int* ldc);

void strtri(char* uplo, char* diag, int* n, float* A, int* lda, int* info);
void dtrtri(char* uplo, char* diag, int* n, double* A, int* lda, int* info);
void ctrtri(char* uplo, char* diag, int* n, rocblas_float_complex* A, int* lda, int* info);
void ztrtri(char* uplo, char* diag, int* n, rocblas_double_complex* A, int* lda, int* info);

void sgetrf(int* m, int* n, float* A, int* lda, int* ipiv, int* info);
void dgetrf(int* m, int* n, double* A, int* lda, int* ipiv, int* info);
void cgetrf(int* m, int* n, rocblas_float_complex* A, int* lda, int* ipiv, int* info);
void zgetrf(int* m, int* n, rocblas_double_complex* A, int* lda, int* ipiv, int* info);

void spotf2(char* uplo, int* n, float* A, int* lda, int* info);
void dpotf2(char* uplo, int* n, double* A, int* lda, int* info);
void cpotf2(char* uplo, int* n, rocblas_float_complex* A, int* lda, int* info);
void zpotf2(char* uplo, int* n, rocblas_double_complex* A, int* lda, int* info);

void spotrf(char* uplo, int* n, float* A, int* lda, int* info);
void dpotrf(char* uplo, int* n, double* A, int* lda, int* info);
void cpotrf(char* uplo, int* n, rocblas_float_complex* A, int* lda, int* info);
void zpotrf(char* uplo, int* n, rocblas_double_complex* A, int* lda, int* info);

void spotrs(char* uplo, int* n, int* nrhs, float* A, int* lda, float* B, int* ldb, int* info);
void dpotrs(char* uplo, int* n, int* nrhs, double* A, int* lda, double* B, int* ldb, int* info);
void cpotrs(char* uplo,
            int* n,
            int* nrhs,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* B,
            int* ldb,
            int* info);
void zpotrs(char* uplo,
            int* n,
            int* nrhs,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* B,
            int* ldb,
            int* info);

void sposv(char* uplo, int* n, int* nrhs, float* A, int* lda, float* B, int* ldb, int* info);
void dposv(char* uplo, int* n, int* nrhs, double* A, int* lda, double* B, int* ldb, int* info);
void cposv(char* uplo,
           int* n,
           int* nrhs,
           rocblas_float_complex* A,
           int* lda,
           rocblas_float_complex* B,
           int* ldb,
           int* info);
void zposv(char* uplo,
           int* n,
           int* nrhs,
           rocblas_double_complex* A,
           int* lda,
           rocblas_double_complex* B,
           int* ldb,
           int* info);

void spotri(char* uplo, int* n, float* A, int* lda, int* info);
void dpotri(char* uplo, int* n, double* A, int* lda, int* info);
void cpotri(char* uplo, int* n, rocblas_float_complex* A, int* lda, int* info);
void zpotri(char* uplo, int* n, rocblas_double_complex* A, int* lda, int* info);

void sgetf2(int* m, int* n, float* A, int* lda, int* ipiv, int* info);
void dgetf2(int* m, int* n, double* A, int* lda, int* ipiv, int* info);
void cgetf2(int* m, int* n, rocblas_float_complex* A, int* lda, int* ipiv, int* info);
void zgetf2(int* m, int* n, rocblas_double_complex* A, int* lda, int* ipiv, int* info);

void sgetrs(char* trans, int* n, int* nrhs, float* A, int* lda, int* ipiv, float* B, int* ldb, int* info);
void dgetrs(char* trans, int* n, int* nrhs, double* A, int* lda, int* ipiv, double* B, int* ldb, int* info);
void cgetrs(char* trans,
            int* n,
            int* nrhs,
            rocblas_float_complex* A,
            int* lda,
            int* ipiv,
            rocblas_float_complex* B,
            int* ldb,
            int* info);
void zgetrs(char* trans,
            int* n,
            int* nrhs,
            rocblas_double_complex* A,
            int* lda,
            int* ipiv,
            rocblas_double_complex* B,
            int* ldb,
            int* info);

void sgesv(int* n, int* nrhs, float* A, int* lda, int* ipiv, float* B, int* ldb, int* info);
void dgesv(int* n, int* nrhs, double* A, int* lda, int* ipiv, double* B, int* ldb, int* info);
void cgesv(int* n,
           int* nrhs,
           rocblas_float_complex* A,
           int* lda,
           int* ipiv,
           rocblas_float_complex* B,
           int* ldb,
           int* info);
void zgesv(int* n,
           int* nrhs,
           rocblas_double_complex* A,
           int* lda,
           int* ipiv,
           rocblas_double_complex* B,
           int* ldb,
           int* info);

void sgels(char* trans,
           int* m,
           int* n,
           int* nrhs,
           float* A,
           int* lda,
           float* B,
           int* ldb,
           float* work,
           int* lwork,
           int* info);
void dgels(char* trans,
           int* m,
           int* n,
           int* nrhs,
           double* A,
           int* lda,
           double* B,
           int* ldb,
           double* work,
           int* lwork,
           int* info);
void cgels(char* trans,
           int* m,
           int* n,
           int* nrhs,
           rocblas_float_complex* A,
           int* lda,
           rocblas_float_complex* B,
           int* ldb,
           rocblas_float_complex* work,
           int* lwork,
           int* info);
void zgels(char* trans,
           int* m,
           int* n,
           int* nrhs,
           rocblas_double_complex* A,
           int* lda,
           rocblas_double_complex* B,
           int* ldb,
           rocblas_double_complex* work,
           int* lwork,
           int* info);

void sgetri(int* n, float* A, int* lda, int* ipiv, float* work, int* lwork, int* info);
void dgetri(int* n, double* A, int* lda, int* ipiv, double* work, int* lwork, int* info);
void cgetri(int* n,
            rocblas_float_complex* A,
            int* lda,
            int* ipiv,
            rocblas_float_complex* work,
            int* lwork,
            int* info);
void zgetri(int* n,
            rocblas_double_complex* A,
            int* lda,
            int* ipiv,
            rocblas_double_complex* work,
            int* lwork,
            int* info);

void strtri(char* uplo, char* diag, int* n, float* A, int* lda, int* info);
void dtrtri(char* uplo, char* diag, int* n, double* A, int* lda, int* info);
void ctrtri(char* uplo, char* diag, int* n, rocblas_float_complex* A, int* lda, int* info);
void ztrtri(char* uplo, char* diag, int* n, rocblas_double_complex* A, int* lda, int* info);

void slarfg(int* n, float* alpha, float* x, int* incx, float* tau);
void dlarfg(int* n, double* alpha, double* x, int* incx, double* tau);
void clarfg(int* n,
            rocblas_float_complex* alpha,
            rocblas_float_complex* x,
            int* incx,
            rocblas_float_complex* tau);
void zlarfg(int* n,
            rocblas_double_complex* alpha,
            rocblas_double_complex* x,
            int* incx,
            rocblas_double_complex* tau);

void slarf(char* side, int* m, int* n, float* x, int* incx, float* alpha, float* A, int* lda, float* work);
void dlarf(char* side,
           int* m,
           int* n,
           double* x,
           int* incx,
           double* alpha,
           double* A,
           int* lda,
           double* work);
void clarf(char* side,
           int* m,
           int* n,
           rocblas_float_complex* x,
           int* incx,
           rocblas_float_complex* alpha,
           rocblas_float_complex* A,
           int* lda,
           rocblas_float_complex* work);
void zlarf(char* side,
           int* m,
           int* n,
           rocblas_double_complex* x,
           int* incx,
           rocblas_double_complex* alpha,
           rocblas_double_complex* A,
           int* lda,
           rocblas_double_complex* work);

void slarft(char* direct, char* storev, int* n, int* k, float* V, int* ldv, float* tau, float* T, int* ldt);
void dlarft(char* direct,
            char* storev,
            int* n,
            int* k,
            double* V,
            int* ldv,
            double* tau,
            double* T,
            int* ldt);
void clarft(char* direct,
            char* storev,
            int* n,
            int* k,
            rocblas_float_complex* V,
            int* ldv,
            rocblas_float_complex* tau,
            rocblas_float_complex* T,
            int* ldt);
void zlarft(char* direct,
            char* storev,
            int* n,
            int* k,
            rocblas_double_complex* V,
            int* ldv,
            rocblas_double_complex* tau,
            rocblas_double_complex* T,
            int* ldt);

void sbdsqr(char* uplo,
            int* n,
            int* nv,
            int* nu,
            int* nc,
            float* D,
            float* E,
            float* V,
            int* ldv,
            float* U,
            int* ldu,
            float* C,
            int* ldc,
            float* W,
            int* info);
void dbdsqr(char* uplo,
            int* n,
            int* nv,
            int* nu,
            int* nc,
            double* D,
            double* E,
            double* V,
            int* ldv,
            double* U,
            int* ldu,
            double* C,
            int* ldc,
            double* W,
            int* info);
void cbdsqr(char* uplo,
            int* n,
            int* nv,
            int* nu,
            int* nc,
            float* D,
            float* E,
            rocblas_float_complex* V,
            int* ldv,
            rocblas_float_complex* U,
            int* ldu,
            rocblas_float_complex* C,
            int* ldc,
            float* W,
            int* info);
void zbdsqr(char* uplo,
            int* n,
            int* nv,
            int* nu,
            int* nc,
            double* D,
            double* E,
            rocblas_double_complex* V,
            int* ldv,
            rocblas_double_complex* U,
            int* ldu,
            rocblas_double_complex* C,
            int* ldc,
            double* W,
            int* info);

void slarfb(char* side,
            char* trans,
            char* direct,
            char* storev,
            int* m,
            int* n,
            int* k,
            float* V,
            int* ldv,
            float* T,
            int* ldt,
            float* A,
            int* lda,
            float* W,
            int* ldw);
void dlarfb(char* side,
            char* trans,
            char* direct,
            char* storev,
            int* m,
            int* n,
            int* k,
            double* V,
            int* ldv,
            double* T,
            int* ldt,
            double* A,
            int* lda,
            double* W,
            int* ldw);
void clarfb(char* side,
            char* trans,
            char* direct,
            char* storev,
            int* m,
            int* n,
            int* k,
            rocblas_float_complex* V,
            int* ldv,
            rocblas_float_complex* T,
            int* ldt,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* W,
            int* ldw);
void zlarfb(char* side,
            char* trans,
            char* direct,
            char* storev,
            int* m,
            int* n,
            int* k,
            rocblas_double_complex* V,
            int* ldv,
            rocblas_double_complex* T,
            int* ldt,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* W,
            int* ldw);

void slatrd(char* uplo, int* n, int* k, float* A, int* lda, float* E, float* tau, float* W, int* ldw);
void dlatrd(char* uplo, int* n, int* k, double* A, int* lda, double* E, double* tau, double* W, int* ldw);
void clatrd(char* uplo,
            int* n,
            int* k,
            rocblas_float_complex* A,
            int* lda,
            float* E,
            rocblas_float_complex* tau,
            rocblas_float_complex* W,
            int* ldw);
void zlatrd(char* uplo,
            int* n,
            int* k,
            rocblas_double_complex* A,
            int* lda,
            double* E,
            rocblas_double_complex* tau,
            rocblas_double_complex* W,
            int* ldw);

void slabrd(int* m,
            int* n,
            int* nb,
            float* A,
            int* lda,
            float* D,
            float* E,
            float* tauq,
            float* taup,
            float* X,
            int* ldx,
            float* Y,
            int* ldy);
void dlabrd(int* m,
            int* n,
            int* nb,
            double* A,
            int* lda,
            double* D,
            double* E,
            double* tauq,
            double* taup,
            double* X,
            int* ldx,
            double* Y,
            int* ldy);
void clabrd(int* m,
            int* n,
            int* nb,
            rocblas_float_complex* A,
            int* lda,
            float* D,
            float* E,
            rocblas_float_complex* tauq,
            rocblas_float_complex* taup,
            rocblas_float_complex* X,
            int* ldx,
            rocblas_float_complex* Y,
            int* ldy);
void zlabrd(int* m,
            int* n,
            int* nb,
            rocblas_double_complex* A,
            int* lda,
            double* D,
            double* E,
            rocblas_double_complex* tauq,
            rocblas_double_complex* taup,
            rocblas_double_complex* X,
            int* ldx,
            rocblas_double_complex* Y,
            int* ldy);

void sgeqr2(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* info);
void dgeqr2(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* info);
void cgeqr2(int* m,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* work,
            int* info);
void zgeqr2(int* m,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* work,
            int* info);

void sgeqrf(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dgeqrf(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cgeqrf(int* m,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* work,
            int* lwork,
            int* info);
void zgeqrf(int* m,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* work,
            int* lwork,
            int* info);

void sgerq2(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* info);
void dgerq2(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* info);
void cgerq2(int* m,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* work,
            int* info);
void zgerq2(int* m,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* work,
            int* info);

void sgerqf(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dgerqf(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cgerqf(int* m,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* work,
            int* lwork,
            int* info);
void zgerqf(int* m,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* work,
            int* lwork,
            int* info);

void sgeql2(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* info);
void dgeql2(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* info);
void cgeql2(int* m,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* work,
            int* info);
void zgeql2(int* m,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* work,
            int* info);

void sgeqlf(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dgeqlf(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cgeqlf(int* m,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* work,
            int* lwork,
            int* info);
void zgeqlf(int* m,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* work,
            int* lwork,
            int* info);

void sgelq2(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* info);
void dgelq2(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* info);
void cgelq2(int* m,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* work,
            int* info);
void zgelq2(int* m,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* work,
            int* info);

void sgelqf(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dgelqf(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cgelqf(int* m,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* work,
            int* lwork,
            int* info);
void zgelqf(int* m,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* work,
            int* lwork,
            int* info);

void clacgv(int* n, rocblas_float_complex* x, int* incx);
void zlacgv(int* n, rocblas_double_complex* x, int* incx);

void slaswp(int* n, float* A, int* lda, int* k1, int* k2, int* ipiv, int* inc);
void dlaswp(int* n, double* A, int* lda, int* k1, int* k2, int* ipiv, int* inc);
void claswp(int* n, rocblas_float_complex* A, int* lda, int* k1, int* k2, int* ipiv, int* inc);
void zlaswp(int* n, rocblas_double_complex* A, int* lda, int* k1, int* k2, int* ipiv, int* inc);

void sorg2r(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* info);
void dorg2r(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* info);
void cung2r(int* m,
            int* n,
            int* k,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* work,
            int* info);
void zung2r(int* m,
            int* n,
            int* k,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* work,
            int* info);

void sorgqr(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dorgqr(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cungqr(int* m,
            int* n,
            int* k,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* work,
            int* lwork,
            int* info);
void zungqr(int* m,
            int* n,
            int* k,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* work,
            int* lwork,
            int* info);

void sorgl2(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* info);
void dorgl2(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* info);
void cungl2(int* m,
            int* n,
            int* k,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* work,
            int* info);
void zungl2(int* m,
            int* n,
            int* k,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* work,
            int* info);

void sorglq(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dorglq(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cunglq(int* m,
            int* n,
            int* k,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* work,
            int* lwork,
            int* info);
void zunglq(int* m,
            int* n,
            int* k,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* work,
            int* lwork,
            int* info);

void sorg2l(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* info);
void dorg2l(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* info);
void cung2l(int* m,
            int* n,
            int* k,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* work,
            int* info);
void zung2l(int* m,
            int* n,
            int* k,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* work,
            int* info);

void sorgql(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dorgql(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cungql(int* m,
            int* n,
            int* k,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* work,
            int* lwork,
            int* info);
void zungql(int* m,
            int* n,
            int* k,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* work,
            int* lwork,
            int* info);

void sorgbr(char* vect,
            int* m,
            int* n,
            int* k,
            float* A,
            int* lda,
            float* Ipiv,
            float* work,
            int* size_w,
            int* info);
void dorgbr(char* vect,
            int* m,
            int* n,
            int* k,
            double* A,
            int* lda,
            double* Ipiv,
            double* work,
            int* size_w,
            int* info);
void cungbr(char* vect,
            int* m,
            int* n,
            int* k,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* Ipiv,
            rocblas_float_complex* work,
            int* size_w,
            int* info);
void zungbr(char* vect,
            int* m,
            int* n,
            int* k,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* Ipiv,
            rocblas_double_complex* work,
            int* size_w,
            int* info);

void sorgtr(char* uplo, int* n, float* A, int* lda, float* Ipiv, float* work, int* size_w, int* info);
void dorgtr(char* uplo, int* n, double* A, int* lda, double* Ipiv, double* work, int* size_w, int* info);
void cungtr(char* uplo,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* Ipiv,
            rocblas_float_complex* work,
            int* size_w,
            int* info);
void zungtr(char* uplo,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* Ipiv,
            rocblas_double_complex* work,
            int* size_w,
            int* info);

void sorm2r(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            float* A,
            int* lda,
            float* ipiv,
            float* C,
            int* ldc,
            float* work,
            int* info);
void dorm2r(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            double* A,
            int* lda,
            double* ipiv,
            double* C,
            int* ldc,
            double* work,
            int* info);
void cunm2r(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* C,
            int* ldc,
            rocblas_float_complex* work,
            int* info);
void zunm2r(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* C,
            int* ldc,
            rocblas_double_complex* work,
            int* info);

void sormqr(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            float* A,
            int* lda,
            float* ipiv,
            float* C,
            int* ldc,
            float* work,
            int* sizeW,
            int* info);
void dormqr(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            double* A,
            int* lda,
            double* ipiv,
            double* C,
            int* ldc,
            double* work,
            int* sizeW,
            int* info);
void cunmqr(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* C,
            int* ldc,
            rocblas_float_complex* work,
            int* sizeW,
            int* info);
void zunmqr(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* C,
            int* ldc,
            rocblas_double_complex* work,
            int* sizeW,
            int* info);

void sorml2(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            float* A,
            int* lda,
            float* ipiv,
            float* C,
            int* ldc,
            float* work,
            int* info);
void dorml2(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            double* A,
            int* lda,
            double* ipiv,
            double* C,
            int* ldc,
            double* work,
            int* info);
void cunml2(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* C,
            int* ldc,
            rocblas_float_complex* work,
            int* info);
void zunml2(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* C,
            int* ldc,
            rocblas_double_complex* work,
            int* info);

void sormlq(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            float* A,
            int* lda,
            float* ipiv,
            float* C,
            int* ldc,
            float* work,
            int* sizeW,
            int* info);
void dormlq(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            double* A,
            int* lda,
            double* ipiv,
            double* C,
            int* ldc,
            double* work,
            int* sizeW,
            int* info);
void cunmlq(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* C,
            int* ldc,
            rocblas_float_complex* work,
            int* sizeW,
            int* info);
void zunmlq(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* C,
            int* ldc,
            rocblas_double_complex* work,
            int* sizeW,
            int* info);

void sorm2l(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            float* A,
            int* lda,
            float* ipiv,
            float* C,
            int* ldc,
            float* work,
            int* info);
void dorm2l(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            double* A,
            int* lda,
            double* ipiv,
            double* C,
            int* ldc,
            double* work,
            int* info);
void cunm2l(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* C,
            int* ldc,
            rocblas_float_complex* work,
            int* info);
void zunm2l(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* C,
            int* ldc,
            rocblas_double_complex* work,
            int* info);

void sormql(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            float* A,
            int* lda,
            float* ipiv,
            float* C,
            int* ldc,
            float* work,
            int* sizeW,
            int* info);
void dormql(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            double* A,
            int* lda,
            double* ipiv,
            double* C,
            int* ldc,
            double* work,
            int* sizeW,
            int* info);
void cunmql(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* C,
            int* ldc,
            rocblas_float_complex* work,
            int* sizeW,
            int* info);
void zunmql(char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* C,
            int* ldc,
            rocblas_double_complex* work,
            int* sizeW,
            int* info);

void sormbr(char* vect,
            char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            float* A,
            int* lda,
            float* ipiv,
            float* C,
            int* ldc,
            float* work,
            int* sizeW,
            int* info);
void dormbr(char* vect,
            char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            double* A,
            int* lda,
            double* ipiv,
            double* C,
            int* ldc,
            double* work,
            int* sizeW,
            int* info);
void cunmbr(char* vect,
            char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* C,
            int* ldc,
            rocblas_float_complex* work,
            int* sizeW,
            int* info);
void zunmbr(char* vect,
            char* side,
            char* trans,
            int* m,
            int* n,
            int* k,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* C,
            int* ldc,
            rocblas_double_complex* work,
            int* sizeW,
            int* info);

void sormtr(char* side,
            char* uplo,
            char* trans,
            int* m,
            int* n,
            float* A,
            int* lda,
            float* ipiv,
            float* C,
            int* ldc,
            float* work,
            int* sizeW,
            int* info);
void dormtr(char* side,
            char* uplo,
            char* trans,
            int* m,
            int* n,
            double* A,
            int* lda,
            double* ipiv,
            double* C,
            int* ldc,
            double* work,
            int* sizeW,
            int* info);
void cunmtr(char* side,
            char* uplo,
            char* trans,
            int* m,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* ipiv,
            rocblas_float_complex* C,
            int* ldc,
            rocblas_float_complex* work,
            int* sizeW,
            int* info);
void zunmtr(char* side,
            char* uplo,
            char* trans,
            int* m,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* ipiv,
            rocblas_double_complex* C,
            int* ldc,
            rocblas_double_complex* work,
            int* sizeW,
            int* info);

void sgebd2(int* m,
            int* n,
            float* A,
            int* lda,
            float* D,
            float* E,
            float* tauq,
            float* taup,
            float* work,
            int* info);
void dgebd2(int* m,
            int* n,
            double* A,
            int* lda,
            double* D,
            double* E,
            double* tauq,
            double* taup,
            double* work,
            int* info);
void cgebd2(int* m,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            float* D,
            float* E,
            rocblas_float_complex* tauq,
            rocblas_float_complex* taup,
            rocblas_float_complex* work,
            int* info);
void zgebd2(int* m,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            double* D,
            double* E,
            rocblas_double_complex* tauq,
            rocblas_double_complex* taup,
            rocblas_double_complex* work,
            int* info);

void sgebrd(int* m,
            int* n,
            float* A,
            int* lda,
            float* D,
            float* E,
            float* tauq,
            float* taup,
            float* work,
            int* size_w,
            int* info);
void dgebrd(int* m,
            int* n,
            double* A,
            int* lda,
            double* D,
            double* E,
            double* tauq,
            double* taup,
            double* work,
            int* size_w,
            int* info);
void cgebrd(int* m,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            float* D,
            float* E,
            rocblas_float_complex* tauq,
            rocblas_float_complex* taup,
            rocblas_float_complex* work,
            int* size_w,
            int* info);
void zgebrd(int* m,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            double* D,
            double* E,
            rocblas_double_complex* tauq,
            rocblas_double_complex* taup,
            rocblas_double_complex* work,
            int* size_w,
            int* info);

void ssytrd(char* uplo,
            int* n,
            float* A,
            int* lda,
            float* D,
            float* E,
            float* tau,
            float* work,
            int* size_w,
            int* info);
void dsytrd(char* uplo,
            int* n,
            double* A,
            int* lda,
            double* D,
            double* E,
            double* tau,
            double* work,
            int* size_w,
            int* info);
void chetrd(char* uplo,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            float* D,
            float* E,
            rocblas_float_complex* tau,
            rocblas_float_complex* work,
            int* size_w,
            int* info);
void zhetrd(char* uplo,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            double* D,
            double* E,
            rocblas_double_complex* tau,
            rocblas_double_complex* work,
            int* size_w,
            int* info);

void ssytd2(char* uplo, int* n, float* A, int* lda, float* D, float* E, float* tau, int* info);
void dsytd2(char* uplo, int* n, double* A, int* lda, double* D, double* E, double* tau, int* info);
void chetd2(char* uplo,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            float* D,
            float* E,
            rocblas_float_complex* tau,
            int* info);
void zhetd2(char* uplo,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            double* D,
            double* E,
            rocblas_double_complex* tau,
            int* info);

void sgesvd(char* jobu,
            char* jobv,
            int* m,
            int* n,
            float* A,
            int* lda,
            float* S,
            float* U,
            int* ldu,
            float* V,
            int* ldv,
            float* E,
            int* lwork,
            int* info);
void dgesvd(char* jobu,
            char* jobv,
            int* m,
            int* n,
            double* A,
            int* lda,
            double* S,
            double* U,
            int* ldu,
            double* V,
            int* ldv,
            double* E,
            int* lwork,
            int* info);
void cgesvd(char* jobu,
            char* jobv,
            int* m,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            float* S,
            rocblas_float_complex* U,
            int* ldu,
            rocblas_float_complex* V,
            int* ldv,
            rocblas_float_complex* work,
            int* lwork,
            float* E,
            int* info);
void zgesvd(char* jobu,
            char* jobv,
            int* m,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            double* S,
            rocblas_double_complex* U,
            int* ldu,
            rocblas_double_complex* V,
            int* ldv,
            rocblas_double_complex* work,
            int* lwork,
            double* E,
            int* info);

void ssterf(int* n, float* D, float* E, int* info);
void dsterf(int* n, double* D, double* E, int* info);

void ssteqr(char* evect, int* n, float* D, float* E, float* C, int* ldc, float* work, int* info);
void dsteqr(char* evect, int* n, double* D, double* E, double* C, int* ldc, double* work, int* info);
void csteqr(char* evect,
            int* n,
            float* D,
            float* E,
            rocblas_float_complex* C,
            int* ldc,
            float* work,
            int* info);
void zsteqr(char* evect,
            int* n,
            double* D,
            double* E,
            rocblas_double_complex* C,
            int* ldc,
            double* work,
            int* info);

void sstedc(char* evect,
            int* n,
            float* D,
            float* E,
            float* C,
            int* ldc,
            float* work,
            int* lwork,
            int* iwork,
            int* liwork,
            int* info);
void dstedc(char* evect,
            int* n,
            double* D,
            double* E,
            double* C,
            int* ldc,
            double* work,
            int* lwork,
            int* iwork,
            int* liwork,
            int* info);
void cstedc(char* evect,
            int* n,
            float* D,
            float* E,
            rocblas_float_complex* C,
            int* ldc,
            rocblas_float_complex* work,
            int* lwork,
            float* rwork,
            int* lrwork,
            int* iwork,
            int* liwork,
            int* info);
void zstedc(char* evect,
            int* n,
            double* D,
            double* E,
            rocblas_double_complex* C,
            int* ldc,
            rocblas_double_complex* work,
            int* lwork,
            double* rwork,
            int* lrwork,
            int* iwork,
            int* liwork,
            int* info);

void ssygs2(int* itype, char* uplo, int* n, float* A, int* lda, float* B, int* ldb, int* info);
void dsygs2(int* itype, char* uplo, int* n, double* A, int* lda, double* B, int* ldb, int* info);
void chegs2(int* itype,
            char* uplo,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* B,
            int* ldb,
            int* info);
void zhegs2(int* itype,
            char* uplo,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* B,
            int* ldb,
            int* info);

void ssygst(int* itype, char* uplo, int* n, float* A, int* lda, float* B, int* ldb, int* info);
void dsygst(int* itype, char* uplo, int* n, double* A, int* lda, double* B, int* ldb, int* info);
void chegst(int* itype,
            char* uplo,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* B,
            int* ldb,
            int* info);
void zhegst(int* itype,
            char* uplo,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* B,
            int* ldb,
            int* info);

void ssyev(char* evect, char* uplo, int* n, float* A, int* lda, float* D, float* work, int* lwork, int* info);
void dsyev(char* evect,
           char* uplo,
           int* n,
           double* A,
           int* lda,
           double* D,
           double* work,
           int* lwork,
           int* info);
void cheev(char* evect,
           char* uplo,
           int* n,
           rocblas_float_complex* A,
           int* lda,
           float* D,
           rocblas_float_complex* work,
           int* lwork,
           float* rwork,
           int* info);
void zheev(char* evect,
           char* uplo,
           int* n,
           rocblas_double_complex* A,
           int* lda,
           double* D,
           rocblas_double_complex* work,
           int* lwork,
           double* rwork,
           int* info);

void ssyevd(char* evect,
            char* uplo,
            int* n,
            float* A,
            int* lda,
            float* D,
            float* work,
            int* lwork,
            int* iwork,
            int* liwork,
            int* info);
void dsyevd(char* evect,
            char* uplo,
            int* n,
            double* A,
            int* lda,
            double* D,
            double* work,
            int* lwork,
            int* iwork,
            int* liwork,
            int* info);
void cheevd(char* evect,
            char* uplo,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            float* D,
            rocblas_float_complex* work,
            int* lwork,
            float* rwork,
            int* lrwork,
            int* iwork,
            int* liwork,
            int* info);
void zheevd(char* evect,
            char* uplo,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            double* D,
            rocblas_double_complex* work,
            int* lwork,
            double* rwork,
            int* lrwork,
            int* iwork,
            int* liwork,
            int* info);

void ssygv(int* itype,
           char* evect,
           char* uplo,
           int* n,
           float* A,
           int* lda,
           float* B,
           int* ldb,
           float* W,
           float* work,
           int* lwork,
           int* info);
void dsygv(int* itype,
           char* evect,
           char* uplo,
           int* n,
           double* A,
           int* lda,
           double* B,
           int* ldb,
           double* W,
           double* work,
           int* lwork,
           int* info);
void chegv(int* itype,
           char* evect,
           char* uplo,
           int* n,
           rocblas_float_complex* A,
           int* lda,
           rocblas_float_complex* B,
           int* ldb,
           float* W,
           rocblas_float_complex* work,
           int* lwork,
           float* rwork,
           int* info);
void zhegv(int* itype,
           char* evect,
           char* uplo,
           int* n,
           rocblas_double_complex* A,
           int* lda,
           rocblas_double_complex* B,
           int* ldb,
           double* W,
           rocblas_double_complex* work,
           int* lwork,
           double* rwork,
           int* info);

void ssygvd(int* itype,
            char* evect,
            char* uplo,
            int* n,
            float* A,
            int* lda,
            float* B,
            int* ldb,
            float* W,
            float* work,
            int* lwork,
            int* iwork,
            int* liwork,
            int* info);
void dsygvd(int* itype,
            char* evect,
            char* uplo,
            int* n,
            double* A,
            int* lda,
            double* B,
            int* ldb,
            double* W,
            double* work,
            int* lwork,
            int* iwork,
            int* liwork,
            int* info);
void chegvd(int* itype,
            char* evect,
            char* uplo,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* B,
            int* ldb,
            float* W,
            rocblas_float_complex* work,
            int* lwork,
            float* rwork,
            int* lrwork,
            int* iwork,
            int* liwork,
            int* info);
void zhegvd(int* itype,
            char* evect,
            char* uplo,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* B,
            int* ldb,
            double* W,
            rocblas_double_complex* work,
            int* lwork,
            double* rwork,
            int* lrwork,
            int* iwork,
            int* liwork,
            int* info);

#ifdef __cplusplus
}
#endif
/************************************************************************/

/************************************************************************/
// These are templated functions used in rocSOLVER clients code

// lacgv

template <>
void cblas_lacgv<rocblas_float_complex>(rocblas_int n, rocblas_float_complex* x, rocblas_int incx)
{
    clacgv(&n, x, &incx);
}

template <>
void cblas_lacgv<rocblas_double_complex>(rocblas_int n, rocblas_double_complex* x, rocblas_int incx)
{
    zlacgv(&n, x, &incx);
}

// laswp

template <>
void cblas_laswp<float>(rocblas_int n,
                        float* A,
                        rocblas_int lda,
                        rocblas_int k1,
                        rocblas_int k2,
                        rocblas_int* ipiv,
                        rocblas_int inc)
{
    slaswp(&n, A, &lda, &k1, &k2, ipiv, &inc);
}

template <>
void cblas_laswp<double>(rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         rocblas_int k1,
                         rocblas_int k2,
                         rocblas_int* ipiv,
                         rocblas_int inc)
{
    dlaswp(&n, A, &lda, &k1, &k2, ipiv, &inc);
}

template <>
void cblas_laswp<rocblas_float_complex>(rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_int k1,
                                        rocblas_int k2,
                                        rocblas_int* ipiv,
                                        rocblas_int inc)
{
    claswp(&n, A, &lda, &k1, &k2, ipiv, &inc);
}

template <>
void cblas_laswp<rocblas_double_complex>(rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_int k1,
                                         rocblas_int k2,
                                         rocblas_int* ipiv,
                                         rocblas_int inc)
{
    zlaswp(&n, A, &lda, &k1, &k2, ipiv, &inc);
}

// larfg

template <>
void cblas_larfg<float>(rocblas_int n, float* alpha, float* x, rocblas_int incx, float* tau)
{
    slarfg(&n, alpha, x, &incx, tau);
}

template <>
void cblas_larfg<double>(rocblas_int n, double* alpha, double* x, rocblas_int incx, double* tau)
{
    dlarfg(&n, alpha, x, &incx, tau);
}

template <>
void cblas_larfg<rocblas_float_complex>(rocblas_int n,
                                        rocblas_float_complex* alpha,
                                        rocblas_float_complex* x,
                                        rocblas_int incx,
                                        rocblas_float_complex* tau)
{
    clarfg(&n, alpha, x, &incx, tau);
}

template <>
void cblas_larfg<rocblas_double_complex>(rocblas_int n,
                                         rocblas_double_complex* alpha,
                                         rocblas_double_complex* x,
                                         rocblas_int incx,
                                         rocblas_double_complex* tau)
{
    zlarfg(&n, alpha, x, &incx, tau);
}

// larf

template <>
void cblas_larf<float>(rocblas_side sideR,
                       rocblas_int m,
                       rocblas_int n,
                       float* x,
                       rocblas_int incx,
                       float* alpha,
                       float* A,
                       rocblas_int lda,
                       float* work)
{
    char side = rocblas2char_side(sideR);
    slarf(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

template <>
void cblas_larf<double>(rocblas_side sideR,
                        rocblas_int m,
                        rocblas_int n,
                        double* x,
                        rocblas_int incx,
                        double* alpha,
                        double* A,
                        rocblas_int lda,
                        double* work)
{
    char side = rocblas2char_side(sideR);
    dlarf(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

template <>
void cblas_larf<rocblas_float_complex>(rocblas_side sideR,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_float_complex* x,
                                       rocblas_int incx,
                                       rocblas_float_complex* alpha,
                                       rocblas_float_complex* A,
                                       rocblas_int lda,
                                       rocblas_float_complex* work)
{
    char side = rocblas2char_side(sideR);
    clarf(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

template <>
void cblas_larf<rocblas_double_complex>(rocblas_side sideR,
                                        rocblas_int m,
                                        rocblas_int n,
                                        rocblas_double_complex* x,
                                        rocblas_int incx,
                                        rocblas_double_complex* alpha,
                                        rocblas_double_complex* A,
                                        rocblas_int lda,
                                        rocblas_double_complex* work)
{
    char side = rocblas2char_side(sideR);
    zlarf(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

// larft

template <>
void cblas_larft<float>(rocblas_direct directR,
                        rocblas_storev storevR,
                        rocblas_int n,
                        rocblas_int k,
                        float* V,
                        rocblas_int ldv,
                        float* tau,
                        float* T,
                        rocblas_int ldt)
{
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    slarft(&direct, &storev, &n, &k, V, &ldv, tau, T, &ldt);
}

template <>
void cblas_larft<double>(rocblas_direct directR,
                         rocblas_storev storevR,
                         rocblas_int n,
                         rocblas_int k,
                         double* V,
                         rocblas_int ldv,
                         double* tau,
                         double* T,
                         rocblas_int ldt)
{
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    dlarft(&direct, &storev, &n, &k, V, &ldv, tau, T, &ldt);
}

template <>
void cblas_larft<rocblas_float_complex>(rocblas_direct directR,
                                        rocblas_storev storevR,
                                        rocblas_int n,
                                        rocblas_int k,
                                        rocblas_float_complex* V,
                                        rocblas_int ldv,
                                        rocblas_float_complex* tau,
                                        rocblas_float_complex* T,
                                        rocblas_int ldt)
{
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    clarft(&direct, &storev, &n, &k, V, &ldv, tau, T, &ldt);
}

template <>
void cblas_larft<rocblas_double_complex>(rocblas_direct directR,
                                         rocblas_storev storevR,
                                         rocblas_int n,
                                         rocblas_int k,
                                         rocblas_double_complex* V,
                                         rocblas_int ldv,
                                         rocblas_double_complex* tau,
                                         rocblas_double_complex* T,
                                         rocblas_int ldt)
{
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    zlarft(&direct, &storev, &n, &k, V, &ldv, tau, T, &ldt);
}

// larfb

template <>
void cblas_larfb<float>(rocblas_side sideR,
                        rocblas_operation transR,
                        rocblas_direct directR,
                        rocblas_storev storevR,
                        rocblas_int m,
                        rocblas_int n,
                        rocblas_int k,
                        float* V,
                        rocblas_int ldv,
                        float* T,
                        rocblas_int ldt,
                        float* A,
                        rocblas_int lda,
                        float* W,
                        rocblas_int ldw)
{
    char side = rocblas2char_side(sideR);
    char trans = rocblas2char_operation(transR);
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    slarfb(&side, &trans, &direct, &storev, &m, &n, &k, V, &ldv, T, &ldt, A, &lda, W, &ldw);
}

template <>
void cblas_larfb<double>(rocblas_side sideR,
                         rocblas_operation transR,
                         rocblas_direct directR,
                         rocblas_storev storevR,
                         rocblas_int m,
                         rocblas_int n,
                         rocblas_int k,
                         double* V,
                         rocblas_int ldv,
                         double* T,
                         rocblas_int ldt,
                         double* A,
                         rocblas_int lda,
                         double* W,
                         rocblas_int ldw)
{
    char side = rocblas2char_side(sideR);
    char trans = rocblas2char_operation(transR);
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    dlarfb(&side, &trans, &direct, &storev, &m, &n, &k, V, &ldv, T, &ldt, A, &lda, W, &ldw);
}

template <>
void cblas_larfb<rocblas_float_complex>(rocblas_side sideR,
                                        rocblas_operation transR,
                                        rocblas_direct directR,
                                        rocblas_storev storevR,
                                        rocblas_int m,
                                        rocblas_int n,
                                        rocblas_int k,
                                        rocblas_float_complex* V,
                                        rocblas_int ldv,
                                        rocblas_float_complex* T,
                                        rocblas_int ldt,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* W,
                                        rocblas_int ldw)
{
    char side = rocblas2char_side(sideR);
    char trans = rocblas2char_operation(transR);
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    clarfb(&side, &trans, &direct, &storev, &m, &n, &k, V, &ldv, T, &ldt, A, &lda, W, &ldw);
}

template <>
void cblas_larfb<rocblas_double_complex>(rocblas_side sideR,
                                         rocblas_operation transR,
                                         rocblas_direct directR,
                                         rocblas_storev storevR,
                                         rocblas_int m,
                                         rocblas_int n,
                                         rocblas_int k,
                                         rocblas_double_complex* V,
                                         rocblas_int ldv,
                                         rocblas_double_complex* T,
                                         rocblas_int ldt,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* W,
                                         rocblas_int ldw)
{
    char side = rocblas2char_side(sideR);
    char trans = rocblas2char_operation(transR);
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    zlarfb(&side, &trans, &direct, &storev, &m, &n, &k, V, &ldv, T, &ldt, A, &lda, W, &ldw);
}

// bdsqr
template <>
void cblas_bdsqr(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_int nv,
                 rocblas_int nu,
                 rocblas_int nc,
                 float* D,
                 float* E,
                 float* V,
                 rocblas_int ldv,
                 float* U,
                 rocblas_int ldu,
                 float* C,
                 rocblas_int ldc,
                 float* work,
                 rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    sbdsqr(&uploC, &n, &nv, &nu, &nc, D, E, V, &ldv, U, &ldu, C, &ldc, work, info);
}

template <>
void cblas_bdsqr(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_int nv,
                 rocblas_int nu,
                 rocblas_int nc,
                 double* D,
                 double* E,
                 double* V,
                 rocblas_int ldv,
                 double* U,
                 rocblas_int ldu,
                 double* C,
                 rocblas_int ldc,
                 double* work,
                 rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    dbdsqr(&uploC, &n, &nv, &nu, &nc, D, E, V, &ldv, U, &ldu, C, &ldc, work, info);
}

template <>
void cblas_bdsqr(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_int nv,
                 rocblas_int nu,
                 rocblas_int nc,
                 float* D,
                 float* E,
                 rocblas_float_complex* V,
                 rocblas_int ldv,
                 rocblas_float_complex* U,
                 rocblas_int ldu,
                 rocblas_float_complex* C,
                 rocblas_int ldc,
                 float* work,
                 rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    cbdsqr(&uploC, &n, &nv, &nu, &nc, D, E, V, &ldv, U, &ldu, C, &ldc, work, info);
}

template <>
void cblas_bdsqr(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_int nv,
                 rocblas_int nu,
                 rocblas_int nc,
                 double* D,
                 double* E,
                 rocblas_double_complex* V,
                 rocblas_int ldv,
                 rocblas_double_complex* U,
                 rocblas_int ldu,
                 rocblas_double_complex* C,
                 rocblas_int ldc,
                 double* work,
                 rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    zbdsqr(&uploC, &n, &nv, &nu, &nc, D, E, V, &ldv, U, &ldu, C, &ldc, work, info);
}

// gesvd
template <>
void cblas_gesvd(rocblas_svect leftv,
                 rocblas_svect rightv,
                 rocblas_int m,
                 rocblas_int n,
                 float* A,
                 rocblas_int lda,
                 float* S,
                 float* U,
                 rocblas_int ldu,
                 float* V,
                 rocblas_int ldv,
                 float* work,
                 rocblas_int lwork,
                 float* E,
                 rocblas_int* info)
{
    char jobu = rocblas2char_svect(leftv);
    char jobv = rocblas2char_svect(rightv);
    sgesvd(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, V, &ldv, E, &lwork, info);
}

template <>
void cblas_gesvd(rocblas_svect leftv,
                 rocblas_svect rightv,
                 rocblas_int m,
                 rocblas_int n,
                 double* A,
                 rocblas_int lda,
                 double* S,
                 double* U,
                 rocblas_int ldu,
                 double* V,
                 rocblas_int ldv,
                 double* work,
                 rocblas_int lwork,
                 double* E,
                 rocblas_int* info)
{
    char jobu = rocblas2char_svect(leftv);
    char jobv = rocblas2char_svect(rightv);
    dgesvd(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, V, &ldv, E, &lwork, info);
}

template <>
void cblas_gesvd(rocblas_svect leftv,
                 rocblas_svect rightv,
                 rocblas_int m,
                 rocblas_int n,
                 rocblas_float_complex* A,
                 rocblas_int lda,
                 float* S,
                 rocblas_float_complex* U,
                 rocblas_int ldu,
                 rocblas_float_complex* V,
                 rocblas_int ldv,
                 rocblas_float_complex* work,
                 rocblas_int lwork,
                 float* E,
                 rocblas_int* info)
{
    char jobu = rocblas2char_svect(leftv);
    char jobv = rocblas2char_svect(rightv);
    cgesvd(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, V, &ldv, work, &lwork, E, info);
}

template <>
void cblas_gesvd(rocblas_svect leftv,
                 rocblas_svect rightv,
                 rocblas_int m,
                 rocblas_int n,
                 rocblas_double_complex* A,
                 rocblas_int lda,
                 double* S,
                 rocblas_double_complex* U,
                 rocblas_int ldu,
                 rocblas_double_complex* V,
                 rocblas_int ldv,
                 rocblas_double_complex* work,
                 rocblas_int lwork,
                 double* E,
                 rocblas_int* info)
{
    char jobu = rocblas2char_svect(leftv);
    char jobv = rocblas2char_svect(rightv);
    zgesvd(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, V, &ldv, work, &lwork, E, info);
}

// latrd
template <>
void cblas_latrd<float, float>(rocblas_fill uplo,
                               rocblas_int n,
                               rocblas_int k,
                               float* A,
                               rocblas_int lda,
                               float* E,
                               float* tau,
                               float* W,
                               rocblas_int ldw)
{
    char uploC = rocblas2char_fill(uplo);
    slatrd(&uploC, &n, &k, A, &lda, E, tau, W, &ldw);
}

template <>
void cblas_latrd<double, double>(rocblas_fill uplo,
                                 rocblas_int n,
                                 rocblas_int k,
                                 double* A,
                                 rocblas_int lda,
                                 double* E,
                                 double* tau,
                                 double* W,
                                 rocblas_int ldw)
{
    char uploC = rocblas2char_fill(uplo);
    dlatrd(&uploC, &n, &k, A, &lda, E, tau, W, &ldw);
}

template <>
void cblas_latrd<rocblas_float_complex, float>(rocblas_fill uplo,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_float_complex* A,
                                               rocblas_int lda,
                                               float* E,
                                               rocblas_float_complex* tau,
                                               rocblas_float_complex* W,
                                               rocblas_int ldw)
{
    char uploC = rocblas2char_fill(uplo);
    clatrd(&uploC, &n, &k, A, &lda, E, tau, W, &ldw);
}

template <>
void cblas_latrd<rocblas_double_complex, double>(rocblas_fill uplo,
                                                 rocblas_int n,
                                                 rocblas_int k,
                                                 rocblas_double_complex* A,
                                                 rocblas_int lda,
                                                 double* E,
                                                 rocblas_double_complex* tau,
                                                 rocblas_double_complex* W,
                                                 rocblas_int ldw)
{
    char uploC = rocblas2char_fill(uplo);
    zlatrd(&uploC, &n, &k, A, &lda, E, tau, W, &ldw);
}

// labrd
template <>
void cblas_labrd<float, float>(rocblas_int m,
                               rocblas_int n,
                               rocblas_int nb,
                               float* A,
                               rocblas_int lda,
                               float* D,
                               float* E,
                               float* tauq,
                               float* taup,
                               float* X,
                               rocblas_int ldx,
                               float* Y,
                               rocblas_int ldy)
{
    int info;
    slabrd(&m, &n, &nb, A, &lda, D, E, tauq, taup, X, &ldx, Y, &ldy);
}

template <>
void cblas_labrd<double, double>(rocblas_int m,
                                 rocblas_int n,
                                 rocblas_int nb,
                                 double* A,
                                 rocblas_int lda,
                                 double* D,
                                 double* E,
                                 double* tauq,
                                 double* taup,
                                 double* X,
                                 rocblas_int ldx,
                                 double* Y,
                                 rocblas_int ldy)
{
    int info;
    dlabrd(&m, &n, &nb, A, &lda, D, E, tauq, taup, X, &ldx, Y, &ldy);
}

template <>
void cblas_labrd<rocblas_float_complex, float>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int nb,
                                               rocblas_float_complex* A,
                                               rocblas_int lda,
                                               float* D,
                                               float* E,
                                               rocblas_float_complex* tauq,
                                               rocblas_float_complex* taup,
                                               rocblas_float_complex* X,
                                               rocblas_int ldx,
                                               rocblas_float_complex* Y,
                                               rocblas_int ldy)
{
    int info;
    clabrd(&m, &n, &nb, A, &lda, D, E, tauq, taup, X, &ldx, Y, &ldy);
}

template <>
void cblas_labrd<rocblas_double_complex, double>(rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_int nb,
                                                 rocblas_double_complex* A,
                                                 rocblas_int lda,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* tauq,
                                                 rocblas_double_complex* taup,
                                                 rocblas_double_complex* X,
                                                 rocblas_int ldx,
                                                 rocblas_double_complex* Y,
                                                 rocblas_int ldy)
{
    int info;
    zlabrd(&m, &n, &nb, A, &lda, D, E, tauq, taup, X, &ldx, Y, &ldy);
}

// orgqr & ungqr
template <>
void cblas_orgqr_ungqr<float>(rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* work,
                              rocblas_int lwork)
{
    int info;
    sorgqr(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orgqr_ungqr<double>(rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* work,
                               rocblas_int lwork)
{
    int info;
    dorgqr(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orgqr_ungqr<rocblas_float_complex>(rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* work,
                                              rocblas_int lwork)
{
    int info;
    cungqr(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orgqr_ungqr<rocblas_double_complex>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork)
{
    int info;
    zungqr(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

// org2r & ung2r
template <>
void cblas_org2r_ung2r<float>(rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* work)
{
    int info;
    sorg2r(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_org2r_ung2r<double>(rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* work)
{
    int info;
    dorg2r(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_org2r_ung2r<rocblas_float_complex>(rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* work)
{
    int info;
    cung2r(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_org2r_ung2r<rocblas_double_complex>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* work)
{
    int info;
    zung2r(&m, &n, &k, A, &lda, ipiv, work, &info);
}

// orglq & unglq
template <>
void cblas_orglq_unglq<float>(rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* work,
                              rocblas_int lwork)
{
    int info;
    sorglq(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orglq_unglq<double>(rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* work,
                               rocblas_int lwork)
{
    int info;
    dorglq(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orglq_unglq<rocblas_float_complex>(rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* work,
                                              rocblas_int lwork)
{
    int info;
    cunglq(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orglq_unglq<rocblas_double_complex>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork)
{
    int info;
    zunglq(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

// orgl2 & ungl2
template <>
void cblas_orgl2_ungl2<float>(rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* work)
{
    int info;
    sorgl2(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_orgl2_ungl2<double>(rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* work)
{
    int info;
    dorgl2(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_orgl2_ungl2<rocblas_float_complex>(rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* work)
{
    int info;
    cungl2(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_orgl2_ungl2<rocblas_double_complex>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* work)
{
    int info;
    zungl2(&m, &n, &k, A, &lda, ipiv, work, &info);
}

// orgql & ungql
template <>
void cblas_orgql_ungql<float>(rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* work,
                              rocblas_int lwork)
{
    int info;
    sorgql(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orgql_ungql<double>(rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* work,
                               rocblas_int lwork)
{
    int info;
    dorgql(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orgql_ungql<rocblas_float_complex>(rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* work,
                                              rocblas_int lwork)
{
    int info;
    cungql(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_orgql_ungql<rocblas_double_complex>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork)
{
    int info;
    zungql(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

// org2l & ung2l
template <>
void cblas_org2l_ung2l<float>(rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* work)
{
    int info;
    sorg2l(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_org2l_ung2l<double>(rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* work)
{
    int info;
    dorg2l(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_org2l_ung2l<rocblas_float_complex>(rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* work)
{
    int info;
    cung2l(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cblas_org2l_ung2l<rocblas_double_complex>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* work)
{
    int info;
    zung2l(&m, &n, &k, A, &lda, ipiv, work, &info);
}

// orgbr & ungbr
template <>
void cblas_orgbr_ungbr<float>(rocblas_storev storev,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* Ipiv,
                              float* work,
                              rocblas_int size_w)
{
    int info;
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';
    sorgbr(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cblas_orgbr_ungbr<double>(rocblas_storev storev,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* Ipiv,
                               double* work,
                               rocblas_int size_w)
{
    int info;
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';
    dorgbr(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cblas_orgbr_ungbr<rocblas_float_complex>(rocblas_storev storev,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* Ipiv,
                                              rocblas_float_complex* work,
                                              rocblas_int size_w)
{
    int info;
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';
    cungbr(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cblas_orgbr_ungbr<rocblas_double_complex>(rocblas_storev storev,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* Ipiv,
                                               rocblas_double_complex* work,
                                               rocblas_int size_w)
{
    int info;
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';
    zungbr(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, &info);
}

// orgtr & ungtr
template <>
void cblas_orgtr_ungtr<float>(rocblas_fill uplo,
                              rocblas_int n,
                              float* A,
                              rocblas_int lda,
                              float* Ipiv,
                              float* work,
                              rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    sorgtr(&uploC, &n, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cblas_orgtr_ungtr<double>(rocblas_fill uplo,
                               rocblas_int n,
                               double* A,
                               rocblas_int lda,
                               double* Ipiv,
                               double* work,
                               rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    dorgtr(&uploC, &n, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cblas_orgtr_ungtr<rocblas_float_complex>(rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* Ipiv,
                                              rocblas_float_complex* work,
                                              rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    cungtr(&uploC, &n, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cblas_orgtr_ungtr<rocblas_double_complex>(rocblas_fill uplo,
                                               rocblas_int n,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* Ipiv,
                                               rocblas_double_complex* work,
                                               rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    zungtr(&uploC, &n, A, &lda, Ipiv, work, &size_w, &info);
}

// ormqr & unmqr
template <>
void cblas_ormqr_unmqr<float>(rocblas_side side,
                              rocblas_operation trans,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* C,
                              rocblas_int ldc,
                              float* work,
                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sormqr(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormqr_unmqr<double>(rocblas_side side,
                               rocblas_operation trans,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* C,
                               rocblas_int ldc,
                               double* work,
                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dormqr(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormqr_unmqr<rocblas_float_complex>(rocblas_side side,
                                              rocblas_operation trans,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_float_complex* work,
                                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunmqr(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormqr_unmqr<rocblas_double_complex>(rocblas_side side,
                                               rocblas_operation trans,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunmqr(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// orm2r & unm2r
template <>
void cblas_orm2r_unm2r<float>(rocblas_side side,
                              rocblas_operation trans,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* C,
                              rocblas_int ldc,
                              float* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sorm2r(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orm2r_unm2r<double>(rocblas_side side,
                               rocblas_operation trans,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* C,
                               rocblas_int ldc,
                               double* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dorm2r(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orm2r_unm2r<rocblas_float_complex>(rocblas_side side,
                                              rocblas_operation trans,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_float_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunm2r(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orm2r_unm2r<rocblas_double_complex>(rocblas_side side,
                                               rocblas_operation trans,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunm2r(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

// ormlq & unmlq
template <>
void cblas_ormlq_unmlq<float>(rocblas_side side,
                              rocblas_operation trans,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* C,
                              rocblas_int ldc,
                              float* work,
                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sormlq(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormlq_unmlq<double>(rocblas_side side,
                               rocblas_operation trans,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* C,
                               rocblas_int ldc,
                               double* work,
                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dormlq(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormlq_unmlq<rocblas_float_complex>(rocblas_side side,
                                              rocblas_operation trans,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_float_complex* work,
                                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunmlq(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormlq_unmlq<rocblas_double_complex>(rocblas_side side,
                                               rocblas_operation trans,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunmlq(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// orml2 & unml2
template <>
void cblas_orml2_unml2<float>(rocblas_side side,
                              rocblas_operation trans,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* C,
                              rocblas_int ldc,
                              float* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sorml2(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orml2_unml2<double>(rocblas_side side,
                               rocblas_operation trans,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* C,
                               rocblas_int ldc,
                               double* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dorml2(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orml2_unml2<rocblas_float_complex>(rocblas_side side,
                                              rocblas_operation trans,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_float_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunml2(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orml2_unml2<rocblas_double_complex>(rocblas_side side,
                                               rocblas_operation trans,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunml2(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

// ormql & unmql
template <>
void cblas_ormql_unmql<float>(rocblas_side side,
                              rocblas_operation trans,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* C,
                              rocblas_int ldc,
                              float* work,
                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sormql(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormql_unmql<double>(rocblas_side side,
                               rocblas_operation trans,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* C,
                               rocblas_int ldc,
                               double* work,
                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dormql(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormql_unmql<rocblas_float_complex>(rocblas_side side,
                                              rocblas_operation trans,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_float_complex* work,
                                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunmql(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormql_unmql<rocblas_double_complex>(rocblas_side side,
                                               rocblas_operation trans,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunmql(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// orm2l & unm2l
template <>
void cblas_orm2l_unm2l<float>(rocblas_side side,
                              rocblas_operation trans,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* C,
                              rocblas_int ldc,
                              float* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sorm2l(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orm2l_unm2l<double>(rocblas_side side,
                               rocblas_operation trans,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* C,
                               rocblas_int ldc,
                               double* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dorm2l(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orm2l_unm2l<rocblas_float_complex>(rocblas_side side,
                                              rocblas_operation trans,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_float_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunm2l(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cblas_orm2l_unm2l<rocblas_double_complex>(rocblas_side side,
                                               rocblas_operation trans,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunm2l(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

// ormbr & unmbr
template <>
void cblas_ormbr_unmbr<float>(rocblas_storev storev,
                              rocblas_side side,
                              rocblas_operation trans,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* C,
                              rocblas_int ldc,
                              float* work,
                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';

    sormbr(&vect, &sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormbr_unmbr<double>(rocblas_storev storev,
                               rocblas_side side,
                               rocblas_operation trans,
                               rocblas_int m,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* C,
                               rocblas_int ldc,
                               double* work,
                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';

    dormbr(&vect, &sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormbr_unmbr<rocblas_float_complex>(rocblas_storev storev,
                                              rocblas_side side,
                                              rocblas_operation trans,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_float_complex* work,
                                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';

    cunmbr(&vect, &sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormbr_unmbr<rocblas_double_complex>(rocblas_storev storev,
                                               rocblas_side side,
                                               rocblas_operation trans,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';

    zunmbr(&vect, &sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// ormtr & unmtr
template <>
void cblas_ormtr_unmtr<float>(rocblas_side side,
                              rocblas_fill uplo,
                              rocblas_operation trans,
                              rocblas_int m,
                              rocblas_int n,
                              float* A,
                              rocblas_int lda,
                              float* ipiv,
                              float* C,
                              rocblas_int ldc,
                              float* work,
                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(trans);

    sormtr(&sideC, &uploC, &transC, &m, &n, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormtr_unmtr<double>(rocblas_side side,
                               rocblas_fill uplo,
                               rocblas_operation trans,
                               rocblas_int m,
                               rocblas_int n,
                               double* A,
                               rocblas_int lda,
                               double* ipiv,
                               double* C,
                               rocblas_int ldc,
                               double* work,
                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(trans);

    dormtr(&sideC, &uploC, &transC, &m, &n, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormtr_unmtr<rocblas_float_complex>(rocblas_side side,
                                              rocblas_fill uplo,
                                              rocblas_operation trans,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* ipiv,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_float_complex* work,
                                              rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(trans);

    cunmtr(&sideC, &uploC, &transC, &m, &n, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormtr_unmtr<rocblas_double_complex>(rocblas_side side,
                                               rocblas_fill uplo,
                                               rocblas_operation trans,
                                               rocblas_int m,
                                               rocblas_int n,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* ipiv,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(trans);

    zunmtr(&sideC, &uploC, &transC, &m, &n, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// scal
/*template <>
void cblas_scal<float>(rocblas_int n, const float alpha, float *x,
                       rocblas_int incx) {
  cblas_sscal(n, alpha, x, incx);
}

template <>
void cblas_scal<double>(rocblas_int n, const double alpha, double *x,
                        rocblas_int incx) {
  cblas_dscal(n, alpha, x, incx);
}

template <>
void cblas_scal<rocblas_float_complex>(rocblas_int n,
                                       const rocblas_float_complex alpha,
                                       rocblas_float_complex *x,
                                       rocblas_int incx) {
  cblas_cscal(n, &alpha, x, incx);
}

template <>
void cblas_scal<rocblas_double_complex>(rocblas_int n,
                                        const rocblas_double_complex alpha,
                                        rocblas_double_complex *x,
                                        rocblas_int incx) {
  cblas_zscal(n, &alpha, x, incx);
}

// copy
template <>
void cblas_copy<float>(rocblas_int n, float *x, rocblas_int incx, float *y,
                       rocblas_int incy) {
  cblas_scopy(n, x, incx, y, incy);
}

template <>
void cblas_copy<double>(rocblas_int n, double *x, rocblas_int incx, double *y,
                        rocblas_int incy) {
  cblas_dcopy(n, x, incx, y, incy);
}

template <>
void cblas_copy<rocblas_float_complex>(rocblas_int n, rocblas_float_complex *x,
                                       rocblas_int incx,
                                       rocblas_float_complex *y,
                                       rocblas_int incy) {
  cblas_ccopy(n, x, incx, y, incy);
}

template <>
void cblas_copy<rocblas_double_complex>(rocblas_int n,
                                        rocblas_double_complex *x,
                                        rocblas_int incx,
                                        rocblas_double_complex *y,
                                        rocblas_int incy) {
  cblas_zcopy(n, x, incx, y, incy);
}

// axpy
template <>
void cblas_axpy<float>(rocblas_int n, float alpha, float *x, rocblas_int incx,
                       float *y, rocblas_int incy) {
  cblas_saxpy(n, alpha, x, incx, y, incy);
}

template <>
void cblas_axpy<double>(rocblas_int n, double alpha, double *x,
                        rocblas_int incx, double *y, rocblas_int incy) {
  cblas_daxpy(n, alpha, x, incx, y, incy);
}

template <>
void cblas_axpy<rocblas_float_complex>(
    rocblas_int n, rocblas_float_complex alpha, rocblas_float_complex *x,
    rocblas_int incx, rocblas_float_complex *y, rocblas_int incy) {
  cblas_caxpy(n, &alpha, x, incx, y, incy);
}

template <>
void cblas_axpy<rocblas_double_complex>(
    rocblas_int n, rocblas_double_complex alpha, rocblas_double_complex *x,
    rocblas_int incx, rocblas_double_complex *y, rocblas_int incy) {
  cblas_zaxpy(n, &alpha, x, incx, y, incy);
}

// swap
template <>
void cblas_swap<float>(rocblas_int n, float *x, rocblas_int incx, float *y,
                       rocblas_int incy) {
  cblas_sswap(n, x, incx, y, incy);
}

template <>
void cblas_swap<double>(rocblas_int n, double *x, rocblas_int incx, double *y,
                        rocblas_int incy) {
  cblas_dswap(n, x, incx, y, incy);
}

template <>
void cblas_swap<rocblas_float_complex>(rocblas_int n, rocblas_float_complex *x,
                                       rocblas_int incx,
                                       rocblas_float_complex *y,
                                       rocblas_int incy) {
  cblas_cswap(n, x, incx, y, incy);
}

template <>
void cblas_swap<rocblas_double_complex>(rocblas_int n,
                                        rocblas_double_complex *x,
                                        rocblas_int incx,
                                        rocblas_double_complex *y,
                                        rocblas_int incy) {
  cblas_zswap(n, x, incx, y, incy);
}

// dot
template <>
void cblas_dot<float>(rocblas_int n, const float *x, rocblas_int incx,
                      const float *y, rocblas_int incy, float *result) {
  *result = cblas_sdot(n, x, incx, y, incy);
}

template <>
void cblas_dot<double>(rocblas_int n, const double *x, rocblas_int incx,
                       const double *y, rocblas_int incy, double *result) {
  *result = cblas_ddot(n, x, incx, y, incy);
}

template <>
void cblas_dot<rocblas_float_complex>(rocblas_int n,
                                      const rocblas_float_complex *x,
                                      rocblas_int incx,
                                      const rocblas_float_complex *y,
                                      rocblas_int incy,
                                      rocblas_float_complex *result) {
  cblas_cdotu_sub(n, x, incx, y, incy, result);
}

template <>
void cblas_dot<rocblas_double_complex>(rocblas_int n,
                                       const rocblas_double_complex *x,
                                       rocblas_int incx,
                                       const rocblas_double_complex *y,
                                       rocblas_int incy,
                                       rocblas_double_complex *result) {
  cblas_zdotu_sub(n, x, incx, y, incy, result);
}

// nrm2
template <>
void cblas_nrm2<float, float>(rocblas_int n, const float *x, rocblas_int incx,
                              float *result) {
  *result = cblas_snrm2(n, x, incx);
}

template <>
void cblas_nrm2<double, double>(rocblas_int n, const double *x,
                                rocblas_int incx, double *result) {
  *result = cblas_dnrm2(n, x, incx);
}

template <>
void cblas_nrm2<rocblas_float_complex, float>(rocblas_int n,
                                              const rocblas_float_complex *x,
                                              rocblas_int incx, float *result) {
  *result = cblas_scnrm2(n, x, incx);
}

template <>
void cblas_nrm2<rocblas_double_complex, double>(rocblas_int n,
                                                const rocblas_double_complex *x,
                                                rocblas_int incx,
                                                double *result) {
  *result = cblas_dznrm2(n, x, incx);
}

// asum
template <>
void cblas_asum<float, float>(rocblas_int n, const float *x, rocblas_int incx,
                              float *result) {
  *result = cblas_sasum(n, x, incx);
}

template <>
void cblas_asum<double, double>(rocblas_int n, const double *x,
                                rocblas_int incx, double *result) {
  *result = cblas_dasum(n, x, incx);
}

template <>
void cblas_asum<rocblas_float_complex, float>(rocblas_int n,
                                              const rocblas_float_complex *x,
                                              rocblas_int incx, float *result) {
  *result = cblas_scasum(n, x, incx);
}

template <>
void cblas_asum<rocblas_double_complex, double>(rocblas_int n,
                                                const rocblas_double_complex *x,
                                                rocblas_int incx,
                                                double *result) {
  *result = cblas_dzasum(n, x, incx);
}

// amax
template <>
void cblas_iamax<float>(rocblas_int n, const float *x, rocblas_int incx,
                        rocblas_int *result) {
  *result = (rocblas_int)cblas_isamax(n, x, incx);
}

template <>
void cblas_iamax<double>(rocblas_int n, const double *x, rocblas_int incx,
                         rocblas_int *result) {
  *result = (rocblas_int)cblas_idamax(n, x, incx);
}

template <>
void cblas_iamax<rocblas_float_complex>(rocblas_int n,
                                        const rocblas_float_complex *x,
                                        rocblas_int incx, rocblas_int *result) {
  *result = (rocblas_int)cblas_icamax(n, x, incx);
}

template <>
void cblas_iamax<rocblas_double_complex>(rocblas_int n,
                                         const rocblas_double_complex *x,
                                         rocblas_int incx,
                                         rocblas_int *result) {
  *result = (rocblas_int)cblas_izamax(n, x, incx);
}

// gemv
template <>
void cblas_gemv<float>(rocblas_operation transA, rocblas_int m, rocblas_int n,
                       float alpha, float *A, rocblas_int lda, float *x,
                       rocblas_int incx, float beta, float *y,
                       rocblas_int incy) {
  cblas_sgemv(CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, alpha, A, lda, x,
              incx, beta, y, incy);
}

template <>
void cblas_gemv<double>(rocblas_operation transA, rocblas_int m, rocblas_int n,
                        double alpha, double *A, rocblas_int lda, double *x,
                        rocblas_int incx, double beta, double *y,
                        rocblas_int incy) {
  cblas_dgemv(CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, alpha, A, lda, x,
              incx, beta, y, incy);
}

template <>
void cblas_gemv<rocblas_float_complex>(
    rocblas_operation transA, rocblas_int m, rocblas_int n,
    rocblas_float_complex alpha, rocblas_float_complex *A, rocblas_int lda,
    rocblas_float_complex *x, rocblas_int incx, rocblas_float_complex beta,
    rocblas_float_complex *y, rocblas_int incy) {
  cblas_cgemv(CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, &alpha, A, lda, x,
              incx, &beta, y, incy);
}

template <>
void cblas_gemv<rocblas_double_complex>(
    rocblas_operation transA, rocblas_int m, rocblas_int n,
    rocblas_double_complex alpha, rocblas_double_complex *A, rocblas_int lda,
    rocblas_double_complex *x, rocblas_int incx, rocblas_double_complex beta,
    rocblas_double_complex *y, rocblas_int incy) {
  cblas_zgemv(CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, &alpha, A, lda, x,
              incx, &beta, y, incy);
}
*/
template <>
void cblas_symv_hemv<float>(rocblas_fill uplo,
                            rocblas_int n,
                            float alpha,
                            float* A,
                            rocblas_int lda,
                            float* x,
                            rocblas_int incx,
                            float beta,
                            float* y,
                            rocblas_int incy)
{
    char uploC = rocblas2char_fill(uplo);
    ssymv(&uploC, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

template <>
void cblas_symv_hemv<double>(rocblas_fill uplo,
                             rocblas_int n,
                             double alpha,
                             double* A,
                             rocblas_int lda,
                             double* x,
                             rocblas_int incx,
                             double beta,
                             double* y,
                             rocblas_int incy)
{
    char uploC = rocblas2char_fill(uplo);
    dsymv(&uploC, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

template <>
void cblas_symv_hemv<rocblas_float_complex>(rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex alpha,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* x,
                                            rocblas_int incx,
                                            rocblas_float_complex beta,
                                            rocblas_float_complex* y,
                                            rocblas_int incy)
{
    char uploC = rocblas2char_fill(uplo);
    chemv(&uploC, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

template <>
void cblas_symv_hemv<rocblas_double_complex>(rocblas_fill uplo,
                                             rocblas_int n,
                                             rocblas_double_complex alpha,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* x,
                                             rocblas_int incx,
                                             rocblas_double_complex beta,
                                             rocblas_double_complex* y,
                                             rocblas_int incy)
{
    char uploC = rocblas2char_fill(uplo);
    zhemv(&uploC, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

// symm & hemm
template <>
void cblas_symm_hemm<float>(rocblas_side side,
                            rocblas_fill uplo,
                            rocblas_int m,
                            rocblas_int n,
                            float alpha,
                            float* A,
                            rocblas_int lda,
                            float* B,
                            rocblas_int ldb,
                            float beta,
                            float* C,
                            rocblas_int ldc)
{
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    ssymm(&sideC, &uploC, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template <>
void cblas_symm_hemm<double>(rocblas_side side,
                             rocblas_fill uplo,
                             rocblas_int m,
                             rocblas_int n,
                             double alpha,
                             double* A,
                             rocblas_int lda,
                             double* B,
                             rocblas_int ldb,
                             double beta,
                             double* C,
                             rocblas_int ldc)
{
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    dsymm(&sideC, &uploC, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template <>
void cblas_symm_hemm<rocblas_float_complex>(rocblas_side side,
                                            rocblas_fill uplo,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex alpha,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* B,
                                            rocblas_int ldb,
                                            rocblas_float_complex beta,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc)
{
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    chemm(&sideC, &uploC, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template <>
void cblas_symm_hemm<rocblas_double_complex>(rocblas_side side,
                                             rocblas_fill uplo,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_double_complex alpha,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* B,
                                             rocblas_int ldb,
                                             rocblas_double_complex beta,
                                             rocblas_double_complex* C,
                                             rocblas_int ldc)
{
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    zhemm(&sideC, &uploC, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

/*
template <>
void cblas_ger<float>(rocblas_int m, rocblas_int n, float alpha, float *x,
                      rocblas_int incx, float *y, rocblas_int incy, float *A,
                      rocblas_int lda) {
  cblas_sger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
void cblas_ger<double>(rocblas_int m, rocblas_int n, double alpha, double *x,
                       rocblas_int incx, double *y, rocblas_int incy, double *A,
                       rocblas_int lda) {
  cblas_dger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
void cblas_syr<float>(rocblas_fill uplo, rocblas_int n, float alpha, float *x,
                      rocblas_int incx, float *A, rocblas_int lda) {
  cblas_ssyr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, A, lda);
}

template <>
void cblas_syr<double>(rocblas_fill uplo, rocblas_int n, double alpha,
                       double *x, rocblas_int incx, double *A,
                       rocblas_int lda) {
  cblas_dsyr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, A, lda);
}

// gemm
template <>
void cblas_gemm<rocblas_half>(rocblas_operation transA,
                              rocblas_operation transB, rocblas_int m,
                              rocblas_int n, rocblas_int k, rocblas_half alpha,
                              rocblas_half *A, rocblas_int lda, rocblas_half *B,
                              rocblas_int ldb, rocblas_half beta,
                              rocblas_half *C, rocblas_int ldc) {
  // cblas does not support rocblas_half, so convert to higher precision float
  // This will give more precise result which is acceptable for testing
  float alpha_float = half_to_float(alpha);
  float beta_float = half_to_float(beta);

  int sizeA = transA == rocblas_operation_none ? k * lda : m * lda;
  int sizeB = transB == rocblas_operation_none ? n * ldb : k * ldb;
  int sizeC = n * ldc;

  std::unique_ptr<float[]> A_float(new float[sizeA]());
  std::unique_ptr<float[]> B_float(new float[sizeB]());
  std::unique_ptr<float[]> C_float(new float[sizeC]());

  for (int i = 0; i < sizeA; i++) {
    A_float[i] = half_to_float(A[i]);
  }
  for (int i = 0; i < sizeB; i++) {
    B_float[i] = half_to_float(B[i]);
  }
  for (int i = 0; i < sizeC; i++) {
    C_float[i] = half_to_float(C[i]);
  }

  // just directly cast, since transA, transB are integers in the enum
  // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA
  // );
  cblas_sgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB,
              m, n, k, alpha_float, const_cast<const float *>(A_float.get()),
              lda, const_cast<const float *>(B_float.get()), ldb, beta_float,
              static_cast<float *>(C_float.get()), ldc);

  for (int i = 0; i < sizeC; i++) {
    C[i] = float_to_half(C_float[i]);
  }
}
*/

template <>
void cblas_gemm<float>(rocblas_operation transA,
                       rocblas_operation transB,
                       rocblas_int m,
                       rocblas_int n,
                       rocblas_int k,
                       float alpha,
                       float* A,
                       rocblas_int lda,
                       float* B,
                       rocblas_int ldb,
                       float beta,
                       float* C,
                       rocblas_int ldc)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA
    // );
    cblas_sgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB, m, n, k, alpha, A,
                lda, B, ldb, beta, C, ldc);
}

template <>
void cblas_gemm<double>(rocblas_operation transA,
                        rocblas_operation transB,
                        rocblas_int m,
                        rocblas_int n,
                        rocblas_int k,
                        double alpha,
                        double* A,
                        rocblas_int lda,
                        double* B,
                        rocblas_int ldb,
                        double beta,
                        double* C,
                        rocblas_int ldc)
{
    cblas_dgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB, m, n, k, alpha, A,
                lda, B, ldb, beta, C, ldc);
}

template <>
void cblas_gemm<rocblas_float_complex>(rocblas_operation transA,
                                       rocblas_operation transB,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_int k,
                                       rocblas_float_complex alpha,
                                       rocblas_float_complex* A,
                                       rocblas_int lda,
                                       rocblas_float_complex* B,
                                       rocblas_int ldb,
                                       rocblas_float_complex beta,
                                       rocblas_float_complex* C,
                                       rocblas_int ldc)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_cgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB, m, n, k, &alpha, A,
                lda, B, ldb, &beta, C, ldc);
}

template <>
void cblas_gemm<rocblas_double_complex>(rocblas_operation transA,
                                        rocblas_operation transB,
                                        rocblas_int m,
                                        rocblas_int n,
                                        rocblas_int k,
                                        rocblas_double_complex alpha,
                                        rocblas_double_complex* A,
                                        rocblas_int lda,
                                        rocblas_double_complex* B,
                                        rocblas_int ldb,
                                        rocblas_double_complex beta,
                                        rocblas_double_complex* C,
                                        rocblas_int ldc)
{
    cblas_zgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB, m, n, k, &alpha, A,
                lda, B, ldb, &beta, C, ldc);
}

/*
// trsm
template <>
void cblas_trsm<float>(rocblas_side side, rocblas_fill uplo,
                       rocblas_operation transA, rocblas_diagonal diag,
                       rocblas_int m, rocblas_int n, float alpha,
                       const float *A, rocblas_int lda, float *B,
                       rocblas_int ldb) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_strsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, alpha, A, lda, B,
              ldb);
}

template <>
void cblas_trsm<double>(rocblas_side side, rocblas_fill uplo,
                        rocblas_operation transA, rocblas_diagonal diag,
                        rocblas_int m, rocblas_int n, double alpha,
                        const double *A, rocblas_int lda, double *B,
                        rocblas_int ldb) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_dtrsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, alpha, A, lda, B,
              ldb);
}

template <>
void cblas_trsm<rocblas_float_complex>(
    rocblas_side side, rocblas_fill uplo, rocblas_operation transA,
    rocblas_diagonal diag, rocblas_int m, rocblas_int n,
    rocblas_float_complex alpha, const rocblas_float_complex *A,
    rocblas_int lda, rocblas_float_complex *B, rocblas_int ldb) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_ctrsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, &alpha, A, lda,
              B, ldb);
}

template <>
void cblas_trsm<rocblas_double_complex>(
    rocblas_side side, rocblas_fill uplo, rocblas_operation transA,
    rocblas_diagonal diag, rocblas_int m, rocblas_int n,
    rocblas_double_complex alpha, const rocblas_double_complex *A,
    rocblas_int lda, rocblas_double_complex *B, rocblas_int ldb) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_ztrsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, &alpha, A, lda,
              B, ldb);
}
*/

// potf2
template <>
void cblas_potf2(rocblas_fill uplo, rocblas_int n, float* A, rocblas_int lda, rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    spotf2(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potf2(rocblas_fill uplo, rocblas_int n, double* A, rocblas_int lda, rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    dpotf2(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potf2(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_float_complex* A,
                 rocblas_int lda,
                 rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    cpotf2(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potf2(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_double_complex* A,
                 rocblas_int lda,
                 rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    zpotf2(&uploC, &n, A, &lda, info);
}

// potrf
template <>
void cblas_potrf(rocblas_fill uplo, rocblas_int n, float* A, rocblas_int lda, rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    spotrf(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potrf(rocblas_fill uplo, rocblas_int n, double* A, rocblas_int lda, rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    dpotrf(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potrf(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_float_complex* A,
                 rocblas_int lda,
                 rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    cpotrf(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potrf(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_double_complex* A,
                 rocblas_int lda,
                 rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    zpotrf(&uploC, &n, A, &lda, info);
}

// potrs
template <>
void cblas_potrs(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_int nrhs,
                 float* A,
                 rocblas_int lda,
                 float* B,
                 rocblas_int ldb)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    spotrs(&uploC, &n, &nrhs, A, &lda, B, &ldb, &info);
}

template <>
void cblas_potrs(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_int nrhs,
                 double* A,
                 rocblas_int lda,
                 double* B,
                 rocblas_int ldb)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    dpotrs(&uploC, &n, &nrhs, A, &lda, B, &ldb, &info);
}

template <>
void cblas_potrs(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_int nrhs,
                 rocblas_float_complex* A,
                 rocblas_int lda,
                 rocblas_float_complex* B,
                 rocblas_int ldb)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    cpotrs(&uploC, &n, &nrhs, A, &lda, B, &ldb, &info);
}

template <>
void cblas_potrs(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_int nrhs,
                 rocblas_double_complex* A,
                 rocblas_int lda,
                 rocblas_double_complex* B,
                 rocblas_int ldb)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    zpotrs(&uploC, &n, &nrhs, A, &lda, B, &ldb, &info);
}

// posv
template <>
void cblas_posv(rocblas_fill uplo,
                rocblas_int n,
                rocblas_int nrhs,
                float* A,
                rocblas_int lda,
                float* B,
                rocblas_int ldb,
                rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    sposv(&uploC, &n, &nrhs, A, &lda, B, &ldb, info);
}

template <>
void cblas_posv(rocblas_fill uplo,
                rocblas_int n,
                rocblas_int nrhs,
                double* A,
                rocblas_int lda,
                double* B,
                rocblas_int ldb,
                rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    dposv(&uploC, &n, &nrhs, A, &lda, B, &ldb, info);
}

template <>
void cblas_posv(rocblas_fill uplo,
                rocblas_int n,
                rocblas_int nrhs,
                rocblas_float_complex* A,
                rocblas_int lda,
                rocblas_float_complex* B,
                rocblas_int ldb,
                rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    cposv(&uploC, &n, &nrhs, A, &lda, B, &ldb, info);
}

template <>
void cblas_posv(rocblas_fill uplo,
                rocblas_int n,
                rocblas_int nrhs,
                rocblas_double_complex* A,
                rocblas_int lda,
                rocblas_double_complex* B,
                rocblas_int ldb,
                rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    zposv(&uploC, &n, &nrhs, A, &lda, B, &ldb, info);
}

// potri
template <>
void cblas_potri(rocblas_fill uplo, rocblas_int n, float* A, rocblas_int lda, rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    spotri(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potri(rocblas_fill uplo, rocblas_int n, double* A, rocblas_int lda, rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    dpotri(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potri(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_float_complex* A,
                 rocblas_int lda,
                 rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    cpotri(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potri(rocblas_fill uplo,
                 rocblas_int n,
                 rocblas_double_complex* A,
                 rocblas_int lda,
                 rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    zpotri(&uploC, &n, A, &lda, info);
}

// getf2
template <>
void cblas_getf2(rocblas_int m,
                 rocblas_int n,
                 float* A,
                 rocblas_int lda,
                 rocblas_int* ipiv,
                 rocblas_int* info)
{
    sgetf2(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getf2(rocblas_int m,
                 rocblas_int n,
                 double* A,
                 rocblas_int lda,
                 rocblas_int* ipiv,
                 rocblas_int* info)
{
    dgetf2(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getf2(rocblas_int m,
                 rocblas_int n,
                 rocblas_float_complex* A,
                 rocblas_int lda,
                 rocblas_int* ipiv,
                 rocblas_int* info)
{
    cgetf2(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getf2(rocblas_int m,
                 rocblas_int n,
                 rocblas_double_complex* A,
                 rocblas_int lda,
                 rocblas_int* ipiv,
                 rocblas_int* info)
{
    zgetf2(&m, &n, A, &lda, ipiv, info);
}

/*
// trtri
template <>
rocblas_int cblas_trtri<float>(rocblas_fill uplo, rocblas_diagonal diag,
rocblas_int n, float *A, rocblas_int lda) { rocblas_int info; char uploC =
rocblas2char_fill(uplo); char diagC = rocblas2char_diagonal(diag);
  strtri(&uploC, &diagC, &n, A, &lda, &info);
  return info;
}

template <>
rocblas_int cblas_trtri<double>(rocblas_fill uplo, rocblas_diagonal diag,
rocblas_int n, double *A, rocblas_int lda) { rocblas_int info; char uploC =
rocblas2char_fill(uplo); char diagC = rocblas2char_diagonal(diag);
  dtrtri(&uploC, &diagC, &n, A, &lda, &info);
  return info;
}

template <>
rocblas_int cblas_trtri<rocblas_float_complex>(rocblas_fill uplo,
rocblas_diagonal diag, rocblas_int n, rocblas_float_complex *A, rocblas_int lda)
{ rocblas_int info; char uploC = rocblas2char_fill(uplo); char diagC =
rocblas2char_diagonal(diag); ctrtri(&uploC, &diagC, &n, A, &lda, &info); return
info;
}

template <>
rocblas_int cblas_trtri<rocblas_double_complex>(rocblas_fill uplo,
rocblas_diagonal diag, rocblas_int n, rocblas_double_complex *A, rocblas_int
lda) { rocblas_int info; char uploC = rocblas2char_fill(uplo); char diagC =
rocblas2char_diagonal(diag); ztrtri(&uploC, &diagC, &n, A, &lda, &info); return
info;
}

// trmm
template <>
void cblas_trmm<float>(rocblas_side side, rocblas_fill uplo,
                       rocblas_operation transA, rocblas_diagonal diag,
                       rocblas_int m, rocblas_int n, float alpha,
                       const float *A, rocblas_int lda, float *B,
                       rocblas_int ldb) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_strmm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, alpha, A, lda, B,
              ldb);
}

template <>
void cblas_trmm<double>(rocblas_side side, rocblas_fill uplo,
                        rocblas_operation transA, rocblas_diagonal diag,
                        rocblas_int m, rocblas_int n, double alpha,
                        const double *A, rocblas_int lda, double *B,
                        rocblas_int ldb) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_dtrmm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, alpha, A, lda, B,
              ldb);
}

template <>
void cblas_trmm<rocblas_float_complex>(
    rocblas_side side, rocblas_fill uplo, rocblas_operation transA,
    rocblas_diagonal diag, rocblas_int m, rocblas_int n,
    rocblas_float_complex alpha, const rocblas_float_complex *A,
    rocblas_int lda, rocblas_float_complex *B, rocblas_int ldb) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_ctrmm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, &alpha, A, lda,
              B, ldb);
}

template <>
void cblas_trmm<rocblas_double_complex>(
    rocblas_side side, rocblas_fill uplo, rocblas_operation transA,
    rocblas_diagonal diag, rocblas_int m, rocblas_int n,
    rocblas_double_complex alpha, const rocblas_double_complex *A,
    rocblas_int lda, rocblas_double_complex *B, rocblas_int ldb) {
  // just directly cast, since transA, transB are integers in the enum
  cblas_ztrmm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
              (CBLAS_TRANSPOSE)transA, (CBLAS_DIAG)diag, m, n, &alpha, A, lda,
              B, ldb);
}
*/

// getrf
template <>
void cblas_getrf<float>(rocblas_int m,
                        rocblas_int n,
                        float* A,
                        rocblas_int lda,
                        rocblas_int* ipiv,
                        rocblas_int* info)
{
    sgetrf(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getrf<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         rocblas_int* ipiv,
                         rocblas_int* info)
{
    dgetrf(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getrf<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_int* ipiv,
                                        rocblas_int* info)
{
    cgetrf(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getrf<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_int* ipiv,
                                         rocblas_int* info)
{
    zgetrf(&m, &n, A, &lda, ipiv, info);
}

// getrs
template <>
void cblas_getrs<float>(rocblas_operation trans,
                        rocblas_int n,
                        rocblas_int nrhs,
                        float* A,
                        rocblas_int lda,
                        rocblas_int* ipiv,
                        float* B,
                        rocblas_int ldb)
{
    rocblas_int info;
    char transC = rocblas2char_operation(trans);
    sgetrs(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}

template <>
void cblas_getrs<double>(rocblas_operation trans,
                         rocblas_int n,
                         rocblas_int nrhs,
                         double* A,
                         rocblas_int lda,
                         rocblas_int* ipiv,
                         double* B,
                         rocblas_int ldb)
{
    rocblas_int info;
    char transC = rocblas2char_operation(trans);
    dgetrs(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}

template <>
void cblas_getrs<rocblas_float_complex>(rocblas_operation trans,
                                        rocblas_int n,
                                        rocblas_int nrhs,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_int* ipiv,
                                        rocblas_float_complex* B,
                                        rocblas_int ldb)
{
    rocblas_int info;
    char transC = rocblas2char_operation(trans);
    cgetrs(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}

template <>
void cblas_getrs<rocblas_double_complex>(rocblas_operation trans,
                                         rocblas_int n,
                                         rocblas_int nrhs,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_int* ipiv,
                                         rocblas_double_complex* B,
                                         rocblas_int ldb)
{
    rocblas_int info;
    char transC = rocblas2char_operation(trans);
    zgetrs(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}

// gesv
template <>
void cblas_gesv<float>(rocblas_int n,
                       rocblas_int nrhs,
                       float* A,
                       rocblas_int lda,
                       rocblas_int* ipiv,
                       float* B,
                       rocblas_int ldb,
                       rocblas_int* info)
{
    sgesv(&n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

template <>
void cblas_gesv<double>(rocblas_int n,
                        rocblas_int nrhs,
                        double* A,
                        rocblas_int lda,
                        rocblas_int* ipiv,
                        double* B,
                        rocblas_int ldb,
                        rocblas_int* info)
{
    dgesv(&n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

template <>
void cblas_gesv<rocblas_float_complex>(rocblas_int n,
                                       rocblas_int nrhs,
                                       rocblas_float_complex* A,
                                       rocblas_int lda,
                                       rocblas_int* ipiv,
                                       rocblas_float_complex* B,
                                       rocblas_int ldb,
                                       rocblas_int* info)
{
    cgesv(&n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

template <>
void cblas_gesv<rocblas_double_complex>(rocblas_int n,
                                        rocblas_int nrhs,
                                        rocblas_double_complex* A,
                                        rocblas_int lda,
                                        rocblas_int* ipiv,
                                        rocblas_double_complex* B,
                                        rocblas_int ldb,
                                        rocblas_int* info)
{
    zgesv(&n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

// gels
template <>
void cblas_gels<float>(rocblas_operation transR,
                       rocblas_int m,
                       rocblas_int n,
                       rocblas_int nrhs,
                       float* A,
                       rocblas_int lda,
                       float* B,
                       rocblas_int ldb,
                       float* work,
                       rocblas_int lwork,
                       rocblas_int* info)
{
    char trans = rocblas2char_operation(transR);
    sgels(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, info);
}

template <>
void cblas_gels<double>(rocblas_operation transR,
                        rocblas_int m,
                        rocblas_int n,
                        rocblas_int nrhs,
                        double* A,
                        rocblas_int lda,
                        double* B,
                        rocblas_int ldb,
                        double* work,
                        rocblas_int lwork,
                        rocblas_int* info)
{
    char trans = rocblas2char_operation(transR);
    dgels(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, info);
}

template <>
void cblas_gels<rocblas_float_complex>(rocblas_operation transR,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_int nrhs,
                                       rocblas_float_complex* A,
                                       rocblas_int lda,
                                       rocblas_float_complex* B,
                                       rocblas_int ldb,
                                       rocblas_float_complex* work,
                                       rocblas_int lwork,
                                       rocblas_int* info)
{
    char trans = rocblas2char_operation(transR);
    cgels(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, info);
}

template <>
void cblas_gels<rocblas_double_complex>(rocblas_operation transR,
                                        rocblas_int m,
                                        rocblas_int n,
                                        rocblas_int nrhs,
                                        rocblas_double_complex* A,
                                        rocblas_int lda,
                                        rocblas_double_complex* B,
                                        rocblas_int ldb,
                                        rocblas_double_complex* work,
                                        rocblas_int lwork,
                                        rocblas_int* info)
{
    char trans = rocblas2char_operation(transR);
    zgels(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, info);
}

// trtri
template <>
void cblas_trtri<float>(rocblas_fill uplo,
                        rocblas_diagonal diag,
                        rocblas_int n,
                        float* A,
                        rocblas_int lda,
                        rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    char diagC = rocblas2char_diagonal(diag);
    strtri(&uploC, &diagC, &n, A, &lda, info);
}

template <>
void cblas_trtri<double>(rocblas_fill uplo,
                         rocblas_diagonal diag,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    char diagC = rocblas2char_diagonal(diag);
    dtrtri(&uploC, &diagC, &n, A, &lda, info);
}

template <>
void cblas_trtri<rocblas_float_complex>(rocblas_fill uplo,
                                        rocblas_diagonal diag,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    char diagC = rocblas2char_diagonal(diag);
    ctrtri(&uploC, &diagC, &n, A, &lda, info);
}

template <>
void cblas_trtri<rocblas_double_complex>(rocblas_fill uplo,
                                         rocblas_diagonal diag,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    char diagC = rocblas2char_diagonal(diag);
    ztrtri(&uploC, &diagC, &n, A, &lda, info);
}

// getri
template <>
void cblas_getri<float>(rocblas_int n,
                        float* A,
                        rocblas_int lda,
                        rocblas_int* ipiv,
                        float* work,
                        rocblas_int lwork,
                        rocblas_int* info)
{
    sgetri(&n, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cblas_getri<double>(rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         rocblas_int* ipiv,
                         double* work,
                         rocblas_int lwork,
                         rocblas_int* info)
{
    dgetri(&n, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cblas_getri<rocblas_float_complex>(rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_int* ipiv,
                                        rocblas_float_complex* work,
                                        rocblas_int lwork,
                                        rocblas_int* info)
{
    cgetri(&n, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cblas_getri<rocblas_double_complex>(rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_int* ipiv,
                                         rocblas_double_complex* work,
                                         rocblas_int lwork,
                                         rocblas_int* info)
{
    zgetri(&n, A, &lda, ipiv, work, &lwork, info);
}

// geqrf
template <>
void cblas_geqrf<float>(rocblas_int m,
                        rocblas_int n,
                        float* A,
                        rocblas_int lda,
                        float* ipiv,
                        float* work,
                        rocblas_int lwork)
{
    int info;
    sgeqrf(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqrf<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         double* ipiv,
                         double* work,
                         rocblas_int lwork)
{
    int info;
    dgeqrf(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqrf<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        rocblas_float_complex* work,
                                        rocblas_int lwork)
{
    int info;
    cgeqrf(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqrf<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* ipiv,
                                         rocblas_double_complex* work,
                                         rocblas_int lwork)
{
    int info;
    zgeqrf(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

// geqr2
template <>
void cblas_geqr2<float>(rocblas_int m, rocblas_int n, float* A, rocblas_int lda, float* ipiv, float* work)
{
    int info;
    sgeqr2(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_geqr2<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         double* ipiv,
                         double* work)
{
    int info;
    dgeqr2(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_geqr2<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        rocblas_float_complex* work)
{
    int info;
    cgeqr2(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_geqr2<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* ipiv,
                                         rocblas_double_complex* work)
{
    int info;
    zgeqr2(&m, &n, A, &lda, ipiv, work, &info);
}

// gerqf
template <>
void cblas_gerqf<float>(rocblas_int m,
                        rocblas_int n,
                        float* A,
                        rocblas_int lda,
                        float* ipiv,
                        float* work,
                        rocblas_int lwork)
{
    int info;
    sgerqf(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_gerqf<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         double* ipiv,
                         double* work,
                         rocblas_int lwork)
{
    int info;
    dgerqf(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_gerqf<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        rocblas_float_complex* work,
                                        rocblas_int lwork)
{
    int info;
    cgerqf(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_gerqf<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* ipiv,
                                         rocblas_double_complex* work,
                                         rocblas_int lwork)
{
    int info;
    zgerqf(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

// gerq2
template <>
void cblas_gerq2<float>(rocblas_int m, rocblas_int n, float* A, rocblas_int lda, float* ipiv, float* work)
{
    int info;
    sgerq2(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_gerq2<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         double* ipiv,
                         double* work)
{
    int info;
    dgerq2(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_gerq2<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        rocblas_float_complex* work)
{
    int info;
    cgerq2(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_gerq2<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* ipiv,
                                         rocblas_double_complex* work)
{
    int info;
    zgerq2(&m, &n, A, &lda, ipiv, work, &info);
}

// geqlf
template <>
void cblas_geqlf<float>(rocblas_int m,
                        rocblas_int n,
                        float* A,
                        rocblas_int lda,
                        float* ipiv,
                        float* work,
                        rocblas_int lwork)
{
    int info;
    sgeqlf(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqlf<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         double* ipiv,
                         double* work,
                         rocblas_int lwork)
{
    int info;
    dgeqlf(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqlf<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        rocblas_float_complex* work,
                                        rocblas_int lwork)
{
    int info;
    cgeqlf(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqlf<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* ipiv,
                                         rocblas_double_complex* work,
                                         rocblas_int lwork)
{
    int info;
    zgeqlf(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

// geql2
template <>
void cblas_geql2<float>(rocblas_int m, rocblas_int n, float* A, rocblas_int lda, float* ipiv, float* work)
{
    int info;
    sgeql2(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_geql2<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         double* ipiv,
                         double* work)
{
    int info;
    dgeql2(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_geql2<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        rocblas_float_complex* work)
{
    int info;
    cgeql2(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_geql2<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* ipiv,
                                         rocblas_double_complex* work)
{
    int info;
    zgeql2(&m, &n, A, &lda, ipiv, work, &info);
}

// gelqf
template <>
void cblas_gelqf<float>(rocblas_int m,
                        rocblas_int n,
                        float* A,
                        rocblas_int lda,
                        float* ipiv,
                        float* work,
                        rocblas_int lwork)
{
    int info;
    sgelqf(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_gelqf<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         double* ipiv,
                         double* work,
                         rocblas_int lwork)
{
    int info;
    dgelqf(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_gelqf<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        rocblas_float_complex* work,
                                        rocblas_int lwork)
{
    int info;
    cgelqf(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_gelqf<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* ipiv,
                                         rocblas_double_complex* work,
                                         rocblas_int lwork)
{
    int info;
    zgelqf(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

// gelq2
template <>
void cblas_gelq2<float>(rocblas_int m, rocblas_int n, float* A, rocblas_int lda, float* ipiv, float* work)
{
    int info;
    sgelq2(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_gelq2<double>(rocblas_int m,
                         rocblas_int n,
                         double* A,
                         rocblas_int lda,
                         double* ipiv,
                         double* work)
{
    int info;
    dgelq2(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_gelq2<rocblas_float_complex>(rocblas_int m,
                                        rocblas_int n,
                                        rocblas_float_complex* A,
                                        rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        rocblas_float_complex* work)
{
    int info;
    cgelq2(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cblas_gelq2<rocblas_double_complex>(rocblas_int m,
                                         rocblas_int n,
                                         rocblas_double_complex* A,
                                         rocblas_int lda,
                                         rocblas_double_complex* ipiv,
                                         rocblas_double_complex* work)
{
    int info;
    zgelq2(&m, &n, A, &lda, ipiv, work, &info);
}

// gebd2
template <>
void cblas_gebd2<float, float>(rocblas_int m,
                               rocblas_int n,
                               float* A,
                               rocblas_int lda,
                               float* D,
                               float* E,
                               float* tauq,
                               float* taup,
                               float* work)
{
    int info;
    sgebd2(&m, &n, A, &lda, D, E, tauq, taup, work, &info);
}

template <>
void cblas_gebd2<double, double>(rocblas_int m,
                                 rocblas_int n,
                                 double* A,
                                 rocblas_int lda,
                                 double* D,
                                 double* E,
                                 double* tauq,
                                 double* taup,
                                 double* work)
{
    int info;
    dgebd2(&m, &n, A, &lda, D, E, tauq, taup, work, &info);
}

template <>
void cblas_gebd2<rocblas_float_complex, float>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_float_complex* A,
                                               rocblas_int lda,
                                               float* D,
                                               float* E,
                                               rocblas_float_complex* tauq,
                                               rocblas_float_complex* taup,
                                               rocblas_float_complex* work)
{
    int info;
    cgebd2(&m, &n, A, &lda, D, E, tauq, taup, work, &info);
}

template <>
void cblas_gebd2<rocblas_double_complex, double>(rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 rocblas_int lda,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* tauq,
                                                 rocblas_double_complex* taup,
                                                 rocblas_double_complex* work)
{
    int info;
    zgebd2(&m, &n, A, &lda, D, E, tauq, taup, work, &info);
}

// gebrd
template <>
void cblas_gebrd<float, float>(rocblas_int m,
                               rocblas_int n,
                               float* A,
                               rocblas_int lda,
                               float* D,
                               float* E,
                               float* tauq,
                               float* taup,
                               float* work,
                               rocblas_int size_w)
{
    int info;
    sgebrd(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, &info);
}

template <>
void cblas_gebrd<double, double>(rocblas_int m,
                                 rocblas_int n,
                                 double* A,
                                 rocblas_int lda,
                                 double* D,
                                 double* E,
                                 double* tauq,
                                 double* taup,
                                 double* work,
                                 rocblas_int size_w)
{
    int info;
    dgebrd(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, &info);
}

template <>
void cblas_gebrd<rocblas_float_complex, float>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_float_complex* A,
                                               rocblas_int lda,
                                               float* D,
                                               float* E,
                                               rocblas_float_complex* tauq,
                                               rocblas_float_complex* taup,
                                               rocblas_float_complex* work,
                                               rocblas_int size_w)
{
    int info;
    cgebrd(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, &info);
}

template <>
void cblas_gebrd<rocblas_double_complex, double>(rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 rocblas_int lda,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* tauq,
                                                 rocblas_double_complex* taup,
                                                 rocblas_double_complex* work,
                                                 rocblas_int size_w)
{
    int info;
    zgebrd(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, &info);
}

// sytrd & hetrd
template <>
void cblas_sytrd_hetrd<float, float>(rocblas_fill uplo,
                                     rocblas_int n,
                                     float* A,
                                     rocblas_int lda,
                                     float* D,
                                     float* E,
                                     float* tau,
                                     float* work,
                                     rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    ssytrd(&uploC, &n, A, &lda, D, E, tau, work, &size_w, &info);
}

template <>
void cblas_sytrd_hetrd<double, double>(rocblas_fill uplo,
                                       rocblas_int n,
                                       double* A,
                                       rocblas_int lda,
                                       double* D,
                                       double* E,
                                       double* tau,
                                       double* work,
                                       rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    dsytrd(&uploC, &n, A, &lda, D, E, tau, work, &size_w, &info);
}

template <>
void cblas_sytrd_hetrd<rocblas_float_complex, float>(rocblas_fill uplo,
                                                     rocblas_int n,
                                                     rocblas_float_complex* A,
                                                     rocblas_int lda,
                                                     float* D,
                                                     float* E,
                                                     rocblas_float_complex* tau,
                                                     rocblas_float_complex* work,
                                                     rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    chetrd(&uploC, &n, A, &lda, D, E, tau, work, &size_w, &info);
}

template <>
void cblas_sytrd_hetrd<rocblas_double_complex, double>(rocblas_fill uplo,
                                                       rocblas_int n,
                                                       rocblas_double_complex* A,
                                                       rocblas_int lda,
                                                       double* D,
                                                       double* E,
                                                       rocblas_double_complex* tau,
                                                       rocblas_double_complex* work,
                                                       rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    zhetrd(&uploC, &n, A, &lda, D, E, tau, work, &size_w, &info);
}

// sytd2 & hetd2
template <>
void cblas_sytd2_hetd2<float, float>(rocblas_fill uplo,
                                     rocblas_int n,
                                     float* A,
                                     rocblas_int lda,
                                     float* D,
                                     float* E,
                                     float* tau)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    ssytd2(&uploC, &n, A, &lda, D, E, tau, &info);
}

template <>
void cblas_sytd2_hetd2<double, double>(rocblas_fill uplo,
                                       rocblas_int n,
                                       double* A,
                                       rocblas_int lda,
                                       double* D,
                                       double* E,
                                       double* tau)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    dsytd2(&uploC, &n, A, &lda, D, E, tau, &info);
}

template <>
void cblas_sytd2_hetd2<rocblas_float_complex, float>(rocblas_fill uplo,
                                                     rocblas_int n,
                                                     rocblas_float_complex* A,
                                                     rocblas_int lda,
                                                     float* D,
                                                     float* E,
                                                     rocblas_float_complex* tau)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    chetd2(&uploC, &n, A, &lda, D, E, tau, &info);
}

template <>
void cblas_sytd2_hetd2<rocblas_double_complex, double>(rocblas_fill uplo,
                                                       rocblas_int n,
                                                       rocblas_double_complex* A,
                                                       rocblas_int lda,
                                                       double* D,
                                                       double* E,
                                                       rocblas_double_complex* tau)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    zhetd2(&uploC, &n, A, &lda, D, E, tau, &info);
}

// sterf
template <>
void cblas_sterf<float>(rocblas_int n, float* D, float* E)
{
    int info;
    ssterf(&n, D, E, &info);
}

template <>
void cblas_sterf<double>(rocblas_int n, double* D, double* E)
{
    int info;
    dsterf(&n, D, E, &info);
}

// steqr
template <>
void cblas_steqr<float, float>(rocblas_evect evect,
                               rocblas_int n,
                               float* D,
                               float* E,
                               float* C,
                               rocblas_int ldc,
                               float* work,
                               rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    ssteqr(&evectC, &n, D, E, C, &ldc, work, info);
}

template <>
void cblas_steqr<double, double>(rocblas_evect evect,
                                 rocblas_int n,
                                 double* D,
                                 double* E,
                                 double* C,
                                 rocblas_int ldc,
                                 double* work,
                                 rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    dsteqr(&evectC, &n, D, E, C, &ldc, work, info);
}
template <>
void cblas_steqr<rocblas_float_complex, float>(rocblas_evect evect,
                                               rocblas_int n,
                                               float* D,
                                               float* E,
                                               rocblas_float_complex* C,
                                               rocblas_int ldc,
                                               float* work,
                                               rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    csteqr(&evectC, &n, D, E, C, &ldc, work, info);
}

template <>
void cblas_steqr<rocblas_double_complex, double>(rocblas_evect evect,
                                                 rocblas_int n,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* C,
                                                 rocblas_int ldc,
                                                 double* work,
                                                 rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    zsteqr(&evectC, &n, D, E, C, &ldc, work, info);
}

// stedc
template <>
void cblas_stedc<float, float>(rocblas_evect evect,
                               rocblas_int n,
                               float* D,
                               float* E,
                               float* C,
                               rocblas_int ldc,
                               float* work,
                               rocblas_int lwork,
                               float* rwork,
                               rocblas_int lrwork,
                               rocblas_int* iwork,
                               rocblas_int liwork,
                               rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    sstedc(&evectC, &n, D, E, C, &ldc, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cblas_stedc<double, double>(rocblas_evect evect,
                                 rocblas_int n,
                                 double* D,
                                 double* E,
                                 double* C,
                                 rocblas_int ldc,
                                 double* work,
                                 rocblas_int lwork,
                                 double* rwork,
                                 rocblas_int lrwork,
                                 rocblas_int* iwork,
                                 rocblas_int liwork,
                                 rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    dstedc(&evectC, &n, D, E, C, &ldc, rwork, &lrwork, iwork, &liwork, info);
}
template <>
void cblas_stedc<rocblas_float_complex, float>(rocblas_evect evect,
                                               rocblas_int n,
                                               float* D,
                                               float* E,
                                               rocblas_float_complex* C,
                                               rocblas_int ldc,
                                               rocblas_float_complex* work,
                                               rocblas_int lwork,
                                               float* rwork,
                                               rocblas_int lrwork,
                                               rocblas_int* iwork,
                                               rocblas_int liwork,
                                               rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    cstedc(&evectC, &n, D, E, C, &ldc, work, &lwork, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cblas_stedc<rocblas_double_complex, double>(rocblas_evect evect,
                                                 rocblas_int n,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* C,
                                                 rocblas_int ldc,
                                                 rocblas_double_complex* work,
                                                 rocblas_int lwork,
                                                 double* rwork,
                                                 rocblas_int lrwork,
                                                 rocblas_int* iwork,
                                                 rocblas_int liwork,
                                                 rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    zstedc(&evectC, &n, D, E, C, &ldc, work, &lwork, rwork, &lrwork, iwork, &liwork, info);
}

// sygs2 & hegs2
template <>
void cblas_sygs2_hegs2<float>(rocblas_eform itype,
                              rocblas_fill uplo,
                              rocblas_int n,
                              float* A,
                              rocblas_int lda,
                              float* B,
                              rocblas_int ldb)
{
    rocblas_int info;
    int itypeI = rocblas2char_eform(itype) - '0';
    char uploC = rocblas2char_fill(uplo);
    ssygs2(&itypeI, &uploC, &n, A, &lda, B, &ldb, &info);
}

template <>
void cblas_sygs2_hegs2<double>(rocblas_eform itype,
                               rocblas_fill uplo,
                               rocblas_int n,
                               double* A,
                               rocblas_int lda,
                               double* B,
                               rocblas_int ldb)
{
    rocblas_int info;
    int itypeI = rocblas2char_eform(itype) - '0';
    char uploC = rocblas2char_fill(uplo);
    dsygs2(&itypeI, &uploC, &n, A, &lda, B, &ldb, &info);
}

template <>
void cblas_sygs2_hegs2<rocblas_float_complex>(rocblas_eform itype,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* B,
                                              rocblas_int ldb)
{
    rocblas_int info;
    int itypeI = rocblas2char_eform(itype) - '0';
    char uploC = rocblas2char_fill(uplo);
    chegs2(&itypeI, &uploC, &n, A, &lda, B, &ldb, &info);
}

template <>
void cblas_sygs2_hegs2<rocblas_double_complex>(rocblas_eform itype,
                                               rocblas_fill uplo,
                                               rocblas_int n,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* B,
                                               rocblas_int ldb)
{
    rocblas_int info;
    int itypeI = rocblas2char_eform(itype) - '0';
    char uploC = rocblas2char_fill(uplo);
    zhegs2(&itypeI, &uploC, &n, A, &lda, B, &ldb, &info);
}

// sygst & hegst
template <>
void cblas_sygst_hegst<float>(rocblas_eform itype,
                              rocblas_fill uplo,
                              rocblas_int n,
                              float* A,
                              rocblas_int lda,
                              float* B,
                              rocblas_int ldb)
{
    rocblas_int info;
    int itypeI = rocblas2char_eform(itype) - '0';
    char uploC = rocblas2char_fill(uplo);
    ssygst(&itypeI, &uploC, &n, A, &lda, B, &ldb, &info);
}

template <>
void cblas_sygst_hegst<double>(rocblas_eform itype,
                               rocblas_fill uplo,
                               rocblas_int n,
                               double* A,
                               rocblas_int lda,
                               double* B,
                               rocblas_int ldb)
{
    rocblas_int info;
    int itypeI = rocblas2char_eform(itype) - '0';
    char uploC = rocblas2char_fill(uplo);
    dsygst(&itypeI, &uploC, &n, A, &lda, B, &ldb, &info);
}

template <>
void cblas_sygst_hegst<rocblas_float_complex>(rocblas_eform itype,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_float_complex* B,
                                              rocblas_int ldb)
{
    rocblas_int info;
    int itypeI = rocblas2char_eform(itype) - '0';
    char uploC = rocblas2char_fill(uplo);
    chegst(&itypeI, &uploC, &n, A, &lda, B, &ldb, &info);
}

template <>
void cblas_sygst_hegst<rocblas_double_complex>(rocblas_eform itype,
                                               rocblas_fill uplo,
                                               rocblas_int n,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               rocblas_double_complex* B,
                                               rocblas_int ldb)
{
    rocblas_int info;
    int itypeI = rocblas2char_eform(itype) - '0';
    char uploC = rocblas2char_fill(uplo);
    zhegst(&itypeI, &uploC, &n, A, &lda, B, &ldb, &info);
}

// syev & heev
template <>
void cblas_syev_heev<float, float>(rocblas_evect evect,
                                   rocblas_fill uplo,
                                   rocblas_int n,
                                   float* A,
                                   rocblas_int lda,
                                   float* D,
                                   float* work,
                                   rocblas_int lwork,
                                   float* rwork,
                                   rocblas_int lrwork,
                                   rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    ssyev(&evectC, &uploC, &n, A, &lda, D, rwork, &lrwork, info);
}

template <>
void cblas_syev_heev<double, double>(rocblas_evect evect,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     double* A,
                                     rocblas_int lda,
                                     double* D,
                                     double* work,
                                     rocblas_int lwork,
                                     double* rwork,
                                     rocblas_int lrwork,
                                     rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    dsyev(&evectC, &uploC, &n, A, &lda, D, rwork, &lrwork, info);
}

template <>
void cblas_syev_heev<rocblas_float_complex, float>(rocblas_evect evect,
                                                   rocblas_fill uplo,
                                                   rocblas_int n,
                                                   rocblas_float_complex* A,
                                                   rocblas_int lda,
                                                   float* D,
                                                   rocblas_float_complex* work,
                                                   rocblas_int lwork,
                                                   float* rwork,
                                                   rocblas_int lrwork,
                                                   rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    cheev(&evectC, &uploC, &n, A, &lda, D, work, &lwork, rwork, info);
}

template <>
void cblas_syev_heev<rocblas_double_complex, double>(rocblas_evect evect,
                                                     rocblas_fill uplo,
                                                     rocblas_int n,
                                                     rocblas_double_complex* A,
                                                     rocblas_int lda,
                                                     double* D,
                                                     rocblas_double_complex* work,
                                                     rocblas_int lwork,
                                                     double* rwork,
                                                     rocblas_int lrwork,
                                                     rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    zheev(&evectC, &uploC, &n, A, &lda, D, work, &lwork, rwork, info);
}

// syevd & heevd
template <>
void cblas_syevd_heevd<float, float>(rocblas_evect evect,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     float* A,
                                     rocblas_int lda,
                                     float* D,
                                     float* work,
                                     rocblas_int lwork,
                                     float* rwork,
                                     rocblas_int lrwork,
                                     rocblas_int* iwork,
                                     rocblas_int liwork,
                                     rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    ssyevd(&evectC, &uploC, &n, A, &lda, D, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cblas_syevd_heevd<double, double>(rocblas_evect evect,
                                       rocblas_fill uplo,
                                       rocblas_int n,
                                       double* A,
                                       rocblas_int lda,
                                       double* D,
                                       double* work,
                                       rocblas_int lwork,
                                       double* rwork,
                                       rocblas_int lrwork,
                                       rocblas_int* iwork,
                                       rocblas_int liwork,
                                       rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    dsyevd(&evectC, &uploC, &n, A, &lda, D, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cblas_syevd_heevd<rocblas_float_complex, float>(rocblas_evect evect,
                                                     rocblas_fill uplo,
                                                     rocblas_int n,
                                                     rocblas_float_complex* A,
                                                     rocblas_int lda,
                                                     float* D,
                                                     rocblas_float_complex* work,
                                                     rocblas_int lwork,
                                                     float* rwork,
                                                     rocblas_int lrwork,
                                                     rocblas_int* iwork,
                                                     rocblas_int liwork,
                                                     rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    cheevd(&evectC, &uploC, &n, A, &lda, D, work, &lwork, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cblas_syevd_heevd<rocblas_double_complex, double>(rocblas_evect evect,
                                                       rocblas_fill uplo,
                                                       rocblas_int n,
                                                       rocblas_double_complex* A,
                                                       rocblas_int lda,
                                                       double* D,
                                                       rocblas_double_complex* work,
                                                       rocblas_int lwork,
                                                       double* rwork,
                                                       rocblas_int lrwork,
                                                       rocblas_int* iwork,
                                                       rocblas_int liwork,
                                                       rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    zheevd(&evectC, &uploC, &n, A, &lda, D, work, &lwork, rwork, &lrwork, iwork, &liwork, info);
}

// sygv & hegv
template <>
void cblas_sygv_hegv<float, float>(rocblas_eform itype,
                                   rocblas_evect evect,
                                   rocblas_fill uplo,
                                   rocblas_int n,
                                   float* A,
                                   rocblas_int lda,
                                   float* B,
                                   rocblas_int ldb,
                                   float* W,
                                   float* work,
                                   rocblas_int lwork,
                                   float* rwork,
                                   rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    ssygv(&itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, work, &lwork, info);
}

template <>
void cblas_sygv_hegv<double, double>(rocblas_eform itype,
                                     rocblas_evect evect,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     double* A,
                                     rocblas_int lda,
                                     double* B,
                                     rocblas_int ldb,
                                     double* W,
                                     double* work,
                                     rocblas_int lwork,
                                     double* rwork,
                                     rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    dsygv(&itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, work, &lwork, info);
}

template <>
void cblas_sygv_hegv<rocblas_float_complex, float>(rocblas_eform itype,
                                                   rocblas_evect evect,
                                                   rocblas_fill uplo,
                                                   rocblas_int n,
                                                   rocblas_float_complex* A,
                                                   rocblas_int lda,
                                                   rocblas_float_complex* B,
                                                   rocblas_int ldb,
                                                   float* W,
                                                   rocblas_float_complex* work,
                                                   rocblas_int lwork,
                                                   float* rwork,
                                                   rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    chegv(&itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, work, &lwork, rwork, info);
}

template <>
void cblas_sygv_hegv<rocblas_double_complex, double>(rocblas_eform itype,
                                                     rocblas_evect evect,
                                                     rocblas_fill uplo,
                                                     rocblas_int n,
                                                     rocblas_double_complex* A,
                                                     rocblas_int lda,
                                                     rocblas_double_complex* B,
                                                     rocblas_int ldb,
                                                     double* W,
                                                     rocblas_double_complex* work,
                                                     rocblas_int lwork,
                                                     double* rwork,
                                                     rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    zhegv(&itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, work, &lwork, rwork, info);
}

// sygvd & hegvd
template <>
void cblas_sygvd_hegvd<float, float>(rocblas_eform itype,
                                     rocblas_evect evect,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     float* A,
                                     rocblas_int lda,
                                     float* B,
                                     rocblas_int ldb,
                                     float* W,
                                     float* work,
                                     rocblas_int lwork,
                                     float* rwork,
                                     rocblas_int lrwork,
                                     rocblas_int* iwork,
                                     rocblas_int liwork,
                                     rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    ssygvd(&itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cblas_sygvd_hegvd<double, double>(rocblas_eform itype,
                                       rocblas_evect evect,
                                       rocblas_fill uplo,
                                       rocblas_int n,
                                       double* A,
                                       rocblas_int lda,
                                       double* B,
                                       rocblas_int ldb,
                                       double* W,
                                       double* work,
                                       rocblas_int lwork,
                                       double* rwork,
                                       rocblas_int lrwork,
                                       rocblas_int* iwork,
                                       rocblas_int liwork,
                                       rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    dsygvd(&itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cblas_sygvd_hegvd<rocblas_float_complex, float>(rocblas_eform itype,
                                                     rocblas_evect evect,
                                                     rocblas_fill uplo,
                                                     rocblas_int n,
                                                     rocblas_float_complex* A,
                                                     rocblas_int lda,
                                                     rocblas_float_complex* B,
                                                     rocblas_int ldb,
                                                     float* W,
                                                     rocblas_float_complex* work,
                                                     rocblas_int lwork,
                                                     float* rwork,
                                                     rocblas_int lrwork,
                                                     rocblas_int* iwork,
                                                     rocblas_int liwork,
                                                     rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    chegvd(&itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, work, &lwork, rwork, &lrwork, iwork,
           &liwork, info);
}

template <>
void cblas_sygvd_hegvd<rocblas_double_complex, double>(rocblas_eform itype,
                                                       rocblas_evect evect,
                                                       rocblas_fill uplo,
                                                       rocblas_int n,
                                                       rocblas_double_complex* A,
                                                       rocblas_int lda,
                                                       rocblas_double_complex* B,
                                                       rocblas_int ldb,
                                                       double* W,
                                                       rocblas_double_complex* work,
                                                       rocblas_int lwork,
                                                       double* rwork,
                                                       rocblas_int lrwork,
                                                       rocblas_int* iwork,
                                                       rocblas_int liwork,
                                                       rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    zhegvd(&itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, work, &lwork, rwork, &lrwork, iwork,
           &liwork, info);
}
