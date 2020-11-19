/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************/

#include <assert.h>
#include <stdint.h>

#include "helpers.hpp"

#define IOTA_MAX_THDS 32

// Fills the given range with sequentially increasing values.
// The name and interface is based on std::iota
template <typename T>
__global__ void __launch_bounds__(IOTA_MAX_THDS) iota_n(T* first, uint32_t count, T value)
{
    const auto idx = hipThreadIdx_x;
    if (idx < count)
    {
        first[idx] = T(idx) + value;
    }
}

template <typename T>
void init_scalars_impl(T* scalars, hipStream_t stream)
{
    assert(scalars != nullptr);
    hipLaunchKernelGGL(iota_n<T>, dim3(1), dim3(IOTA_MAX_THDS), 0, stream, scalars, 3, -1);
}

void init_scalars(float* scalars, hipStream_t stream)
{
    init_scalars_impl(scalars, stream);
}

void init_scalars(double* scalars, hipStream_t stream)
{
    init_scalars_impl(scalars, stream);
}

void init_scalars(rocblas_float_complex* scalars, hipStream_t stream)
{
    init_scalars_impl(scalars, stream);
}

void init_scalars(rocblas_double_complex* scalars, hipStream_t stream)
{
    init_scalars_impl(scalars, stream);
}
