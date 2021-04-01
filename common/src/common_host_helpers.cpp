/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <chrono>

#include "common_host_helpers.hpp"

/***********************************************************************
 * timing functions                                                    *
 ***********************************************************************/

namespace ROCSOLVER_COMMON_NAMESPACE
{

/*! \brief  CPU Timer(in microsecond): no GPU synchronization */
double get_time_us_no_sync()
{
    namespace sc = std::chrono;
    const sc::steady_clock::time_point t = sc::steady_clock::now();
    return double(sc::duration_cast<sc::microseconds>(t.time_since_epoch()).count());
}

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and
 * return wall time */
double get_time_us()
{
    hipDeviceSynchronize();
    return get_time_us_no_sync();
}

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and
 * return wall time */
double get_time_us_sync(hipStream_t stream)
{
    hipStreamSynchronize(stream);
    return get_time_us_no_sync();
}

}
