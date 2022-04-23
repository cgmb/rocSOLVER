# ########################################################################
# Copyright (c) 2016-2022 Advanced Micro Devices, Inc.
# ########################################################################

include(CheckLanguage)

if(UNIX AND NOT BUILD_ADDRESS_SANITIZER)
  set(BUILD_FORTRAN_SAMPLES ON)
else()
  set(BUILD_FORTRAN_SAMPLES OFF)
endif()

if(BUILD_FORTRAN_SAMPLES)
  # the interface to rocblas for Fortran programs
  add_library(rocblas_module OBJECT
    "${ROCBLAS_INCLUDE_DIR}/rocblas_module.f90"
  )
endif()

set(BUILD_HIP_SAMPLES OFF)
if(CMAKE_VERSION VERSION_GREATER "3.21.2")
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    set(BUILD_HIP_SAMPLES ON)
  endif()
endif()

# declare sample programs
add_executable(example-c-basic
  example_basic.c
)
add_executable(example-cpp-basic
  example_basic.cpp
)
if(BUILD_FORTRAN_SAMPLES)
  add_executable(example-fortran-basic
    example_basic.f90
    $<TARGET_OBJECTS:rocblas_module>
  )
endif()
if(BUILD_HIP_SAMPLES)
  add_executable(example-hip-basic
    example_basic.hip
  )
endif()
add_executable(example-c-hmm
  example_hmm.c
)
add_executable(example-cpp-logging
  example_logging.cpp
)
add_executable(example-c-batched
  example_batched.c
)
add_executable(example-c-strided-batched
  example_strided_batched.c
)

# group sample programs by language
set(c_samples
  example-c-basic
  example-c-hmm
  example-c-batched
  example-c-strided-batched
)
set(cpp_samples
  example-cpp-basic
  example-cpp-logging
)
if(BUILD_FORTRAN_SAMPLES)
  set(fortran_samples
    example-fortran-basic
  )
endif()
if(BUILD_HIP_SAMPLES)
  set(hip_samples
    example-hip-basic
  )
endif()

# set flags for building the sample programs
foreach(exe ${c_samples} ${cpp_samples} ${fortran_samples})
  target_link_libraries(${exe} PRIVATE roc::rocsolver)
  rocm_install(TARGETS ${exe} COMPONENT samples)
endforeach()

foreach(exe ${cpp_samples})
  set_target_properties(${exe} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
endforeach()

foreach(exe ${c_samples})
  set_target_properties(${exe} PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
  )
endforeach()

foreach(exe ${hip_samples})
  set_target_properties(${exe} PROPERTIES
    HIP_STANDARD 11
    HIP_STANDARD_REQUIRED ON
    HIP_EXTENSIONS OFF
  )
endforeach()
