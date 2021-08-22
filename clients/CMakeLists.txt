# ########################################################################
# Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
# ########################################################################

project(rocsolver-clients LANGUAGES C CXX)
if(UNIX)
  enable_language(Fortran)
endif()

# Specify where to put the client binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

# The rocsolver target will exist if the library is being built along with the clients,
# but if this is a clients-only build, we'll have to search for it.
if(NOT TARGET rocsolver)
  find_package(rocsolver REQUIRED CONFIG PATHS ${ROCM_PATH}/rocsolver /opt/rocm/rocsolver)
  get_imported_target_location(location roc::rocsolver)
  message(STATUS "Found rocSOLVER: ${location}")
endif()

if(BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
  # Linking lapack library requires fortran flags
  find_package(cblas REQUIRED CONFIG)
  target_include_directories(cblas SYSTEM INTERFACE ${CBLAS_INCLUDE_DIRS})

  include(FortranCInterface)
  FortranCInterface_HEADER(FortranCInterface.h
    MACRO_NAMESPACE "FC_"
    SYMBOLS
      cbdsqr
      cgebd2
      cgebrd
      cgelq2
      cgelqf
      cgels
      cgeql2
      cgeqlf
      cgeqr2
      cgeqrf
      cgerq2
      cgerqf
      cgesv
      cgesvd
      cgetf2
      cgetrf
      cgetri
      cgetrs
      cheev
      cheevd
      chegs2
      chegst
      chegv
      chegvd
      chemm
      chemv
      chetd2
      chetrd
      clabrd
      clacgv
      clange
      clarf
      clarfb
      clarfg
      clarft
      claswp
      clatrd
      cposv
      cpotf2
      cpotrf
      cpotri
      cpotrs
      cstedc
      csteqr
      ctrtri
      cung2l
      cung2r
      cungbr
      cungl2
      cunglq
      cungql
      cungqr
      cungtr
      cunm2l
      cunm2r
      cunmbr
      cunml2
      cunmlq
      cunmql
      cunmqr
      cunmtr
      daxpy
      dbdsqr
      dgebd2
      dgebrd
      dgelq2
      dgelqf
      dgels
      dgeql2
      dgeqlf
      dgeqr2
      dgeqrf
      dgerq2
      dgerqf
      dgesv
      dgesvd
      dgetf2
      dgetrf
      dgetri
      dgetrs
      dlabrd
      dlange
      dlarf
      dlarfb
      dlarfg
      dlarft
      dlaswp
      dlatrd
      dorg2l
      dorg2r
      dorgbr
      dorgl2
      dorglq
      dorgql
      dorgqr
      dorgtr
      dorm2l
      dorm2r
      dormbr
      dorml2
      dormlq
      dormql
      dormqr
      dormtr
      dposv
      dpotf2
      dpotrf
      dpotri
      dpotrs
      dstedc
      dsteqr
      dsterf
      dsyev
      dsyevd
      dsygs2
      dsygst
      dsygv
      dsygvd
      dsymm
      dsymv
      dsytd2
      dsytrd
      dtrtri
      sbdsqr
      sgebd2
      sgebrd
      sgelq2
      sgelqf
      sgels
      sgeql2
      sgeqlf
      sgeqr2
      sgeqrf
      sgerq2
      sgerqf
      sgesv
      sgesvd
      sgetf2
      sgetrf
      sgetri
      sgetrs
      slabrd
      slange
      slarf
      slarfb
      slarfg
      slarft
      slaswp
      slatrd
      sorg2l
      sorg2r
      sorgbr
      sorgl2
      sorglq
      sorgql
      sorgqr
      sorgtr
      sorm2l
      sorm2r
      sormbr
      sorml2
      sormlq
      sormql
      sormqr
      sormtr
      sposv
      spotf2
      spotrf
      spotri
      spotrs
      sstedc
      ssteqr
      ssterf
      ssyev
      ssyevd
      ssygs2
      ssygst
      ssygv
      ssygvd
      ssymm
      ssymv
      ssytd2
      ssytrd
      strtri
      zaxpy
      zbdsqr
      zgebd2
      zgebrd
      zgelq2
      zgelqf
      zgels
      zgeql2
      zgeqlf
      zgeqr2
      zgeqrf
      zgerq2
      zgerqf
      zgesv
      zgesvd
      zgetf2
      zgetrf
      zgetri
      zgetrs
      zheev
      zheevd
      zhegs2
      zhegst
      zhegv
      zhegvd
      zhemm
      zhemv
      zhetd2
      zhetrd
      zlabrd
      zlacgv
      zlange
      zlarf
      zlarfb
      zlarfg
      zlarft
      zlaswp
      zlatrd
      zposv
      zpotf2
      zpotrf
      zpotri
      zpotrs
      zstedc
      zsteqr
      ztrtri
      zung2l
      zung2r
      zungbr
      zungl2
      zunglq
      zungql
      zungqr
      zungtr
      zunm2l
      zunm2r
      zunmbr
      zunml2
      zunmlq
      zunmql
      zunmqr
      zunmtr
  )

  add_library(clients-common INTERFACE)
  target_include_directories(clients-common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
  )
  target_link_libraries(clients-common INTERFACE fmt::fmt)
  set(common_source_files
    common/lapack_host_reference.cpp
    rocblascommon/utility.cpp
  )
  prepend_path("${CMAKE_CURRENT_SOURCE_DIR}/" common_source_files common_source_paths)
  target_sources(clients-common INTERFACE ${common_source_paths})

  if(BUILD_CLIENTS_BENCHMARKS)
    add_subdirectory(benchmarks)
  endif()

  if(BUILD_CLIENTS_TESTS)
    add_subdirectory(gtest)
  endif()
endif()

if(BUILD_CLIENTS_SAMPLES)
  add_subdirectory(samples)
endif()
