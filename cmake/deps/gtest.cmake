# ########################################################################
# Copyright (c) 2021 Advanced Micro Devices, Inc.
# ########################################################################

include(CPM)

CPMFindPackage(
  NAME GTest
  VERSION 1.10.0
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 703bd9caab50b139428cea1aaff9974ebee5742e # release-1.10.0
  EXCLUDE_FROM_ALL ON
)

# Add targets only defined via find_package when found via add_subdirectory
if(NOT TARGET GTest::GTest)
  add_library(GTest::GTest ALIAS gtest)
endif()
