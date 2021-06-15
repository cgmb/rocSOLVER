# ########################################################################
# Copyright (c) 2021 Advanced Micro Devices, Inc.
# ########################################################################

include(CPM)

CPMFindPackage(
  NAME fmt
  VERSION 7.1.3
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 7bdf0628b1276379886c7f6dda2cef2b3b374f0b # 7.1.3
  EXCLUDE_FROM_ALL ON
)
