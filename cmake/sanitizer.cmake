# ########################################################################
# Copyright (c) 2021 Advanced Micro Devices, Inc.
# ########################################################################

# this interface library will always exist, but will only add flags
# to linked targets if address-sanitizer is enabled
add_library( sanitizer-flags INTERFACE )

if( BUILD_ADDRESS_SANITIZER )
  target_compile_options( sanitizer-flags
    INTERFACE
      -fsanitize=address
      -shared-libasan
  )
  target_link_options( sanitizer-flags
    INTERFACE
      -fsanitize=address
      -shared-libasan
      -fuse-ld=lld
  )
endif( )
