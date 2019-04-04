# A very simple script to find rkpm-rk4
#
# This moudle exports:
#   rkpm-rk4_FOUND
#   rkpm-rk4_LIBRARY
#   rkpm-rk4_INCLUDE_DIR
#
message("Trying to find rkpm-rk4..")

set(rkpm-rk4_SEARCH_PATHS
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local
    /usr
    /sw # Fink
    /opt/local # DarwinPorts
    /opt/csw # Blastwave
    /opt
    ${rkpm-rk4_DIR})

find_library(rkpm-rk4_LIBRARY
  NAMES rkpm-rk4 librkpm-rk4
  HINTS ${rkpm-rk4_DIR}
  PATH_SUFFIXES lib
  PATHS ${rkpm-rk4_SEARCH_PATHS})

find_path(rkpm-rk4_INCLUDE_DIR rkpm-rk4/body.h
  HINTS ${rkpm-rk4_DIR}
  PATH_SUFFIXES inc)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(rkpm-rk4 REQUIRED_VARS rkpm-rk4_LIBRARY rkpm-rk4_INCLUDE_DIR)
