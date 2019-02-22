# A very simple script to find mfree_iwf
#
# This moudle exports:
#   MFREE_IWF_FOUND
#   MFREE_IWF_LIBRARY
#   MFREE_IWF_INCLUDE_DIR
#
message("Trying to find mfree_iwf..")

set(MFREE_IWF_SEARCH_PATHS
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local
    /usr
    /sw # Fink
    /opt/local # DarwinPorts
    /opt/csw # Blastwave
    /opt
    ${MFREE_IWF_DIR})

find_library(MFREE_IWF_LIBRARY
  NAMES mfree_iwf libmfree_iwf
  HINTS ${MFREE_IWF_DIR}
  PATH_SUFFIXES lib
  PATHS ${MFREE_IWF_SEARCH_PATHS})

find_path(MFREE_IWF_INCLUDE_DIR mfree_iwf/body.h
  HINTS ${MFREE_IWF_DIR}
  PATH_SUFFIXES inc)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MFREE_IWF REQUIRED_VARS MFREE_IWF_LIBRARY MFREE_IWF_INCLUDE_DIR)
