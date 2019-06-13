# A very simple script to find shell-element
#
# This moudle exports:
#   shell-element_FOUND
#   shell-element_LIBRARY
#   shell-element_INCLUDE_DIR
#
message("Trying to find shell-element..")

set(LIBMESH_SEARCH_PATHS
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local
    /usr
    /sw # Fink
    /opt/local # DarwinPorts
    /opt/csw # Blastwave
    /opt
    ${LIBMESH_DIR})

find_library(libmesh_LIBRARY
    NAMES mesh_opt
    HINTS ${LIBMESH_INCLUDE_DIR}
    PATH_SUFFIXES lib
    PATH ${LIBMESH_SEARCH_PATHS})

set(shell-element_SEARCH_PATHS
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local
    /usr
    /sw # Fink
    /opt/local # DarwinPorts
    /opt/csw # Blastwave
    /opt
    ${shell-element_DIR})

find_library(shell-element_LIBRARY
  NAMES shell-element
  HINTS ${shell-element_DIR}
  PATH_SUFFIXES lib
  PATHS ${shell-element_SEARCH_PATHS})

find_path(shell-element_INCLUDE_DIR fem-shell.h
  HINTS ${shell-element_DIR}
  PATH_SUFFIXES include/shell-element)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(shell-element REQUIRED_VARS shell-element_LIBRARY shell-element_INCLUDE_DIR)
