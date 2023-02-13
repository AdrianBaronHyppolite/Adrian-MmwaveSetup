INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_MMWAVE_BEAM_CONTROL mmwave_beam_control)

FIND_PATH(
    MMWAVE_BEAM_CONTROL_INCLUDE_DIRS
    NAMES mmwave_beam_control/api.h
    HINTS $ENV{MMWAVE_BEAM_CONTROL_DIR}/include
        ${PC_MMWAVE_BEAM_CONTROL_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    MMWAVE_BEAM_CONTROL_LIBRARIES
    NAMES gnuradio-mmwave_beam_control
    HINTS $ENV{MMWAVE_BEAM_CONTROL_DIR}/lib
        ${PC_MMWAVE_BEAM_CONTROL_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/mmwave_beam_controlTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MMWAVE_BEAM_CONTROL DEFAULT_MSG MMWAVE_BEAM_CONTROL_LIBRARIES MMWAVE_BEAM_CONTROL_INCLUDE_DIRS)
MARK_AS_ADVANCED(MMWAVE_BEAM_CONTROL_LIBRARIES MMWAVE_BEAM_CONTROL_INCLUDE_DIRS)
