################################################################################
# libtkaldi ( + TorchScript binding)
################################################################################
file(
  GLOB_RECURSE
  LIBTKALDI_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/base/*.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/feat/*.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix/*.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/util/*.cc
)

add_library(
  tkaldi
  SHARED
  ${LIBTKALDI_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/register.cc
)

target_include_directories(
  tkaldi
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(
  tkaldi
  ${TORCH_LIBRARIES}
)

################################################################################
# Executables
################################################################################
add_executable(
  compute-kaldi-pitch-feats
  ${CMAKE_CURRENT_SOURCE_DIR}/src/featbin/compute-kaldi-pitch-feats.cc
)

target_link_libraries(
  compute-kaldi-pitch-feats
  tkaldi
)
