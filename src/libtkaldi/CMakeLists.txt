file(
  GLOB_RECURSE
  LIBTKALDI_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}
  *.cc
)

add_library(
  _tkaldi
  SHARED
  ${LIBTKALDI_SOURCES}
)
set_target_properties(_tkaldi PROPERTIES PREFIX "")

target_include_directories(
  _tkaldi
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  )

target_link_libraries(
  _tkaldi
  ${TORCH_LIBRARIES}
 )
