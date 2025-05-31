cmake_minimum_required(VERSION 3.25)
project(PyxisCppProject VERSION 0.0.24)
# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add source files
set(SOURCES
src/datum_compensate.cpp
src/crypto.cpp
)
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
# target pyxis_static###########################################################
# Build a static library
add_library(pyxis_static STATIC ${SOURCES})
target_include_directories(pyxis_static PUBLIC include include/pyxis)
set_target_properties(pyxis_static PROPERTIES
    OUTPUT_NAME "pyxis"
    POSITION_INDEPENDENT_CODE ON) # Required for shared libraries

# install static
install(TARGETS pyxis_static
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

# Configure and install .pc files
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pyxis_static.pc.in
        ${CMAKE_BINARY_DIR}/pyxis_static.pc
        @ONLY
    )
install(
        FILES ${CMAKE_BINARY_DIR}/pyxis_static.pc
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig
    )

# target pyxis_static###########################################################
# Build a shared library
add_library(pyxis_shared SHARED ${SOURCES})
target_include_directories(pyxis_shared PUBLIC include)
set_target_properties(pyxis_shared PROPERTIES
    OUTPUT_NAME "pyxis"
    POSITION_INDEPENDENT_CODE ON)

# Install targets
install(TARGETS pyxis_shared
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/pyxis
        DESTINATION include/pyxis
        FILES_MATCHING PATTERN "*.h")
# Configure and install .pc files
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pyxis_shared.pc.in
    ${CMAKE_BINARY_DIR}/pyxis_shared.pc
    @ONLY
)
install(
    FILES ${CMAKE_BINARY_DIR}/pyxis_shared.pc
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig
)
