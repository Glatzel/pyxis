cmake_minimum_required(VERSION 3.25)
project(PyxisCppProject VERSION 0.0.35)
# set cpp project dir
set(CPP_CMAKE_DIR ../cpp)

# PTX target####################################################################
# Generate PTX files
set (SOURCES ./src/crypto_cuda.cu ./src/datum_compensate_cuda.cu)
set(PTX_OUT_DIR ${CMAKE_BINARY_DIR}/ptx)
foreach(SOURCE ${SOURCES})
    get_filename_component(SOURCE_NAME ${SOURCE} NAME_WE)
    set(PTX_FILE ${PTX_OUT_DIR}/${SOURCE_NAME}.ptx)
    add_custom_command(
        OUTPUT ${PTX_FILE}
        COMMAND nvcc -fmad=false -I ${CMAKE_CURRENT_SOURCE_DIR}/${CPP_CMAKE_DIR}/src -I ${CMAKE_CURRENT_SOURCE_DIR}/${CPP_CMAKE_DIR}/include -ptx ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE} -o ${PTX_FILE}
        DEPENDS ${SOURCE}
        COMMENT "Generating PTX file for ${SOURCE}"
    )
    add_custom_target(generate_ptx_${SOURCE_NAME} ALL DEPENDS ${PTX_FILE})
endforeach()
# Install PTX files
install(DIRECTORY ${PTX_OUT_DIR}/ DESTINATION ptx)
