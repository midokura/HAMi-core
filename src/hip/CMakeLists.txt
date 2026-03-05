set(HIP_SRC
    hook.c
    memory.c
)

add_library(hip_obj OBJECT ${HIP_SRC})
target_include_directories(hip_obj PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/multiprocess
    ${ROCM_HOME}/include
    ${ROCM_HOME}/include/hip
)
target_compile_options(hip_obj PRIVATE -fPIC -D_GNU_SOURCE -Wall -Wextra)
