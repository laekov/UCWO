project(UCWO)

cmake_minimum_required(VERSION 3.8)

option("USE_CUDA" "Build CUDA example" ON)

function(add_bin src)
    get_filename_component(bin "${src}" NAME_WE)
    add_executable("${bin}" "")
    target_sources("${bin}"
        PRIVATE
        "${src}")
    target_link_libraries("${bin}" ucwo)
    add_test(NAME "${bin}" COMMAND "${bin}")
    set_property(TARGET "${bin}" PROPERTY CXX_STANDARD 17)
endfunction(add_bin)

include_directories("include")

file(GLOB_RECURSE SRCS
    src/ucwo.cc
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

find_package(MPI)
find_package(UCX)
include_directories(${MPI_INCLUDE_PATH})

add_library(ucwo SHARED src/ucwo.cc)

target_link_libraries(ucwo mpi ucp uct ucs)

add_bin("src/put_bench.cc")
add_bin("src/rand_put_bench.cc")
add_bin("src/hello.cc")

if(USE_CUDA)
    enable_language(CUDA)
    add_bin("src/hello_cuda.cu")
endif()
