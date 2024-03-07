cmake_minimum_required(VERSION 3.13)
set(TEST_LANG CXX)
set(TEST_EXT cpp)
project(DF_RHF-miniapp LANGUAGES CXX)

#set project program name 
set(app DF_RHF)
set(BLAS_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)


message(STATUS "BLAS libs: ${BLAS_LIBRARIES}")
message(STATUS "BLAS includes: ${BLAS_INCLUDE_DIRS}")


find_package(LAPACK REQUIRED)
message(STATUS "LAPACK libs: ${LAPACK_LIBRARIES}")
message(STATUS "LAPACK includes: ${LAPACK_INCLUDE_DIRS}")

# find_package(HDF5 COMPONENTS CXX HL REQUIRED)


find_package(MPI REQUIRED)
# limit C++ errors to one

add_executable(DF_RHF source/df_fock_build.cpp)

#inlcude blas and lapack 
target_include_directories(${app} PUBLIC ${BLAS_INCLUDE_DIRS} ${LAPACK_INCLUDE_DIRS})
#link blas and lapack
target_link_libraries(${app} PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})


# target_include_directories(${app} PUBLIC ${HDF5_INCLUDE_DIRS})
# target_link_libraries(${app} PUBLIC ${HDF5_LIBRARIES})

#  cmake -CMAKE_CXX_COMPILER=CC cmake -DBLAS_LIBRARIES= ..