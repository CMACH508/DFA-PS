cmake_minimum_required(VERSION 3.14)

set(CMAKE_C_COMPILER gcc-11 CACHE PATH "" FORCE)
set(CMAKE_CXX_COMPILER g++-11 CACHE PATH "" FORCE)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif()

project(main)

# include_directories("/usr/local/include")
# include_directories("/usr/local/include/eigen3")
include_directories("../Common")
include_directories("../Common/third_party")

file(GLOB common_src ../Common/*.cpp)

# add_library(common STATIC ${common_files} )
file(GLOB decision_model_src ../Common/DecisionModel/*.cpp)

# add_library(decision_model STATIC ${decision_model_files} )
file(GLOB tfa_src ../Common/FactorAnalysis/*.cpp)

# add_library(tfa STATIC ${tfa_files} )
file(GLOB net_src ../Common/Net/*.cpp)

# add_library(net ${net_files} )
file(GLOB scilib_src ../Common/SciLib/*.cpp)

file(GLOB third_party_src ../Common/third_party/*.c)

add_executable(DFA-PS main.cpp ${common_src} ${decision_model_src} ${tfa_src} ${net_src} ${scilib_src} ${third_party_src})

set_target_properties(DFA-PS
	PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "../x64/Release"
	LIBRARY_OUTPUT_DIRECTORY "../x64/Release"
	RUNTIME_OUTPUT_DIRECTORY "../x64/Release"
)

# target_link_libraries(mp common decision_model tfa net)

# eigen
find_package(Eigen3)
target_link_libraries(DFA-PS Eigen3::Eigen)

# fmt
find_package(fmt)

# target_link_libraries(mp fmt::fmt-header-only)
target_link_libraries(DFA-PS fmt::fmt)

# boost
find_package(Boost REQUIRED COMPONENTS json serialization program_options)
include_directories(${Boost_INCLUDE_DIRS})

# message("${Boost_LIBRARIES}")
target_link_libraries(DFA-PS ${Boost_LIBRARIES})

# openmp
find_package(OpenMP)

if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# torch
list(APPEND CMAKE_PREFIX_PATH ~/libtorch/share/cmake/Torch)
find_package(Torch)
include_directories(DFA-PS ${TORCH_INCLUDE_DIRS})
target_link_libraries(DFA-PS ${TORCH_LIBRARIES})

# MKL
message("MKL path: $ENV{MKLROOT}/lib/cmake/mkl")
list(APPEND CMAKE_PREFIX_PATH $ENV{MKLROOT}/lib/cmake/mkl)
find_package(MKL CONFIG REQUIRED)
include_directories(${MKL_INCLUDE})
target_link_libraries(DFA-PS MKL::MKL)

target_compile_features(DFA-PS PUBLIC cxx_std_17)

# target_compile_options(mp PUBLIC -Wall -Wextra -Werror)
# target_compile_options(mp PUBLIC -fsanitize=address)
add_custom_command(TARGET DFA-PS POST_BUILD
	COMMAND echo "----------------------------------------------"

	# COMMAND ./../x64/Release/DFA-PS
	COMMAND echo "----------------------------------------------")
