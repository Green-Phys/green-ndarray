project(ndarray_test)

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.3 # or a later release_
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_executable(ndarray_test ndarray_test.cpp ndarray_math_test.cpp
        ndarray_storage_test.cpp)
target_link_libraries(ndarray_test
        PRIVATE
        Catch2::Catch2WithMain
        GREEN::NDARRAY)

include(CTest)
include(Catch)
catch_discover_tests(ndarray_test)