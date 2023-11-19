[![GitHub license](https://img.shields.io/github/license/Green-Phys/green-grids?cacheSeconds=3600&color=informational&label=License)](./LICENSE)
[![GitHub license](https://img.shields.io/badge/C%2B%2B-17-blue)](https://en.cppreference.com/w/cpp/compiler_support/17)

![symm](https://github.com/Green-Phys/green-ndarrays/actions/workflows/test.yaml/badge.svg)
[![codecov](https://codecov.io/github/Green-Phys/green-ndarray/graph/badge.svg?token=FL4FPI48MD)](https://codecov.io/github/Green-Phys/green-ndarray)

# green-ndarray
Header-only library for basic operations with multidimensional arrays 

## Basic usage

`green-ndarray` is a C++ header-only library that provides basic routines to operate with multi-dimensional data.
It allows to create a multidimensional array that owns memory, as well as an array for an allocated memory.

To add this library into your project, first 

```CMake
Include(FetchContent)

FetchContent_Declare(
        green-ndarray
        GIT_REPOSITORY https://github.com/Green-Phys/green-ndarray.git
        GIT_TAG origin/main # or a later release
)
FetchContent_MakeAvailable(green-h5pp)
```
Add predefined alias `GREEN::NDARRAY` it to your target:
```CMake
target_link_libraries(<target> PUBLIC GREEN::NDARRAY)
```
And then simply include the following header:
```cpp
#include <green/ndarray/ndarray.h>
```

***
Here is a small example of some possible operations:


```cpp
using namespace green::ndarray;

// Create empty 5d array of doubles
ndarray<double, 5> array;

// Resize array to a new shape
array.resize(3,4,5,6,7);

// Set all elements of array to be 3

array = 3.0;

// Create 5d array of doubles of shape (3,4,5,6,7) and set all elements to 1
ndarray<double, 5> array2(3,4,5,6,7);
array2 = 1.0;

// Take a 2-dimesional slice of array at (1,2,3) leading index
// Both array and array3 point at the same memory region.
// array3 has a non-zero offset from the beginning of a memory region
ndarray<double, 2> array3 = array(1,2,3);

// Get an element of array2 at (1,2,3,4,5) coordinate

double value = array2(1,2,3,4,5);

```

To perform basic mathematic operations you need to include `<green/ndarray/ndarray_math.h>` header.

The following operations are implemented
    - Addition and Subtraction
    - Inplace Addition and Subtraction
    - Addition and Subtraction of a scalar
    - Multiplication and Division by a scalar
    - Inplace operations with scalar
    - Negation of an array

Datatypes of two operands do not need to be the same. For inplace operations value type of a RHS should be 
convertible to a value type of a RHS array. For non-inplace operation an array of common value type for both operands.



# Acknowledgements

This work is supported by National Science Foundation under the award CSSI-2310582
