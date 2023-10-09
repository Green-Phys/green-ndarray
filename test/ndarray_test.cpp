/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#include <green/ndarray/ndarray.h>

#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <random>

#include "common.h"

void test_const_array(ndarray::ndarray<double, 5>& arr1, const ndarray::ndarray<double, 5>& arr2) {
  ndarray::ndarray<const double, 2> slice  = arr2(0, 1, 2);
  double                            slice2 = slice(0, 0);
  REQUIRE(std::abs(arr1(0, 1, 2, 0, 0) - slice2) < 1e-12);
}

TEST_CASE("NDArrayTest") {
  SECTION("Init") {
    ndarray::ndarray<double, 5> array(1, 2, 3, 4, 5);
    ndarray::ndarray<double, 5> default_array;
    REQUIRE(array.size() == 1 * 2 * 3 * 4 * 5);
    REQUIRE(array.strides()[0] == 120);
    REQUIRE(array.strides()[1] == 60);
    REQUIRE(array.strides()[4] == 1);
    REQUIRE(array.shape()[0] == 1);
    REQUIRE(array.shape()[1] == 2);
    REQUIRE(array.shape()[3] == 4);
  }

  SECTION("Init With Vector") {
    std::vector<size_t>         shape{1, 2, 30, 2};
    ndarray::ndarray<double, 4> array(shape);
    initialize_array(array);
    REQUIRE(std::equal(shape.begin(), shape.end(), array.shape().begin()));
  }

  SECTION("Init With Array") {
    std::array<size_t, 4>       shape{1, 2, 30, 2};
    ndarray::ndarray<double, 4> array(shape);
    initialize_array(array);
    REQUIRE(array.shape() == shape);
  }

  SECTION("Init NULL Array and assign new reference") {
    std::array<size_t, 4>                                  shape{2, 2, 1, 1};
    std::vector<double>                                    data{1, 2, 30, 2};
    ndarray::ndarray<double, 4, ndarray::REFERENCE_MEMORY> array_ref(nullptr, shape);
    ndarray::ndarray<double, 4>                            array(shape);
    ndarray::ndarray<double, 4, ndarray::REFERENCE_MEMORY> array_ref2(nullptr, 2, 2, 1, 1);
    array_ref.set_ref(data.data());
    array_ref2.set_ref(array.data());
    initialize_array(array_ref2);
    REQUIRE(array_ref(0, 0, 0, 0) == 1);
    REQUIRE(array_ref(0, 1, 0, 0) == 2);
    REQUIRE(array_ref(1, 0, 0, 0) == 30);
    REQUIRE(array_ref(1, 1, 0, 0) == 2);
    REQUIRE(
        std::equal(array.begin(), array.end(), array_ref2.begin(), [](double a, double b) { return std::abs(a - b) < 1e-12; }));
  }

  SECTION("Slice") {
    ndarray::ndarray<double, 5> array(1, 2, 3, 4, 5);
    initialize_array(array);
    ndarray::ndarray<double, 3> array2(array, 0, 1);
    REQUIRE(array2.size() == 3 * 4 * 5);
    REQUIRE(array2.strides()[0] == 20);
    REQUIRE(array2.strides()[1] == 5);
    REQUIRE(array2.strides()[2] == 1);
    REQUIRE(array2.shape()[0] == 3);
    REQUIRE(array2.shape()[1] == 4);
    REQUIRE(array2.shape()[2] == 5);

    ndarray::ndarray<double, 2> array3 = array2(2);

    REQUIRE(array3.size() == 4 * 5);
    REQUIRE(array3.strides()[0] == 5);
    REQUIRE(array3.strides()[1] == 1);
    REQUIRE(array3.shape()[0] == 4);
    REQUIRE(array3.shape()[1] == 5);
  }

  SECTION("Scalar") {
    ndarray::ndarray<double, 5> array(1, 2, 3, 4, 5);
    initialize_array(array);
    float                value  = array(0, 1, 2, 3, 4);
    std::complex<double> value2 = array(0, 1, 2, 3, 4);
    REQUIRE(std::abs(value - value2.real()) < 1e-8);

    // take reference to an element
    double&                     val   = array(0, 1, 2, 3, 4);

    // take a slice
    ndarray::ndarray<double, 3> slice = array(0, 1);
    // check that value in the slice points to the same storage
    REQUIRE(std::abs(val - slice(2, 3, 4)) < 1e-12);
    // change value at the reference point
    val = 3.0;
    // check that value of the slice has been correctly changed
    REQUIRE(std::abs(val - slice(2, 3, 4)) < 1e-12);

    double v;
    array(0, 1, 1, 1, 1) = 33.0;
    REQUIRE(std::abs(slice(1, 1, 1) - 33) < 1e-12);
  }

#ifndef NDEBUG
  SECTION("WrongDimensions") {
    ndarray::ndarray<double, 5> array(1, 2, 3, 4, 5);
    initialize_array(array);
    // throw if index value is larger than size of corresponding dimension
    REQUIRE_THROWS(array(5, 5));
    // the same for constructors
    REQUIRE_THROWS(ndarray::ndarray<double, 3>(array, 5, 5));
  }
#endif

  SECTION("ConstArray") {
    ndarray::ndarray<double, 5> arr1(1, 2, 3, 4, 5);
    ndarray::ndarray<double, 5> arr2(1, 2, 3, 4, 5);
    initialize_array(arr1);
    arr2 = arr1;
    test_const_array(arr1, arr2);
  }

  SECTION("Copy") {
    ndarray::ndarray<double, 5> arr1(1, 2, 3, 4, 5);
    initialize_array(arr1);
    // create const array
    const ndarray::ndarray<double, 5> arr2 = arr1.copy();
    // make copy of const array to array of consts
    ndarray::ndarray<const double, 5> arr3 = arr2.copy();
    // copy of array of consts to array of consts
    ndarray::ndarray<const double, 5> arr4 = arr3.copy();
    // copy of array of consts to non-const array
    ndarray::ndarray<double, 5>       arr5 = arr3.copy();
    REQUIRE(std::abs(arr1(0, 1, 2, 0, 0) - arr2(0, 1, 2, 0, 0)) < 1e-12);
    REQUIRE(std::abs(arr1(0, 1, 2, 0, 0) - arr3(0, 1, 2, 0, 0)) < 1e-12);
    REQUIRE(std::abs(arr1(0, 1, 2, 0, 0) - arr4(0, 1, 2, 0, 0)) < 1e-12);
    REQUIRE(std::abs(arr1(0, 1, 2, 0, 0) - arr5(0, 1, 2, 0, 0)) < 1e-12);
    // change value in origin
    arr1(0, 1, 2, 0, 0) = -5;
    REQUIRE_FALSE(std::abs(double(arr1(0, 1, 2, 0, 0)) - double(arr2(0, 1, 2, 0, 0))) < 1e-9);
    REQUIRE_FALSE(std::abs(double(arr1(0, 1, 2, 0, 0)) - double(arr3(0, 1, 2, 0, 0))) < 1e-9);
    REQUIRE_FALSE(std::abs(double(arr1(0, 1, 2, 0, 0)) - double(arr4(0, 1, 2, 0, 0))) < 1e-9);
    REQUIRE_FALSE(std::abs(double(arr1(0, 1, 2, 0, 0)) - double(arr5(0, 1, 2, 0, 0))) < 1e-9);
  }

  SECTION("CopyOfSlice") {
    ndarray::ndarray<double, 5> arr1(1, 2, 3, 4, 5);
    initialize_array(arr1);
    // create const array
    ndarray::ndarray<double, 3> arr2 = arr1(0, 1);
    // make copy of const array to array of consts
    ndarray::ndarray<double, 3> arr3 = arr2.copy();
    REQUIRE(std::abs(arr1(0, 1, 2, 0, 0) - arr2(2, 0, 0)) < 1e-12);
    REQUIRE(std::abs(arr1(0, 1, 2, 0, 0) - arr3(2, 0, 0)) < 1e-12);
    for (int i = 0; i < 1; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 3; ++k) {
          REQUIRE(std::abs(arr2(i, j, k) - arr3(i, j, k)) < 1e-12);
        }
      }
    }
    // change value in origin
    arr1(0, 1, 2, 2, 2) = -5;
    REQUIRE_FALSE(std::abs(double(arr2(2, 2, 2)) - double(arr3(2, 2, 2))) < 1e-12);
  }

  SECTION("SetValue") {
    ndarray::ndarray<double, 5> arr1(1, 2, 3, 4, 5);
    initialize_array(arr1);
    double value = arr1(0, 0, 0, 0, 0);
    arr1.set_value(value + 2.0);
    REQUIRE(std::all_of(arr1.begin(), arr1.end(), [&](double x) { return std::abs(x - (value + 2.0)) < 1e-12; }));
    arr1.set_zero();
    REQUIRE(std::all_of(arr1.begin(), arr1.end(), [&](double x) { return std::abs(x) < 1e-12; }));
  }

  SECTION("Reshape") {
    ndarray::ndarray<double, 5> array(1, 2, 3, 4, 5);
    ndarray::ndarray<double, 4> array2(2, 3, 4, 5);
    initialize_array(array);
    std::vector<size_t>         shape{1, 2, 30, 2};
    std::vector<size_t>         strides{120, 60, 2, 1};
    std::array<size_t, 6>       shape_arr{1, 1, 2, 3, 4, 5};
    ndarray::ndarray<double, 4> reshaped_array  = array.reshape(1, 2, 30, 2);
    ndarray::ndarray<double, 6> reshaped_array2 = array.reshape(shape_arr);
    REQUIRE(std::equal(shape.begin(), shape.end(), reshaped_array.shape().begin()));
    REQUIRE(std::equal(shape_arr.begin(), shape_arr.end(), reshaped_array2.shape().begin()));
    REQUIRE(std::equal(strides.begin(), strides.end(), reshaped_array.strides().begin()));
    REQUIRE_THROWS(array.reshape(shape));
    auto reshaped_array3 = array2.reshape(shape);
    REQUIRE(std::equal(shape.begin(), shape.end(), reshaped_array3.shape().begin()));
  }

  SECTION("RangeLoop") {
    ndarray::ndarray<double, 5> array(50, 20, 3, 4, 1);
    array.set_value(2.0);
    for (auto v : array) {
      REQUIRE(std::abs(v - 2.0) < 1e-12);
    }
  }

  SECTION("ElementAccess") {
    ndarray::ndarray<double, 5> arr1(1, 2, 3, 4, 5);
    initialize_array(arr1);
    ndarray::ndarray<double, 2> arr2 = arr1(0, 1, 2);
    REQUIRE(arr1(0, 1, 2, 1, 1) == arr2(1, 1));
  }
}
