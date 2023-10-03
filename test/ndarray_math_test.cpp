/*
 * Copyright (c) 2021-2022 Sergei Iskakov
 *
 */

#include <green/ndarray/ndarray_math.h>

#include <catch2/catch_test_macros.hpp>

#include "common.h"

TEST_CASE("NDArrayMathTest") {
  SECTION("MathAddSub") {
    ndarray::ndarray<double> arr1(1, 2, 3, 4);
    initialize_array(arr1);
    ndarray::ndarray<double> arr2(1, 2, 3, 4);
    initialize_array(arr2);
    ndarray::ndarray<double> arr3 = arr1 + arr2;
    REQUIRE(std::abs(double(arr1(0, 1, 2, 0) + arr2(0, 1, 2, 0)) - arr3(0, 1, 2, 0)) < 1e-12);
    ndarray::ndarray<double> arr4 = arr1.copy();
    arr1 += arr2;
    REQUIRE(std::abs(arr1(0, 1, 2, 0) - arr3(0, 1, 2, 0)) < 1e-12);
    arr1 -= arr2;
    REQUIRE(std::abs(arr1(0, 1, 0, 2) - arr4(0, 1, 0, 2)) < 1e-12);
  }

  SECTION("InplaceMathAddSub") {
    ndarray::ndarray<double> arr1(1, 2, 3, 4);
    initialize_array(arr1);
    ndarray::ndarray<double> arr2(1, 2, 3, 4);
    initialize_array(arr2);
    ndarray::ndarray<double> arr3 = arr1(0, 1);
    ndarray::ndarray<double> arr4 = arr2(0, 0);

    ndarray::ndarray<double> arr5 = arr3.copy();
    ndarray::ndarray<double> arr6 = arr4.copy();

    arr3 += arr4;
    arr5 += arr6;

    REQUIRE(std::abs(arr3(0, 1) - arr5(0, 1)) < 1e-12);
    arr3 -= arr4;
    arr5 -= arr6;
    REQUIRE(std::abs(arr3(1, 2) - arr5(1, 2)) < 1e-12);
  }

  SECTION("MathAddSubConversion") {
    ndarray::ndarray<double> arr1(1, 2, 3, 4);
    initialize_array(arr1);
    ndarray::ndarray<std::complex<double>> arr2(1, 2, 3, 4);
    initialize_array(arr2);
    ndarray::ndarray<std::complex<double>> arr3 = arr1 + arr2;
    ndarray::ndarray<std::complex<double>> arr4 = arr3 - arr1;
    std::complex<double>                   a1   = arr1(0, 1, 0, 2);
    std::complex<double>                   a2   = arr2(0, 1, 0, 2);
    std::complex<double>                   a3   = arr3(0, 1, 0, 2);
    std::complex<double>                   a4   = arr4(0, 1, 0, 2);

    std::complex<double>                   a12  = a1 + a2;
    REQUIRE(std::abs(a12.real() - a3.real()) < 1e-12);
    REQUIRE(std::abs(a2.real() - a4.real()) < 1e-12);
  }

  SECTION("MathScalarAddSub") {
    ndarray::ndarray<double> arr1(1, 2, 3, 4);
    initialize_array(arr1);
    double                   shift = 15.0;
    ndarray::ndarray<double> arr2  = arr1 + shift;
    REQUIRE(std::abs((arr1(0, 1, 2, 2) + 15.0) - arr2(0, 1, 2, 2)) < 1e-12);
    ndarray::ndarray<double> arr3 = arr2 - shift;
    REQUIRE(std::abs(arr1(0, 1, 2, 0) - arr3(0, 1, 2, 0)) < 1e-12);
    ndarray::ndarray<double> arr4 = shift + arr1;
    REQUIRE(std::abs(arr4(0, 1, 0, 2) - arr2(0, 1, 0, 2)) < 1e-12);
  }

  SECTION("UnaryOp") {
    ndarray::ndarray<double> arr1(1, 2, 3, 4);
    initialize_array(arr1);
    ndarray::ndarray<double> arr2 = -arr1;
    REQUIRE(std::equal(arr1.begin(), arr1.end(), arr2.begin(), [&](double a, double b) { return std::abs(a + b) < 1e-12; }));
  }

  SECTION("Comparison") {
    ndarray::ndarray<double> arr1(1, 2, 3, 4);
    initialize_array(arr1);
    ndarray::ndarray<double> arr2(1, 2, 3, 4);
    arr2 += arr1;

    ndarray::ndarray<double> arr3(1, 2, 3, 4);
    initialize_array(arr3);
    ndarray::ndarray<std::complex<double>> arr4(1, 2, 3, 4);
    arr4 += arr3;

    ndarray::ndarray<int> arr5(1, 2, 3, 4);
    initialize_array(arr3);
    ndarray::ndarray<double> arr6(1, 2, 3, 4);
    arr6 += arr5;

    REQUIRE(arr1 == arr2);
    REQUIRE(arr3 == arr4);
    REQUIRE(arr5 == arr6);
  }

  SECTION("Transpose") {
    ndarray::ndarray<double> array(5, 5, 3, 4);
    initialize_array(array);
    REQUIRE_THROWS_AS(transpose(array, "ijkl->ikl"), std::runtime_error);
    REQUIRE_THROWS_AS(transpose(array, "ijk->ikj"), std::runtime_error);
    REQUIRE_THROWS_AS(transpose(array, "ijkl->ikj1"), std::runtime_error);
    REQUIRE_THROWS_AS(transpose(array, "ijk1->ikjl"), std::runtime_error);
#ifndef NDEBUG
    REQUIRE_THROWS_AS(transpose(array, "ijkl->ikjm"), std::runtime_error);
#endif
    ndarray::ndarray<double> result = transpose(array, "  ijkl -> ikjl ");
    for (int i = 0; i < 1; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 3; ++k) {
          for (int l = 0; l < 4; ++l) {
            REQUIRE(std::abs(array(i, j, k, l) - result(i, k, j, l)) < 1e-12);
          }
        }
      }
    }
  }
}