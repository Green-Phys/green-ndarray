/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#include <green/ndarray/ndarray_math.h>

#include <catch2/catch_test_macros.hpp>

#include "common.h"

TEST_CASE("NDArrayMathTest") {
  SECTION("MathAddSub") {
    ndarray::ndarray<double, 4> arr1(1, 2, 3, 4);
    initialize_array(arr1);
    ndarray::ndarray<double, 4> arr2(1, 2, 3, 4);
    initialize_array(arr2);
    ndarray::ndarray<double, 4> arr3 = arr1 + arr2;
    REQUIRE(std::abs(double(arr1(0, 1, 2, 0) + arr2(0, 1, 2, 0)) - arr3(0, 1, 2, 0)) < 1e-12);
    ndarray::ndarray<double, 4> arr4 = arr1.copy();
    arr1 += arr2;
    REQUIRE(std::abs(arr1(0, 1, 2, 0) - arr3(0, 1, 2, 0)) < 1e-12);
    arr1 -= arr2;
    REQUIRE(std::abs(arr1(0, 1, 0, 2) - arr4(0, 1, 0, 2)) < 1e-12);
  }

  SECTION("InplaceMathAddSub") {
    ndarray::ndarray<double, 4> arr1(1, 2, 3, 4);
    initialize_array(arr1);
    ndarray::ndarray<double, 4> arr2(1, 2, 3, 4);
    initialize_array(arr2);
    ndarray::ndarray<double, 2> arr3 = arr1(0, 1);
    ndarray::ndarray<double, 2> arr4 = arr2(0, 0);

    ndarray::ndarray<double, 2> arr5 = arr3.copy();
    ndarray::ndarray<double, 2> arr6 = arr4.copy();

    arr3 += arr4;
    arr5 += arr6;

    REQUIRE(std::abs(arr3(0, 1) - arr5(0, 1)) < 1e-12);
    arr3 -= arr4;
    arr5 -= arr6;
    REQUIRE(std::abs(arr3(1, 2) - arr5(1, 2)) < 1e-12);
  }

  SECTION("InplaceMathWithScalars") {
    ndarray::ndarray<double, 4> arr1(1, 2, 3, 4);
    initialize_array(arr1);
    auto                                      arr1_copy = arr1.copy();
    ndarray::ndarray<std::complex<double>, 4> arr2(1, 2, 3, 4);
    initialize_array(arr2);
    auto                 arr2_copy = arr2.copy();
    float                add       = 1.0f;
    double               sub       = 2.0;
    std::complex<float>  mult      = 1.0if;
    std::complex<double> div       = 3. + 2.0i;
    arr1 += add;
    REQUIRE(std::equal(arr1.begin(), arr1.end(), arr1_copy.begin(),
                       [&](double a, double b) { return std::abs(a - (b + add)) < 1e-12; }));
    arr1 -= sub;
    REQUIRE(std::equal(arr1.begin(), arr1.end(), arr1_copy.begin(),
                       [&](double a, double b) { return std::abs(a - (b + add - sub)) < 1e-12; }));
    arr2 *= mult;
    REQUIRE(std::equal(arr2.begin(), arr2.end(), arr2_copy.begin(),
                       [&](const auto& a, const auto& b) { return std::abs(a.real() - b.imag()) < 1e-12; }));
    arr2 /= div;
    REQUIRE(std::equal(arr2.begin(), arr2.end(), arr2_copy.begin(),
                       [&](const auto& a, const auto& b) { return std::abs(a - b * std::complex<double>(mult) / div) < 1e-12; }));
    auto arr3 = arr2(0, 1) * mult;
    REQUIRE(std::abs(arr3(0, 0) - arr2(0, 1, 0, 0) * std::complex<double>(mult)) < 1e-12);
  }

  SECTION("MathAddSubConversion") {
    ndarray::ndarray<double, 4> arr1(1, 2, 3, 4);
    initialize_array(arr1);
    ndarray::ndarray<std::complex<double>, 4> arr2(1, 2, 3, 4);
    initialize_array(arr2);
    ndarray::ndarray<std::complex<double>, 4> arr3 = arr1 + arr2;
    ndarray::ndarray<std::complex<double>, 4> arr4 = arr3 - arr1;
    std::complex<double>                      a1   = arr1(0, 1, 0, 2);
    std::complex<double>                      a2   = arr2(0, 1, 0, 2);
    std::complex<double>                      a3   = arr3(0, 1, 0, 2);
    std::complex<double>                      a4   = arr4(0, 1, 0, 2);

    std::complex<double>                      a12  = a1 + a2;
    REQUIRE(std::abs(a12.real() - a3.real()) < 1e-12);
    REQUIRE(std::abs(a2.real() - a4.real()) < 1e-12);
  }

  SECTION("MathScalarAddSub") {
    ndarray::ndarray<double, 4> arr1(1, 2, 3, 4);
    initialize_array(arr1);
    double                      shift = 15.0;
    ndarray::ndarray<double, 4> arr2  = arr1 + shift;
    REQUIRE(std::abs((arr1(0, 1, 2, 2) + 15.0) - arr2(0, 1, 2, 2)) < 1e-12);
    ndarray::ndarray<double, 4> arr3 = arr2 - shift;
    REQUIRE(std::abs(arr1(0, 1, 2, 0) - arr3(0, 1, 2, 0)) < 1e-12);
    ndarray::ndarray<double, 4> arr4 = shift + arr1;
    REQUIRE(std::abs(arr4(0, 1, 0, 2) - arr2(0, 1, 0, 2)) < 1e-12);
  }

  SECTION("UnaryOp") {
    ndarray::ndarray<double, 4> arr1(1, 2, 3, 4);
    initialize_array(arr1);
    ndarray::ndarray<double, 4> arr2 = -arr1;
    REQUIRE(std::equal(arr1.begin(), arr1.end(), arr2.begin(), [&](double a, double b) { return std::abs(a + b) < 1e-12; }));
  }

  SECTION("Comparison") {
    ndarray::ndarray<double, 4> arr1(1, 2, 3, 4);
    initialize_array(arr1);
    ndarray::ndarray<double, 4> arr2(1, 2, 3, 4);
    arr2 += arr1;

    ndarray::ndarray<double, 4> arr3(1, 2, 3, 4);
    initialize_array(arr3);
    ndarray::ndarray<std::complex<double>, 4> arr4(1, 2, 3, 4);
    arr4 += arr3;

    ndarray::ndarray<int, 4> arr5(1, 2, 3, 4);
    initialize_array(arr3);
    ndarray::ndarray<double, 4> arr6(1, 2, 3, 4);
    arr6 += arr5;

    REQUIRE(arr1 == arr2);
    REQUIRE(arr3 == arr4);
    REQUIRE(arr5 == arr6);
  }

  SECTION("Transpose") {
    ndarray::ndarray<double, 4> array(5, 5, 3, 4);
    initialize_array(array);
    REQUIRE_THROWS_AS(transpose(array, "ijkl->ikl"), std::runtime_error);
    REQUIRE_THROWS_AS(transpose(array, "ijk->ikj"), std::runtime_error);
    REQUIRE_THROWS_AS(transpose(array, "ijkl->ikj1"), std::runtime_error);
    REQUIRE_THROWS_AS(transpose(array, "ijk1->ikjl"), std::runtime_error);
#ifndef NDEBUG
    REQUIRE_THROWS_AS(transpose(array, "ijkl->ikjm"), std::runtime_error);
#endif
    ndarray::ndarray<double, 4> result = transpose(array, "  ijkl -> ikjl ");
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