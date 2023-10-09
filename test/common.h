/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#ifndef NDARRAY_COMMON_H
#define NDARRAY_COMMON_H

#include <green/ndarray/ndarray.h>

#include <random>

using namespace green;

using namespace std::complex_literals;

template <typename T, size_t Dim, ndarray::storage_type ST>
inline void initialize_array(ndarray::ndarray<T, Dim, ST>& array) {
  // Specify the engine and distribution.
  std::mt19937                           mersenne_engine(1);  // Generates pseudo-random integers
  std::uniform_real_distribution<double> dist{0.0, 10.0};

  std::generate(array.begin(), array.end(), [&dist, &mersenne_engine]() -> T { return T(dist(mersenne_engine)); });
}

#endif  // NDARRAY_COMMON_H
