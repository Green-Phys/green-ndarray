/*
 * Copyright (c) 2021-2022 Sergei Iskakov
 *
 */

#ifndef NDARRAY_COMMON_H
#define NDARRAY_COMMON_H

#include <green/ndarray/ndarray.h>

#include <random>

using namespace green;

template <typename T, ndarray::storage_type ST>
inline void initialize_array(ndarray::ndarray<T, ST>& array) {
  // Specify the engine and distribution.
  std::mt19937                           mersenne_engine(1);  // Generates pseudo-random integers
  std::uniform_real_distribution<double> dist{0.0, 10.0};

  std::generate(array.begin(), array.end(), [&dist, &mersenne_engine]() -> T { return T(dist(mersenne_engine)); });
}

#endif  // NDARRAY_COMMON_H
