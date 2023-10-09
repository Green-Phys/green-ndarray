/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#ifndef ALPS_NDARRAY_MATH_H
#define ALPS_NDARRAY_MATH_H

#include <green/ndarray/ndarray.h>

namespace green::ndarray {

  namespace detail {
    template <typename T, size_t Dim, storage_type ST>
    ndarray<T, Dim> transpose_impl(const ndarray<T, Dim, ST>& array, const std::vector<size_t>& pattern) {
      std::vector<size_t> shape(array.shape().size());
      for (size_t i(0); i < array.shape().size(); ++i) {
        shape[pattern[i]] = array.shape()[i];
      }
      ndarray<T, Dim>     result(shape);
      std::vector<size_t> indices(array.dim(), 0);
      for (size_t i(0); i < array.size(); ++i) {
        size_t res = i;
        for (size_t ind(0); ind < array.dim(); ++ind) {
          indices[pattern[array.dim() - ind - 1]] = res % array.shape()[array.dim() - ind - 1];
          res /= array.shape()[array.dim() - ind - 1];
        }
        std::transform(indices.begin(), indices.end(), result.strides().begin(), indices.begin(), std::multiplies<size_t>());
        size_t ind                  = std::accumulate(indices.begin(), indices.end(), size_t(0), std::plus<size_t>());
        result.storage().get()[ind] = array.storage().get()[array.offset() + i];
      }
      return result;
    }
  }  // namespace detail

  // Arithmetic operations on tensors

  // inplace operators

  template <typename T1, typename T2, size_t Dim, storage_type ST, storage_type ST2>
  std::enable_if_t<std::is_convertible<T2, T1>::value, ndarray<T1, Dim, ST>>& operator+=(ndarray<T1, Dim, ST>&        first,
                                                                                         const ndarray<T2, Dim, ST2>& second) {
#ifndef NDEBUG
    if (!std::equal(first.shape().begin(), first.shape().end(), second.shape().begin())) {
      throw std::runtime_error("Arrays size is miss matched.");
    }
#endif
    std::transform(first.begin(), first.end(), second.begin(), first.begin(),
                   [&](const T1 f, const T2 s) { return T1(f) + T1(s); });
    return first;
  }

  template <typename T1, typename T2, size_t Dim, storage_type ST, storage_type ST2>
  std::enable_if_t<std::is_convertible<T2, T1>::value, ndarray<T1, Dim, ST>>& operator-=(ndarray<T1, Dim, ST>&        first,
                                                                                         const ndarray<T2, Dim, ST2>& second) {
#ifndef NDEBUG
    if (!std::equal(first.shape().begin(), first.shape().end(), second.shape().begin())) {
      throw std::runtime_error("Arrays size is miss matched.");
    }
#endif
    std::transform(first.begin(), first.end(), second.begin(), first.begin(),
                   [&](const T1 f, const T2 s) { return T1(f) - T1(s); });
    return first;
  }

  // Inplace operations with scalars

  template <typename T1, typename T2, size_t Dim, storage_type ST>
  std::enable_if_t<is_scalar_v<T2> && std::is_convertible_v<T2, T1>, ndarray<T1, Dim, ST>>& operator/=(
      ndarray<T1, Dim, ST>& first, T2 second) {
    std::transform(first.begin(), first.end(), first.begin(), [&](const T1 f) { return T1(f) / T1(second); });
    return first;
  }

  template <typename T1, typename T2, size_t Dim, storage_type ST>
  std::enable_if_t<is_scalar_v<T2> && std::is_convertible_v<T2, T1>, ndarray<T1, Dim, ST>>& operator*=(
      ndarray<T1, Dim, ST>& first, T2 second) {
    std::transform(first.begin(), first.end(), first.begin(), [&](const T1 f) { return T1(f) * T1(second); });
    return first;
  }

  template <typename T1, typename T2, size_t Dim, storage_type ST>
  std::enable_if_t<is_scalar_v<T2> && std::is_convertible_v<T2, T1>, ndarray<T1, Dim, ST>>& operator+=(
      ndarray<T1, Dim, ST>& first, T2 second) {
    std::transform(first.begin(), first.end(), first.begin(), [&](const T1 f) { return T1(f) + T1(second); });
    return first;
  }

  template <typename T1, typename T2, size_t Dim, storage_type ST>
  std::enable_if_t<is_scalar_v<T2> && std::is_convertible_v<T2, T1>, ndarray<T1, Dim, ST>>& operator-=(
      ndarray<T1, Dim, ST>& first, T2 second) {
    std::transform(first.begin(), first.end(), first.begin(), [&](const T1 f) { return T1(f) - T1(second); });
    return first;
  }

  // Binary operations with tensors
  template <typename T1, typename T2, size_t Dim, storage_type ST, storage_type ST2>
  ndarray<std::common_type_t<T1, T2>, Dim> operator+(const ndarray<T1, Dim, ST>& first, const ndarray<T2, Dim, ST2>& second) {
    using result_t = std::common_type_t<T1, T2>;
#ifndef NDEBUG
    if (!std::equal(first.shape().begin(), first.shape().end(), second.shape().begin())) {
      throw std::runtime_error("Arrays size is miss matched.");
    }
#endif
    ndarray<result_t, Dim> result(first.shape());
    std::transform(first.begin(), first.end(), second.begin(), result.begin(),
                   [&](const T1 f, const T2 s) { return result_t(f) + result_t(s); });
    return result;
  };

  template <typename T1, typename T2, size_t Dim, storage_type ST, storage_type ST2>
  ndarray<std::common_type_t<T1, T2>, Dim> operator-(const ndarray<T1, Dim, ST>& first, const ndarray<T2, Dim, ST2>& second) {
    using result_t = std::common_type_t<T1, T2>;
#ifndef NDEBUG
    if (!std::equal(first.shape().begin(), first.shape().end(), second.shape().begin())) {
      throw std::runtime_error("Arrays size is miss matched.");
    }
#endif
    ndarray<result_t, Dim> result(first.shape());
    std::transform(first.begin(), first.end(), second.begin(), result.begin(),
                   [&](const T1 f, const T2 s) { return result_t(f) - result_t(s); });
    return result;
  };

  // Binary operations with scalars
#define MATH_OP_WITH_SCALAR(OP)                                                                                                 \
  template <typename T1, typename T2, size_t Dim, storage_type ST>                                                              \
  std::enable_if_t<is_scalar_v<T2>, ndarray<std::common_type_t<T1, T2>, Dim>> operator OP(const ndarray<T1, Dim, ST>& first,    \
                                                                                          T2                          second) { \
    using result_t = std::common_type_t<T1, T2>;                                                                                \
    ndarray<result_t, Dim> result(first.shape());                                                                               \
    std::transform(first.begin(), first.end(), result.begin(), [&](const T1 f) { return result_t(f) OP result_t(second); });    \
    return result;                                                                                                              \
  };                                                                                                                            \
                                                                                                                                \
  template <typename T1, typename T2, size_t Dim, storage_type ST>                                                              \
  std::enable_if_t<is_scalar_v<T1>, ndarray<std::common_type_t<T1, T2>, Dim>> operator OP(T1                          first,    \
                                                                                          const ndarray<T2, Dim, ST>& second) { \
    return second OP first;                                                                                                     \
  }

  MATH_OP_WITH_SCALAR(+)
  MATH_OP_WITH_SCALAR(-)
  MATH_OP_WITH_SCALAR(*)
  MATH_OP_WITH_SCALAR(/)

  // Unary operations

  template <typename T1, size_t Dim, storage_type ST>
  ndarray<T1, Dim> operator-(const ndarray<T1, Dim, ST>& first) {
    ndarray<T1, Dim> result(first.shape());
    std::transform(first.begin(), first.end(), result.begin(), [&](const T1 f) { return -f; });
    return result;
  };

  // Comparisons

  template <typename T1, typename T2, size_t Dim, storage_type ST, storage_type ST2>
  bool operator==(const ndarray<T1, Dim, ST>& lhs, const ndarray<T2, Dim, ST2>& rhs) {
    using result_t = std::common_type_t<T1, T2>;
#ifndef NDEBUG
    if (!std::equal(lhs.shape().begin(), lhs.shape().end(), rhs.shape().begin())) {
      throw std::runtime_error("Arrays size is miss matched.");
    }
#endif
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                      [](T1 l, T2 r) { return std::abs(result_t(l) - result_t(r)) < 1e-12; });
  };

  template <typename T, size_t Dim, storage_type ST>
  ndarray<T, Dim> transpose(const ndarray<T, Dim, ST>& array, const std::string& string_pattern) {
    size_t find = string_pattern.find("->");
    if (find == std::string::npos) {
      throw std::runtime_error("Incorrect transpose_impl pattern.");
    }
    std::string from = trim(string_pattern.substr(0, find));
    std::string to   = trim(string_pattern.substr(find + 2, string_pattern.size() - 1));

    if (from.length() != to.length()) {
      throw std::runtime_error("Transpose source and target indices have different size.");
    }
    if (from.length() != array.dim()) {
      throw std::runtime_error("Number of transpose_impl indices and array dimension are different size.");
    }
    if ((!all_latin(from)) || (!all_latin(to))) {
      throw std::runtime_error("Transpose indices should be latin letters.");
    }

#ifndef NDEBUG
    for (const auto& s1 : from) {
      bool in = false;
      for (const auto& s2 : to) {
        if (s1 == s2) {
          in = true;
          break;
        }
      }
      if (!in) {
        throw std::runtime_error("Some LHS transpose indices are not found in RHS transpose_impl indices.");
      }
    }
#endif

    std::map<char, size_t> index_map;
    for (size_t i = 0; i < to.length(); ++i) {
      index_map[to[i]] = i;
    }
    std::vector<size_t> pattern(to.length());
    for (size_t j = 0; j < from.length(); ++j) {
      pattern[j] = index_map[from[j]];
    }
    return detail::transpose_impl(array, pattern);
  }

}  // namespace green::ndarray

#endif  // ALPS_NDARRAY_MATH_H
