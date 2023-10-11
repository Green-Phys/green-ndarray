/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#ifndef ALPS_NDARRAY_H
#define ALPS_NDARRAY_H

#include <algorithm>
#include <array>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "storage.h"

namespace green::ndarray {

  template <typename T>
  struct is_complex : std::false_type {};
  template <typename T>
  struct is_complex<std::complex<T>> : std::true_type {};
  template <typename T>
  constexpr bool is_complex_v = is_complex<T>::value;
  template <typename T>
  using is_scalar = std::integral_constant<bool, std::is_arithmetic<T>::value || is_complex<std::remove_const_t<T>>::value>;
  template <typename T>
  constexpr bool is_scalar_v = is_scalar<T>::value;

  template <typename T, size_t Dim>
  struct ndarray {
    static_assert(is_scalar<T>::value, "ndarray element type should be of a scalar type");

    /**
     * Default constructor
     */
    ndarray() : shape_(), strides_(), size_(0), offset_(0) {
      std::fill(shape_.begin(), shape_.end(), 0);
      std::fill(strides_.begin(), strides_.end(), 0);
    }

    ndarray(const std::array<size_t, Dim>& shape, const std::array<size_t, Dim> strides, size_t offset, storage_t storage) :
        shape_(shape), strides_(strides), size_(size_for_shape(shape)), offset_(offset), storage_(std::move(storage)) {}

    /**
     * Constructor for initialization from dimensions (allocates memory for attribute storage_).
     *
     * @tparam Indices type for indices.
     * @param[in] d1 is first dimension.
     * @param[in] inds are after first dimensions.
     */
    template <typename... Indices>
    ndarray(size_t d1, Indices... inds) :
        ndarray(std::array<size_t, sizeof...(inds) + 1>{
            {d1, size_t(inds)...}
    }) {}

    /**
     * Constructor for initialization from array of dimensions (allocates memory for attribute storage_).
     *
     * @param[in] shape is array while D is its dimension.
     */
    explicit ndarray(const std::array<size_t, Dim>& shape) :
        shape_(get_shape(shape)), strides_(strides_for_shape(shape)), size_(size_for_shape(shape)), offset_(0),
        storage_(sizeof(T) * size_) {
      set_value(0.0);
    }
    explicit ndarray(const std::vector<size_t>& shape) :
        shape_(get_shape(shape)), strides_(strides_for_shape(shape)), size_(size_for_shape(shape)), offset_(0),
        storage_(sizeof(T) * size_) {
      set_value(0.0);
    }

    /**
     * Constructor for initialization from array of dimensions (allocates memory for attribute storage_).
     *
     * @param[in] shape is array while D is its dimension.
     */
    template <typename... Indices>
    explicit ndarray(T* data, size_t dim1, Indices... inds) :
        ndarray(data, std::array<size_t, sizeof...(Indices) + 1>{
                          {dim1, size_t(inds)...}
    }) {}

    /**
     * Constructor for initialization from array of dimensions (allocates memory for attribute storage_).
     *
     * @param[in] shape is array while D is its dimension.
     */
    template <typename shape_type>
    explicit ndarray(T* data, const shape_type& shape) :
        shape_(get_shape(shape)), strides_(strides_for_shape(shape)), size_(size_for_shape(shape)), offset_(0),
        storage_(data, size_) {
      if (data) set_value(0.0);
    }

    /**
     * Constructor for slicing of existing instance.
     *
     * @tparam Indices type for indices.
     * @param[in] ref is existing instance for slicing.
     * @param[in] inds are indices for slicing.
     */
    template <typename T2 = typename std::remove_const<T>::type, size_t Dim2, typename Ind, typename... Indices>
    ndarray(const ndarray<T2, Dim2>& ref, Ind ind1, Indices... inds) :
        ndarray(ref, std::array<size_t, sizeof...(inds) + 1ul>{
                         {size_t(ind1), size_t(inds)...}
    }) {}

    /**
     * Constructor for slicing of an existing instance.
     *
     * @param[in] ref is existing instance for slicing.
     * @param[in] inds is array contains indices for slicing.
     */
    template <typename T2 = std::remove_const_t<T>, size_t Dim2, size_t D>
    ndarray(const ndarray<T2, Dim2>& ref, std::array<size_t, D>&& inds) :
        shape_(get_shape(ref.shape(), inds)), strides_(strides_for_shape(shape_)), size_(size_for_shape(shape_)),
        offset_(ref.offset() + compute_offset(ref.strides(), inds)), storage_(ref.storage()) {}

    /**
     * Copy constructors
     */

    ndarray(const ndarray<T, Dim>& rhs) :
        shape_(rhs.shape()), strides_(rhs.strides()), size_(rhs.size()), offset_(rhs.offset()), storage_(rhs.storage()) {}

    /**
     *
     * @tparam T2
     * @param rhs
     */
    template <typename T2 = std::remove_const_t<T>>
    ndarray(const ndarray<std::enable_if_t<!std::is_same_v<T, T2>, T2>, Dim>& rhs) :
        shape_(rhs.shape()), strides_(rhs.strides()), size_(rhs.size()), offset_(rhs.offset()), storage_(rhs.storage()) {}

    template <typename T2 = std::remove_const_t<T>>
    ndarray(const ndarray<const std::enable_if_t<std::is_same_v<T, T2>, T2>, Dim>& rhs) :
        shape_(rhs.shape()), strides_(rhs.strides()), size_(rhs.size()), offset_(rhs.offset()), storage_(rhs.storage()) {}

    template <size_t Dim2, typename = std::enable_if<Dim != Dim2>, typename T2 = std::remove_const_t<T>>
    explicit ndarray(const ndarray<T2, Dim2>& rhs) :
        shape_(), strides_(), size_(rhs.size()), offset_(rhs.offset()), storage_(rhs.storage()) {}

    /*
     * Move constructor
     */

    ndarray(ndarray<T, Dim>&& rhs) :
        shape_(rhs.shape_), strides_(rhs.strides_), size_(rhs.size_), offset_(rhs.offset_), storage_(rhs.storage_) {
      rhs.storage_ = storage_t();
    }

    /*
     * Assignment operators
     */

    ndarray<T, Dim>& operator=(const ndarray<T, Dim>& rhs) {
      shape_   = rhs.shape_;
      strides_ = rhs.strides_;
      size_    = rhs.size_;
      offset_  = rhs.offset_;
      storage_ = rhs.storage_;
      return *this;
    }

    ndarray<T, Dim>& operator=(ndarray<T, Dim>&& rhs) noexcept {
      shape_       = std::move(rhs.shape_);
      strides_     = std::move(rhs.strides_);
      size_        = rhs.size_;
      offset_      = rhs.offset_;
      storage_     = std::move(rhs.storage_);
      rhs.storage_ = storage_t();
      return *this;
    }

    /**
     * Assign scalar value to all elements
     *
     * @tparam Scalar - scalar type
     * @param rhs     - value of a scalar
     * @return current ndarray with updated value
     */
    template <typename Scalar, typename = typename std::enable_if<is_scalar<Scalar>::value>::type>
    ndarray<T, Dim>& operator=(const Scalar rhs) {
      std::fill(begin(), end(), T(rhs));
      return *this;
    };

    /**
     * Deep copy of array
     *
     * @return new array that is a full copy of current array
     */
    ndarray<std::remove_const_t<T>, Dim> copy() const {
      ndarray<std::remove_const_t<T>, Dim> ret(shape_);
      std::copy(begin(), end(), ret.begin());
      return ret;
    }

    virtual ~ndarray() {}

    /**
     * Returns constant pointer to an element for specific indidces
     *
     * @tparam Indices - indices type (should be convertible to size_t)
     * @param inds - indices of an element
     * @return constant pointer to an element at (inds...)
     */
    template <typename... Indices>
    const T* ref(Indices... inds) const {
#ifndef NDEBUG
      size_t num_of_inds = sizeof...(Indices);
      check_dimensions(num_of_inds);
#endif
      return &storage_.get<T>()[offset_ + get_index(inds...)];
    }

    /**
     * Returns pointer to an element for specific indidces
     *
     * @tparam Indices - types of indices (should be convertible to size_t)
     * @param inds - indices of an element
     * @return pointer to an element at (inds...)
     */
    template <typename... Indices>
    T* ref(Indices... inds) {
#ifndef NDEBUG
      size_t num_of_inds = sizeof...(Indices);
      check_dimensions(num_of_inds);
#endif
      return &storage_.get<T>()[offset_ + get_index(inds...)];
    }

    /**
     * Extract a sub-ndarray at given indices `inds`
     *
     * @tparam Indices - types of indices (should be convertible to size_t)
     * @param inds - indices of a sub-ndarray
     * @return sub-ndarray at `inds` indices
     */
    template <typename... Indices>
    std::enable_if_t<sizeof...(Indices) < Dim, ndarray<T, Dim - sizeof...(Indices)>> operator()(Indices... inds) {
#ifndef NDEBUG
      size_t num_of_inds = sizeof...(Indices);
      check_dimensions(shape_, num_of_inds);
#endif
      return ndarray<T, Dim - sizeof...(Indices)>(*this, inds...);
    };

    /**
     * Extract a const sub-ndarray at a given coordinates `ind`
     *
     * @tparam Indices type of indices (should be convertible to size_t)
     * @param inds - coordinates of a sub-ndarray
     * @return const sub-ndarray at `inds` coordinates
     */
    template <typename... Indices>
    std::enable_if_t<sizeof...(Indices) < Dim, ndarray<const typename std::remove_const<T>::type, Dim - sizeof...(Indices)>>
    operator()(Indices... inds) const {
#ifndef NDEBUG
      size_t num_of_inds = sizeof...(Indices);
      check_dimensions(shape_, num_of_inds);
#endif
      return ndarray<const typename std::remove_const<T>::type, Dim - sizeof...(Indices)>(*this, inds...);
    };

    /**
     * Extract a scalar storage_t at given indices `inds`
     *
     * TODO: add tests
     *
     * @tparam Indices type of indices (should be convertible to size_t)
     * @param inds - indices of a sub-ndarray
     * @return value at `inds` indices
     */
    template <typename... Indices>
    const std::enable_if_t<sizeof...(Indices) == Dim, T>& operator()(Indices... inds) const {
#ifndef NDEBUG
      size_t num_of_inds = sizeof...(Indices);
      if (num_of_inds != shape_.size()) {
        throw std::runtime_error("Number of indices (" + std::to_string(num_of_inds) + ") is not equal to array's dimension (" +
                                 std::to_string(shape_.size()) + ")");
      }
#endif
      return storage_.get<T>()[offset_ + get_index(inds...)];
    }

    /**
     * Extract a scalar storage_t at a given indices `inds`
     *
     * @tparam Indices type of indices (should be convertible to size_t)
     * @param inds - coordinates of a sub-ndarray
     * @return value at `inds` coordinates
     */
    template <typename... Indices>
    std::enable_if_t<sizeof...(Indices) == Dim, T>& operator()(Indices... inds) {
#ifndef NDEBUG
      size_t num_of_inds = sizeof...(Indices);
      if (num_of_inds != shape_.size()) {
        throw std::runtime_error("Number of indices (" + std::to_string(num_of_inds) + ") is not equal to array's dimension (" +
                                 std::to_string(shape_.size()) + ")");
      }
#endif
      return storage_.get<T>()[offset_ + get_index(inds...)];
    }

    /**
     * Set all elements of an ndarray to be `value`
     *
     * @tparam T2 - type of a value (should be convertible to type of an ndarray)
     * @param value - value of all elements of ndarray
     */
    template <typename T2>
    typename std::enable_if<is_scalar<T2>::value && std::is_convertible<T2, T>::value>::type set_value(T2 value) {
      std::fill(begin(), end(), T(value));
    }

    /**
     * Set all elements of the array to zero
     */
    void set_zero() { set_value(T(0)); }
    /**
     * Set all elements of the array to one
     */
    void set_one() { set_value(T(1)); }

    /*
     * Shape and size related operations
     */

    template <typename... Indices>
    auto reshape(size_t ind1, Indices... shape_inds) const {
      std::array<size_t, sizeof...(Indices) + 1> new_shape{
          {ind1, size_t(shape_inds)...}
      };
#ifndef NDEBUG
      if (size_for_shape(new_shape) != size_) throw std::logic_error("new shape is not consistent with old one");
#endif
      ndarray<T, sizeof...(Indices) + 1> result(*this);
      return result.inplace_reshape(new_shape);
    }

    template <size_t NewDim>
    auto reshape(const std::array<size_t, NewDim>& new_shape) const {
#ifndef NDEBUG
      if (size_for_shape(new_shape) != size_) throw std::logic_error("new shape is not consistent with old one");
#endif
      ndarray<T, NewDim> result(*this);
      return result.inplace_reshape(new_shape);
    }

    auto reshape(const std::vector<size_t>& new_shape_v) const {
      std::array<size_t, Dim> new_shape;
      std::copy(new_shape_v.begin(), new_shape_v.end(), new_shape.begin());
#ifndef NDEBUG
      if (new_shape_v.size() != Dim || size_for_shape(new_shape) != size_)
        throw std::logic_error("new shape is not consistent with old one");
#endif
      ndarray<T, Dim> result(*this);
      return result.inplace_reshape(new_shape);
    }

    ndarray<T, Dim> inplace_reshape(const std::array<size_t, Dim>& shape) {
      if (offset_ != 0) {
        throw std::logic_error("new shape is not consistent with old one");
      }
      shape_   = shape;
      strides_ = strides_for_shape(shape);
      return *this;
    }

    template <typename... Indices>
    auto resize(size_t ind1, Indices... shape_inds) const {
      std::array<size_t, sizeof...(Indices) + 1> new_shape{
          {ind1, size_t(shape_inds)...}
      };
      return resize(new_shape);
    }

    template <size_t NewDim>
    auto resize(const std::array<size_t, NewDim>& new_shape) const {
      return ndarray<T, NewDim>(new_shape);
    }

    auto resize(const std::vector<size_t>& new_shape_v) const {
      std::array<size_t, Dim> new_shape;
      std::copy(new_shape_v.begin(), new_shape_v.end(), new_shape.begin());
#ifndef NDEBUG
      if (new_shape_v.size() != Dim) throw std::logic_error("new shape dimensions are not consistent with old one");
#endif
      ndarray<T, Dim> result(new_shape);
      return result;
    }

    // Type change

    template <typename T2>
    auto view() {
      if (sizeof(T) < sizeof(T2) and (shape_[Dim - 1] % (sizeof(T2) / sizeof(T))) != 0) {
        throw std::runtime_error("Array");
      }
      if (sizeof(T) < sizeof(T2) and (offset_ % (sizeof(T2) / sizeof(T))) != 0 ) {
        throw std::runtime_error("Array");
      }
      std::array<size_t, Dim> new_shape(shape_);
      new_shape[Dim - 1] = (shape_[Dim - 1] * sizeof(T)) / sizeof(T2);
      size_t new_offset = (offset_ * sizeof(T)) / sizeof(T2);
      ndarray<T2, Dim> result(new_shape, strides_for_shape(new_shape), new_offset, storage_);
      return result;
    }

    template <typename T2>
    auto astype() {

      std::array<size_t, Dim> new_shape(shape_);
      ndarray<T2, Dim> result(new_shape);
      std::transform(begin(), end(), result.begin(), [] (const T&a) {
        if constexpr (is_complex_v<T> && !is_complex_v<T2>) {
          std::cerr<<"Imaginary part will be discarded when converting from complex into real";
          return T2(a.real());
        } else {
          return T2(a);
        }
      });
      return result;
    }

    // Data accessors

    /**
     * Reset underlying pointer to a new data. This method only works if array does not own its memory.
     *
     * @param new_data pointer to a new data
     */
    void                           set_ref(T* new_data) { storage_.reset(new_data); }

    /**
     * @return constant pointer to the first element of the array
     */
    const T*                       data() const { return storage_.get<T>() + offset_; }
    /**
     * @return pointer to the first element of the array
     */
    T*                             data() { return storage_.get<T>() + offset_; }

    /**
     * Access to the first element for range-based loops
     * @return
     */
    const T*                       begin() const { return storage_.get<T>() + offset_; }
    /**
     * Access to the first element for range-based loops
     * @return
     */
    T*                             begin() { return storage_.get<T>() + offset_; }

    /**
     * Access to the last+1 element for range-based loops
     * @return
     */
    const T*                       end() const { return storage_.get<T>() + offset_ + size_; }
    /**
     * Access to the last+1 element for range-based loops
     * @return
     */
    T*                             end() { return storage_.get<T>() + offset_ + size_; }

    /**
     * @return shared_ptr object used to store underlying data
     */
    const storage_t&               storage() const { return storage_; }

    storage_t&                     storage() { return storage_; }

    /**
     * @return total number of elements in the array
     */
    size_t                         size() const { return size_; }

    /**
     * @return offset of the first element from the storage_ pointer
     */
    size_t                         offset() const { return offset_; }

    /**
     * @return std::vector that contains the shape of the array
     */
    const std::array<size_t, Dim>& shape() const { return shape_; }
    std::array<size_t, Dim>&       shape() { return shape_; }
    /**
     * @return std::vector that contains strides of the array
     */
    const std::array<size_t, Dim>& strides() const { return strides_; }

    /**
     * @return dimension of the array
     */
    size_t                         dim() const { return shape_.size(); }

  private:
    std::array<size_t, Dim> shape_;
    std::array<size_t, Dim> strides_;
    size_t                  size_{};
    size_t                  offset_{};
    storage_t               storage_;

    template <typename... Indices>
    size_t get_index(Indices... inds) const {
#ifndef NDEBUG
      if (sizeof...(Indices) > shape_.size()) throw std::logic_error("wrong dimensions");
#endif

      std::array<size_t, sizeof...(Indices)> ind_arr{{size_t(inds)...}};
#ifndef NDEBUG
      for (size_t i = 0; i < ind_arr.size(); ++i) {
        if (ind_arr[i] >= shape_[i]) throw std::logic_error(std::to_string(i) + "-th index is larger than its dimension.");
      }
#endif
      std::transform(ind_arr.begin(), ind_arr.end(), strides_.begin(), ind_arr.begin(), std::multiplies<size_t>());
      size_t ind = std::accumulate(ind_arr.begin(), ind_arr.end(), size_t(0), std::plus<size_t>());
      return ind;
    }

    template <typename Container, typename Container2>
    size_t compute_offset(Container&& strides, Container2&& inds) const {
      return std::inner_product(inds.begin(), inds.end(), strides.begin(), 0ul);
    }

    /**
     * Compute size of an ndarray with given shape
     *
     * @tparam Container - type of a shape array
     * @param shape - shape of an ndarray
     * @return number of elements that ndarray of a given shape would have
     */
    template <typename Container>
    size_t size_for_shape(const Container& shape) const {
      return std::accumulate(shape.begin(), shape.end(), 1ul, std::multiplies<size_t>());
    }

    /**
     * Extract a shape of a sub-ndarray of a given ndarray of `old_shape` shape
     *
     * @tparam D - dimension of a new ndarray
     * @param old_shape - shape of an existing ndarray
     * @param inds - coordinates of a sub-ndarray
     * @return shape of a sub-ndarray
     */
    template <typename Container, size_t D = 0>
    std::array<size_t, Dim> get_shape(const Container& old_shape, const std::array<size_t, D>& inds = {{}}) const {
#ifndef NDEBUG
      size_t num_of_inds = D;
      check_dimensions(old_shape, num_of_inds);
      for (size_t i = 0; i < inds.size(); ++i) {
        if (inds[i] >= old_shape[i]) throw std::logic_error(std::to_string(i) + "-th index is larger than its dimension.");
      }
#endif
      std::array<size_t, Dim> shape;
      std::copy(old_shape.data() + D, old_shape.data() + old_shape.size(), shape.data());
      return shape;
    }

    /**
     * Compute strides for an ndarray with specific shape
     *
     * @tparam Container - container type for a shape object (should have `size()` and `data()` functions
     * @param shape - shape of an ndarray
     * @return a vector of strides for an ndarray of given shape
     */
    template <typename Container>
    std::array<size_t, Dim> strides_for_shape(Container&& shape) const {
      std::array<size_t, Dim> str;
      if (shape.size() == 0) return str;
      str[shape.size() - 1] = 1;
      for (int k = int(shape.size()) - 2; k >= 0; --k) str[k] = str[k + 1] * shape.data()[k + 1];
      return str;
    }

    /**
     * Check that array is zero-dimension. Throw an exception if it's not.
     */
    void check_zero_dimension() const {
      if (!shape_.empty()) {
        throw std::runtime_error("Array is not directly castable to a scalar. Array's dimension is " +
                                 std::to_string(shape_.size()));
      }
    }

    template <typename Container>
    void check_dimensions(const Container& shape, size_t num_of_inds) const {
      if (num_of_inds > shape.size()) {
        throw std::runtime_error("Number of indices (" + std::to_string(num_of_inds) + ") is larger than array's dimension (" +
                                 std::to_string(shape.size()) + ")");
      }
    }
  };
}  // namespace green::ndarray

#endif  // ALPS_NDARRAY_H
