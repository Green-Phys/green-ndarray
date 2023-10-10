/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#ifndef NDARRAY_STORAGE_H
#define NDARRAY_STORAGE_H

#include <cassert>
#include <cstdlib>
#include <functional>
#include <stdexcept>

namespace green::ndarray {
  struct shared_mem_blk {
    void*   ptr;
    size_t  size;
    size_t* count;
  };

  inline void standard_deallocation(shared_mem_blk blk) {
    if (!blk.count) return;
    size_t& count = *blk.count;
    --count;
    assert(count >= 0);
    if (blk.ptr == nullptr) {
      return;
    }
    if (count == 0) {
      std::free(blk.ptr);
      std::free(blk.count);
      blk.ptr   = nullptr;
      blk.count = nullptr;
    }
  }

  inline void noop_deallocation(shared_mem_blk blk) {
    if (!blk.count) return;
    size_t& count = *blk.count;
    --count;
    if (count == 0) {
      std::free(blk.count);
      blk.count = nullptr;
      return;
    }
    if (blk.ptr == nullptr) return;
    assert(count >= 0);
  }

  /**
   * Simple reference counting storage for a raw data. Works with both self-managing allocated data and with outside data.
   * For self-managing data the proper destructor is assigned and data will be dallocated if current object is the only object
   * holding reference to the data
   */
  class storage_t {
  public:
    /**
     * Default constructor
     */
    storage_t() : data_{nullptr, 0, nullptr}, release_(noop_deallocation) {}

    /**
     * Create storage and allocate data of `size' bytes
     * @param size - number of bytes to allocate
     */
    explicit storage_t(size_t size) : data_{std::malloc(size), size, new size_t(1)}, release_(standard_deallocation) {}
    /**
     * Create storage for outside managed data
     *
     * @param ref - pointer to the data.
     * @param size - optional
     */
    explicit storage_t(void* ref, size_t size = 0) : data_{ref, size, new size_t(1)}, release_(noop_deallocation) {}

    /**
     * Move constructor
     * @param rhs - object to be moved
     */
    storage_t(storage_t&& rhs) : data_(rhs.data_), release_(rhs.release_) {
      rhs.data_.ptr   = nullptr;
      rhs.data_.count = nullptr;
    }
    /**
     * Copy constructor. New storage will point to the exact same location in memory and reference counter will be incremented.
     *
     * @param rhs - objects to be copied
     */
    storage_t(const storage_t& rhs) : data_(rhs.data_), release_(rhs.release_) { ++(*data_.count); }

    /**
     * Destructor will release possesion of the data. For self-managed data memory will be freed if needed.
     */
    ~storage_t() {
      if (release_) release_(data_);
    }

    /**
     * Copy assignment
     *
     * @param rhs object to assign from
     * @return storage that point at the exact same memory as `rhs`
     */
    storage_t& operator=(const storage_t& rhs) {
      data_ = rhs.data_;
      ++(*data_.count);
      release_ = rhs.release_;
      return *this;
    }
    /**
     * Move-assignment. Move data from rhs to current storage.
     *
     * @param rhs
     * @return storage that point at the exact same memory as `rhs`
     */
    storage_t& operator=(storage_t&& rhs) {
      data_           = std::move(rhs.data_);
      release_        = std::move(rhs.release_);
      rhs.data_.ptr   = nullptr;
      rhs.data_.count = nullptr;
      return *this;
    }

    /**
     * Get pointer to the managed data casted to a proper type.
     *
     * @tparam T - type to cast pointer into.
     * @return pointer to a mamged data.
     */
    template <typename T>
    T* get() const {
      if (data_.size % sizeof(T) != 0) {
        throw std::runtime_error("data can not be represented in chosen type");
      }
      return static_cast<T*>(data_.ptr);
    }

    /**
     * Reset storage to a new data and assign proper memory handling function.
     *
     * @param new_data - pointer to a new memory region.
     * @param release_fun - function used to release data posession. by default noop function is used.
     * @param size - size of the managed data, 0 by default.
     */
    void reset(void* new_data, std::function<void(shared_mem_blk)> release_fun = noop_deallocation, size_t size = 0) {
      release_(data_);
      release_ = release_fun;
      data_    = {new_data, size, new size_t(1)};
    }

    // next two functions are made public for test puropse
    const shared_mem_blk&               data() const { return data_; }
    std::function<void(shared_mem_blk)> release() const { return release_; }

  private:
    shared_mem_blk                      data_;
    std::function<void(shared_mem_blk)> release_;
  };

}  // namespace green::ndarray

#endif  // NDARRAY_STORAGE_H
