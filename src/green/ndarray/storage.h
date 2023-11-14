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
    void*  ptr;
    size_t size;
    int    count;
  };

  typedef void (*dealloc_fun)(shared_mem_blk& blk);

  inline void  standard_deallocation(shared_mem_blk& blk) {
    // if (!blk.count) return;
    assert(blk.count > 0);
    --blk.count;
    if (blk.count == 0) {
      if (blk.ptr) std::free(blk.ptr);
      blk.ptr = nullptr;
      delete &blk;
    }
  }

  inline void noop_deallocation(shared_mem_blk& blk) {
    if (!blk.count) return;
    --blk.count;
    if (blk.count == 0) {
      delete &blk;
      return;
    }
    if (blk.ptr == nullptr) return;
    assert(blk.count >= 0);
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
             storage_t() : data_(new shared_mem_blk{nullptr, 0, 0}), release_(noop_deallocation) {}

    /**
     * Create storage and allocate data of `size' bytes
     * @param size - number of bytes to allocate
     */
    explicit storage_t(size_t size) : data_(new shared_mem_blk{std::malloc(size), size, 1}), release_(standard_deallocation) {}
    /**
     * Create storage for outside managed data
     *
     * @param ref - pointer to the data.
     * @param size - optional
     */
    explicit storage_t(void* ref, size_t size = 0) : data_(new shared_mem_blk{ref, size, 1}), release_(noop_deallocation) {}

    /**
     * Move constructor
     * @param rhs - object to be moved
     */
             storage_t(storage_t&& rhs) noexcept : data_(rhs.data_), release_(rhs.release_) { rhs.data_ = nullptr; }
    /**
     * Copy constructor. New storage will point to the exact same location in memory and reference counter will be incremented.
     *
     * @param rhs - objects to be copied
     */
             storage_t(const storage_t& rhs) : data_(rhs.data_), release_(rhs.release_) { ++(data_->count); }

    /**
     * Destructor will release possesion of the data. For self-managed data memory will be freed if needed.
     */
    ~        storage_t() {
      if (release_ && data_) release_(*data_);
    }

    /**
     * Copy assignment
     *
     * @param rhs object to assign from
     * @return storage that point at the exact same memory as `rhs`
     */
    storage_t& operator=(const storage_t& rhs) {
      if (release_) release_(*data_);
      data_ = rhs.data_;
      if (data_->count) {
        ++(data_->count);
      }
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
      if (release_) release_(*data_);
      data_        = rhs.data_;
      release_     = rhs.release_;
      rhs.data_    = nullptr;
      rhs.release_ = nullptr;
      return *this;
    }

    /**
     * Get pointer to the managed data casted to a proper type.
     *
     * @tparam T - type to cast pointer into.
     * @return pointer to a managed data.
     */
    template <typename T>
    T* get() const {
#ifndef NDEBUG
      if (data_->size % sizeof(T) != 0) {
        throw std::runtime_error("data can not be represented in chosen type");
      }
#endif
      return static_cast<T*>(data_->ptr);
    }

    /**
     * Reset storage to a new data and assign proper memory handling function.
     *
     * @param new_data - pointer to a new memory region.
     * @param release_fun - function used to release data posession. by default noop function is used.
     * @param size - size of the managed data, 0 by default.
     */
    void reset(void* new_data, dealloc_fun release_fun = noop_deallocation, size_t size = 0) {
      release_(*data_);
      release_ = release_fun;
      data_    = new shared_mem_blk{new_data, size, 1};
    }

    // next two functions are made public for test puropse
    [[nodiscard]] const shared_mem_blk& data() const { return *data_; }
    [[nodiscard]] dealloc_fun           release() const { return release_; }

  private:
    shared_mem_blk* data_;
    dealloc_fun     release_;
  };

}  // namespace green::ndarray

#endif  // NDARRAY_STORAGE_H
