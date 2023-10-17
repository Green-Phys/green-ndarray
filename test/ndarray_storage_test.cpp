/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#include <green/ndarray/storage.h>

#include <catch2/catch_test_macros.hpp>
#include <complex>

namespace gn = green::ndarray;

size_t getAddress(std::function<void(gn::shared_mem_blk)> f) {
  typedef void(fnType)(gn::shared_mem_blk);
  fnType** fnPointer = f.template target<fnType*>();
  return (size_t)*fnPointer;
}

TEST_CASE("Storage") {
  SECTION("Create") {
    gn::storage_t* st1 = new gn::storage_t;
    gn::storage_t  st2(10);
    REQUIRE(st1->data().ptr == nullptr);
    REQUIRE(st1->data().size == 0);
    REQUIRE(st1->data().count == nullptr);
    REQUIRE(st2.data().ptr != nullptr);
    REQUIRE(st2.data().size == 10);
    REQUIRE(*st2.data().count == 1);
    *st1 = st2;
    REQUIRE(st1->data().ptr != nullptr);
    REQUIRE(st1->data().ptr == st2.data().ptr);
    REQUIRE(st1->data().size == 10);
    REQUIRE(*st1->data().count == 2);
    {
      gn::storage_t st3 = st2;
      REQUIRE(*st2.data().count == 3);
      REQUIRE(*st3.data().count == 3);
      REQUIRE(st2.data().ptr == st3.data().ptr);
      gn::storage_t st4(st2);
      REQUIRE(*st3.data().count == 4);
      REQUIRE(*st2.data().count == 4);
      REQUIRE(st4.data().ptr == st4.data().ptr);
    }
    delete st1;
    REQUIRE(*st2.data().count == 1);

    auto        fn = [](size_t sz) -> gn::storage_t { return gn::storage_t(sz); };
    auto        fx = [](gn::storage_t s) -> gn::storage_t { return s; };

    gn::storage_t st4(fx(gn::storage_t(10)));
    REQUIRE(*st4.data().count == 1);
    REQUIRE(st4.data().size == 10);
    gn::storage_t st5;
    st5 = fn(20);
    REQUIRE(*st5.data().count == 1);
    REQUIRE(st5.data().size == 20);
  }
  SECTION("Copy Assignment") {
    gn::storage_t st1(100);
    gn::storage_t st2(200);
    {
      gn::storage_t st3(st1);
      REQUIRE(*st3.data().count == 2);
      st1 = st2;
      REQUIRE(*st3.data().count == 1);
    }
  }

  SECTION("Move Assignment") {
    gn::storage_t st1(100);
    {
      auto f_move = [] (gn::storage_t st) {return st;};
      gn::storage_t st3(st1);
      REQUIRE(*st3.data().count == 2);
      st1 = f_move(gn::storage_t(100));
      REQUIRE(*st3.data().count == 1);
    }
    REQUIRE(*st1.data().count == 1);
  }

  SECTION("Create ref") {
    std::vector<double> x(100);
    gn::storage_t       st1(x.data());
    REQUIRE(getAddress(st1.release()) == (size_t)&gn::noop_deallocation);
    REQUIRE(st1.get<double>() == x.data());
  }
  SECTION("Reset ref") {
    std::vector<double> x(100);
    gn::storage_t             st1(50);
    const gn::shared_mem_blk* ref = &st1.data();
    st1.reset(x.data());
    REQUIRE(ref->ptr == x.data());
  }
  SECTION("Change Storage type") {
    std::vector<double> x(100);
    gn::storage_t       st1(x.data());
    REQUIRE(getAddress(st1.release()) == (size_t)&gn::noop_deallocation);
    REQUIRE(st1.get<double>() == x.data());
    REQUIRE(st1.data().size == 0);
    {
      gn::storage_t st2(20 * sizeof(double));
      st1 = st2;
      REQUIRE(*st1.data().count == 2);
    }
    REQUIRE(st1.data().size == 20 * sizeof(double));
    REQUIRE(*st1.data().count == 1);
    REQUIRE(getAddress(st1.release()) == (size_t)&gn::standard_deallocation);
    REQUIRE(st1.get<double>() != x.data());
    st1 = gn::storage_t(x.data());
    REQUIRE(getAddress(st1.release()) == (size_t)&gn::noop_deallocation);
    REQUIRE(st1.get<double>() == x.data());
  }
  SECTION("Access data") {
    gn::storage_t st1(2 * sizeof(double));
    st1.get<double>()[0] = 10;
    st1.get<double>()[1] = 15;
    auto x               = *st1.get<std::complex<double>>();
    REQUIRE(std::abs(x.real() - 10.0) < 1e-12);
    REQUIRE(std::abs(x.imag() - 15.0) < 1e-12);
    gn::storage_t st2(sizeof(double));
#ifndef NDEBUG
    REQUIRE_THROWS(st2.get<std::complex<double>>());
#endif
  }
}