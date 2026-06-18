dnl SPDX-License-Identifier: BSD-3-Clause
dnl Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).

AC_PREREQ([2.69])
AC_INIT([gdtq], [0.0.2], [tkouya@gmail.com])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([inc/gqd_type.h])
AC_CONFIG_HEADERS([config.h])

AM_INIT_AUTOMAKE([1.13 foreign subdir-objects -Wall -Wno-portability dist-xz no-dist-gzip])
AM_SILENT_RULES([yes])

dnl ---- compilers ----
AC_PROG_CC
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_MAKE_SET

dnl ---- static archiver (for the consolidated libgqd.a) ----
AM_PROG_AR
AC_PROG_RANLIB

dnl OpenMP for the benchmark CPU side.
dnl AC_OPENMP queries the *current* language; we need C++ flags for
dnl benchmark.cpp / test_util.cpp, so push C++ before invoking it.
AC_LANG_PUSH([C++])
AC_OPENMP
AC_LANG_POP([C++])
AC_SUBST([OPENMP_CXXFLAGS])
dnl Last-resort fallback: if AC_OPENMP detected nothing (e.g. older
dnl autoconf), assume gcc/g++ and use -fopenmp.
AS_IF([test -z "$OPENMP_CXXFLAGS"], [OPENMP_CXXFLAGS="-fopenmp"])

dnl ---- CUDA ----
AX_CUDA

dnl ---- QD library (CPU dd_real / qd_real, used by benchmark) ----
AC_ARG_WITH([qd],
  [AS_HELP_STRING([--with-qd=PATH], [QD library install prefix])],
  [QD_PREFIX="$withval"], [QD_PREFIX=""])
AC_ARG_WITH([qd-include],
  [AS_HELP_STRING([--with-qd-include=DIR], [QD library include directory])],
  [QD_INCLUDE="$withval"],
  [AS_IF([test -n "$QD_PREFIX"], [QD_INCLUDE="$QD_PREFIX/include"], [QD_INCLUDE=""])])
AC_ARG_WITH([qd-lib],
  [AS_HELP_STRING([--with-qd-lib=DIR], [QD library lib directory])],
  [QD_LIB="$withval"],
  [AS_IF([test -n "$QD_PREFIX"], [QD_LIB="$QD_PREFIX/lib"], [QD_LIB=""])])

QD_CPPFLAGS=""
QD_LDFLAGS=""
QD_LIBS="-lqd"
AS_IF([test -n "$QD_INCLUDE"], [QD_CPPFLAGS="-I$QD_INCLUDE"])
AS_IF([test -n "$QD_LIB"], [QD_LDFLAGS="-L$QD_LIB"])
AC_SUBST([QD_CPPFLAGS])
AC_SUBST([QD_LDFLAGS])
AC_SUBST([QD_LIBS])

dnl ---- User-supplied CPU triple-double library (for benchmark gold) ----
AC_ARG_WITH([tdlib],
  [AS_HELP_STRING([--with-tdlib=PATH], [CPU triple-double library install prefix])],
  [TDLIB_PREFIX="$withval"], [TDLIB_PREFIX=""])
AC_ARG_WITH([tdlib-include],
  [AS_HELP_STRING([--with-tdlib-include=DIR], [CPU TD library include directory])],
  [TDLIB_INCLUDE="$withval"],
  [AS_IF([test -n "$TDLIB_PREFIX"], [TDLIB_INCLUDE="$TDLIB_PREFIX/include"], [TDLIB_INCLUDE=""])])
AC_ARG_WITH([tdlib-lib],
  [AS_HELP_STRING([--with-tdlib-lib=DIR], [CPU TD library lib directory])],
  [TDLIB_LIB="$withval"],
  [AS_IF([test -n "$TDLIB_PREFIX"], [TDLIB_LIB="$TDLIB_PREFIX/lib"], [TDLIB_LIB=""])])
AC_ARG_WITH([tdlib-name],
  [AS_HELP_STRING([--with-tdlib-name=NAME], [Link name for CPU TD library (default: td)])],
  [TDLIB_NAME="$withval"], [TDLIB_NAME="td"])
AC_ARG_WITH([tdlib-header],
  [AS_HELP_STRING([--with-tdlib-header=HDR], [Header file for CPU TD type (default: td/td_real.h)])],
  [TDLIB_HEADER="$withval"], [TDLIB_HEADER="td/td_real.h"])
AC_ARG_WITH([tdlib-type],
  [AS_HELP_STRING([--with-tdlib-type=NAME], [C++ type name of CPU TD scalar (default: td_real)])],
  [TDLIB_TYPE="$withval"], [TDLIB_TYPE="td_real"])

TDLIB_CPPFLAGS=""
TDLIB_LDFLAGS=""
TDLIB_LIBS=""
HAVE_TDLIB=no
AS_IF([test -n "$TDLIB_INCLUDE" -o -n "$TDLIB_LIB" -o -n "$TDLIB_PREFIX"], [
  AS_IF([test -n "$TDLIB_INCLUDE"], [TDLIB_CPPFLAGS="-I$TDLIB_INCLUDE"])
  AS_IF([test -n "$TDLIB_LIB"], [TDLIB_LDFLAGS="-L$TDLIB_LIB"])
  TDLIB_LIBS="-l$TDLIB_NAME"
  HAVE_TDLIB=yes
  AC_DEFINE([HAVE_TDLIB], [1], [Define to 1 if a CPU triple-double library is available])
  AC_DEFINE_UNQUOTED([TDLIB_HEADER], ["$TDLIB_HEADER"], [Header path of CPU TD library])
  AC_DEFINE_UNQUOTED([TDLIB_TYPE], [$TDLIB_TYPE], [C++ type name for CPU TD scalar])
])
AC_SUBST([TDLIB_CPPFLAGS])
AC_SUBST([TDLIB_LDFLAGS])
AC_SUBST([TDLIB_LIBS])
AM_CONDITIONAL([HAVE_TDLIB], [test "x$HAVE_TDLIB" = "xyes"])

dnl ---- benchmark switch ----
AC_ARG_ENABLE([benchmark],
  [AS_HELP_STRING([--disable-benchmark], [do not build the benchmark in test/])],
  [enable_benchmark=$enableval], [enable_benchmark=yes])
AM_CONDITIONAL([ENABLE_BENCHMARK], [test "x$enable_benchmark" = "xyes"])

AC_CONFIG_FILES([
  Makefile
  inc/Makefile
  docs/Makefile
  test/Makefile
  gqd.pc
  gqd-shared.pc
])
AC_OUTPUT

echo ""
echo "  ${PACKAGE_NAME} ${PACKAGE_VERSION} configuration"
echo "  -----------------------------------"
echo "  prefix             : ${prefix}"
echo "  static library     : ${libdir}/libgqd.a   (device-linkable; for kernels)"
echo "  shared library     : ${libdir}/libgqd.so  (host API only)"
echo "  headers            : ${includedir}/gdtq/"
echo "  pkg-config         : ${libdir}/pkgconfig/gqd.pc, gqd-shared.pc"
echo "  CUDA toolkit dir   : ${CUDA_DIR}"
echo "  CUDA arch          : ${CUDA_ARCH}"
echo "  NVCC               : ${NVCC}"
echo "  QD includes        : ${QD_CPPFLAGS:-<system>}"
echo "  QD libs            : ${QD_LDFLAGS} ${QD_LIBS}"
echo "  CPU TD library     : ${HAVE_TDLIB}"
AS_IF([test "x$HAVE_TDLIB" = "xyes"], [
echo "    header           : ${TDLIB_HEADER}"
echo "    type             : ${TDLIB_TYPE}"
echo "    link             : ${TDLIB_LDFLAGS} ${TDLIB_LIBS}"
])
echo "  build benchmark    : ${enable_benchmark}"
echo ""
