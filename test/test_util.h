#ifndef __GQD_TEST_UTIL_H__
#define __GQD_TEST_UTIL_H__

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "gqd_type.h"
#include <sys/time.h>
#include <qd/qd_real.h>

#ifdef HAVE_TDLIB
/* User-supplied CPU triple-double type, configured at build time via
 * --with-tdlib-header / --with-tdlib-type.  The type must expose a
 * `qd_real`-like interface: random construction from string, member
 * `.x[0..2]` access, arithmetic operators, abs(), to_string(). */
#  include TDLIB_HEADER
#endif


void randArray(dd_real* data, const unsigned numElement,
	       dd_real low, dd_real up, int seed = 0);


void randArray(qd_real* data, const unsigned numElement,
               qd_real low, qd_real up, int seed = 0);

void qd2gqd(dd_real* dd_data, gdd_real* gdd_data, const unsigned int numElement);

void qd2gqd(qd_real* qd_data, gqd_real* gqd_data, const unsigned int numElement);

void gqd2qd(gdd_real* gdd_data, dd_real* dd_data, const unsigned int numElement);

void gqd2qd(gqd_real* gqd_data, qd_real* qd_data, const unsigned int numElement);

/* qd_real <-> gtd_real bridges: lets the gtd_real GPU path be exercised
 * against a qd_real CPU reference when no dedicated TD CPU library is
 * configured (HAVE_TDLIB undefined).  The 4th qd component is dropped
 * on the way down and zero-filled on the way back. */
void qd2gqd(qd_real* qd_data, gtd_real* gtd_data, const unsigned int numElement);

void gqd2qd(gtd_real* gtd_data, qd_real* qd_data, const unsigned int numElement);

int checkTwoArray( const dd_real* gold, const dd_real* ref, const int numElement );

int checkTwoArray( const qd_real* gold, const qd_real* ref, const int numElement );

/* ---- float-based bridges (gds_real / gts_real / gqs_real) ----
 * Convert between the dd_real/qd_real CPU reference and the float-based
 * GPU containers.  The float layers have ~14 / ~21 / ~28 decimal digits
 * of precision respectively, so the reported relative error reflects
 * float-layer precision rather than the dd/qd reference. */
void qd2gqd(dd_real* dd_data, gds_real* gds_data, const unsigned int numElement);
void gqd2qd(gds_real* gds_data, dd_real* dd_data, const unsigned int numElement);

void qd2gqd(qd_real* qd_data, gts_real* gts_data, const unsigned int numElement);
void gqd2qd(gts_real* gts_data, qd_real* qd_data, const unsigned int numElement);

void qd2gqd(qd_real* qd_data, gqs_real* gqs_data, const unsigned int numElement);
void gqd2qd(gqs_real* gqs_data, qd_real* qd_data, const unsigned int numElement);

#ifdef HAVE_TDLIB
void randArray(TDLIB_TYPE* data, const unsigned numElement,
               TDLIB_TYPE low, TDLIB_TYPE up, int seed = 0);

void qd2gqd(TDLIB_TYPE* td_data, gtd_real* gtd_data, const unsigned int numElement);

void gqd2qd(gtd_real* gtd_data, TDLIB_TYPE* td_data, const unsigned int numElement);

int checkTwoArray( const TDLIB_TYPE* gold, const TDLIB_TYPE* ref, const int numElement );
#endif

/* timing functions */
inline double getSec( struct timeval tvStart, struct timeval tvEnd ) {
        double tStart = (double)tvStart.tv_sec + 1e-6*tvStart.tv_usec;
        double tEnd = (double)tvEnd.tv_sec + 1e-6*tvEnd.tv_usec;
        return (tEnd - tStart);
}

#define INIT_TIMER struct timeval start, end;
#define START_TIMER gettimeofday(&start, NULL);
#define END_TIMER   gettimeofday(&end, NULL);
#define PRINT_TIMER_SEC(msg) printf("*** %s: %.3f sec ***\n", msg, getSec(start, end));

#endif /* __GQD_TEST_UTIL_H__ */


