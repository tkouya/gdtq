/*
 * CUDA 13: double4 / long4 / ulong4 / longlong4 / ulonglong4 are
 * deprecated in favour of the new _16a / _32a variants.  Since this
 * benchmark indirectly uses double4 through gqd_real and pulls in
 * <vector_types.h> through <cuda.h>, the suppression macro must be
 * defined *before any include* so that the deprecation attribute is
 * not attached to the vector types when the header is first processed.
 */
#ifndef __NV_NO_VECTOR_DEPRECATION_DIAG
#define __NV_NO_VECTOR_DEPRECATION_DIAG
#endif

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <qd/qd_real.h>
#include <qd/fpu.h>
#include <omp.h>
#include <cuda.h>
#include "test_util.h"
#include "test_common.h"
#include "gqdtest.h"

using namespace std;


/* general macro utilities */
#define FUNC_START_MSG printf("%s start ............................................\n", __func__);
#define FUNC_END_MSG   printf("%s done  ...........................................\n\n", __func__);


template<class c_t, class g_t>
void test_sqrt(const unsigned int numElement) {

        FUNC_START_MSG;

        c_t* dd_in = new c_t[numElement];
        c_t* gold_out = new c_t[numElement];
        c_t low = "0.0";
        c_t high = "1.0";
        randArray(dd_in, numElement, low, high);
        g_t* gdd_in = new g_t[numElement];
        g_t* gdd_out = new g_t[numElement];
        qd2gqd(dd_in, gdd_in, numElement);


        unsigned int numBlock = 128;
        unsigned int numThread = 128;
        device_math(gdd_in, numElement, gdd_out, SQRT, numBlock, numThread);
        c_t* gpu_out = new c_t[numElement];
        gqd2qd(gdd_out, gpu_out, numElement);

        INIT_TIMER;
        START_TIMER;
#pragma omp parallel for
	 for(unsigned int i = 0; i < numElement; i++) {
                gold_out[i] = sqrt(dd_in[i]);
        }
        END_TIMER;
        PRINT_TIMER_SEC("CPU sqrt");

        checkTwoArray(gold_out, gpu_out, numElement);

        delete[] dd_in;
        delete[] gold_out;
        delete[] gdd_in;
        delete[] gdd_out;
        delete[] gpu_out;

        FUNC_END_MSG;
}

template<class c_t, class g_t>
void test_exp(const unsigned int numElement) {

	FUNC_START_MSG;


        c_t* dd_in = new c_t[numElement];
        c_t* gold_out = new c_t[numElement];
        c_t low = "0.0";
        c_t high = "1.0";
        randArray(dd_in, numElement, low, high);
        g_t* gdd_in = new g_t[numElement];
        g_t* gdd_out = new g_t[numElement];
        qd2gqd(dd_in, gdd_in, numElement);


        unsigned int numBlock = 128;
        unsigned int numThread = 128;
        device_math(gdd_in, numElement, gdd_out, EXP, numBlock, numThread);
        c_t* gpu_out = new c_t[numElement];
        gqd2qd(gdd_out, gpu_out, numElement);

	INIT_TIMER;
	START_TIMER;
#pragma omp parallel for
	for(unsigned int i = 0; i < numElement; i++) {
		gold_out[i] = exp(dd_in[i]);
	}
	END_TIMER;
	PRINT_TIMER_SEC("CPU exp");
	
	checkTwoArray(gold_out, gpu_out, numElement);

	delete[] dd_in;
	delete[] gold_out;
	delete[] gdd_in;
	delete[] gdd_out;
	delete[] gpu_out;

	FUNC_END_MSG;
}


template<class c_t, class g_t>
void test_log(const unsigned int numElement) {

        FUNC_START_MSG;

        c_t* dd_in = new c_t[numElement];
        c_t* gold_out = new c_t[numElement];
        c_t low = "0.0";
        c_t high = "1.0";
        randArray(dd_in, numElement, low, high);
        g_t* gdd_in = new g_t[numElement];
        g_t* gdd_out = new g_t[numElement];
        qd2gqd(dd_in, gdd_in, numElement);


        unsigned int numBlock = 128;
        unsigned int numThread = 128;
        device_math(gdd_in, numElement, gdd_out, LOG, numBlock, numThread);
        c_t* gpu_out = new c_t[numElement];
        gqd2qd(gdd_out, gpu_out, numElement);


        INIT_TIMER;
        START_TIMER;
#pragma omp parallel for
        for(unsigned int i = 0; i < numElement; i++) {
                gold_out[i] = log(dd_in[i]);
        }
        END_TIMER;
        PRINT_TIMER_SEC("CPU log");

        checkTwoArray(gold_out, gpu_out, numElement);

        delete[] dd_in;
        delete[] gold_out;
        delete[] gdd_in;
        delete[] gdd_out;
        delete[] gpu_out;

        FUNC_END_MSG;
}

template<class c_t, class g_t>
void test_sin(const unsigned int numElement) {

        FUNC_START_MSG;

        c_t* dd_in = new c_t[numElement];
        c_t* gold_out = new c_t[numElement];
        c_t low = "0.0";
        c_t high = "1.0";
        randArray(dd_in, numElement, low, high);
        g_t* gdd_in = new g_t[numElement];
        g_t* gdd_out = new g_t[numElement];
        qd2gqd(dd_in, gdd_in, numElement);


        unsigned int numBlock = 128;
        unsigned int numThread = 128;
        device_math(gdd_in, numElement, gdd_out, SIN, numBlock, numThread);
        c_t* gpu_out = new c_t[numElement];
        gqd2qd(gdd_out, gpu_out, numElement);


        INIT_TIMER;
        START_TIMER;
#pragma omp parallel for
        for(unsigned int i = 0; i < numElement; i++) {
                gold_out[i] = sin(dd_in[i]);
        }
        END_TIMER;
        PRINT_TIMER_SEC("CPU sin");

        checkTwoArray(gold_out, gpu_out, numElement);

        delete[] dd_in;
        delete[] gold_out;
        delete[] gdd_in;
        delete[] gdd_out;
        delete[] gpu_out;

        FUNC_END_MSG;
}

template<class c_t, class g_t>
void test_cos(unsigned int numElement) {

        FUNC_START_MSG;

        c_t* dd_in = new c_t[numElement];
        c_t* gold_out = new c_t[numElement];
        c_t low = "0.0";
        c_t high = "1.0";
        randArray(dd_in, numElement, low, high);
        g_t* gdd_in = new g_t[numElement];
        g_t* gdd_out = new g_t[numElement];
        qd2gqd(dd_in, gdd_in, numElement);


        unsigned int numBlock = 128;
        unsigned int numThread = 128;
        device_math(gdd_in, numElement, gdd_out, COS, numBlock, numThread);
        c_t* gpu_out = new c_t[numElement];
        gqd2qd(gdd_out, gpu_out, numElement);


        INIT_TIMER;
        START_TIMER;
#pragma omp parallel for
        for(unsigned int i = 0; i < numElement; i++) {
                gold_out[i] = cos(dd_in[i]);
        }
        END_TIMER;
        PRINT_TIMER_SEC("CPU cos");

        checkTwoArray(gold_out, gpu_out, numElement);

        delete[] dd_in;
        delete[] gold_out;
        delete[] gdd_in;
        delete[] gdd_out;
        delete[] gpu_out;

        FUNC_END_MSG;
}


template<class c_t, class g_t>
void test_tan(unsigned int numElement) {

        FUNC_START_MSG;

        c_t* dd_in = new c_t[numElement];
        c_t* gold_out = new c_t[numElement];
        c_t low = "0.0";
        c_t high = "1.0";
        randArray(dd_in, numElement, low, high);
        g_t* gdd_in = new g_t[numElement];
        g_t* gdd_out = new g_t[numElement];
        qd2gqd(dd_in, gdd_in, numElement);


        unsigned int numBlock = 128;
        unsigned int numThread = 128;
        device_math(gdd_in, numElement, gdd_out, TAN, numBlock, numThread);
        c_t* gpu_out = new c_t[numElement];
        gqd2qd(gdd_out, gpu_out, numElement);


        INIT_TIMER;
        START_TIMER;
#pragma omp parallel for
        for(unsigned int i = 0; i < numElement; i++) {
                gold_out[i] = tan(dd_in[i]);
        }
        END_TIMER;
        PRINT_TIMER_SEC("CPU tan");

        checkTwoArray(gold_out, gpu_out, numElement);

        delete[] dd_in;
        delete[] gold_out;
        delete[] gdd_in;
        delete[] gdd_out;
        delete[] gpu_out;

        FUNC_END_MSG;
}

/*
template<class c_t, class g_t>
void test_atan(unsigned int numElement) {

        FUNC_START_MSG;

        c_t* dd_in = new c_t[numElement];
        c_t* gold_out = new c_t[numElement];
        c_t low = "0.0";
        c_t high = "1.0";
        randArray(dd_in, numElement, low, high);
        g_t* gdd_in = new g_t[numElement];
        g_t* gdd_out = new g_t[numElement];
        qd2gqd(dd_in, gdd_in, numElement);


        unsigned int numBlock = 128;
        unsigned int numThread = 128;
        device_math(gdd_in, numElement, gdd_out, ATAN, numBlock, numThread);
        c_t* gpu_out = new c_t[numElement];
        gqd2qd(gdd_out, gpu_out, numElement);


        INIT_TIMER;
        START_TIMER;
#pragma omp parallel for
        for(unsigned int i = 0; i < numElement; i++) {
                gold_out[i] = atan(dd_in[i]);
        }
        END_TIMER;
        PRINT_TIMER_SEC("CPU tan");

        checkTwoArray(gold_out, gpu_out, numElement);

        delete[] dd_in;
        delete[] gold_out;
        delete[] gdd_in;
        delete[] gdd_out;
        delete[] gpu_out;

        FUNC_END_MSG;
}
*/


template<class c_t, class g_t>
void test_add(const unsigned int numElement) {

        FUNC_START_MSG;

        c_t* dd_in1 = new c_t[numElement];
	c_t* dd_in2 = new c_t[numElement];
        c_t* gold_out = new c_t[numElement];
        c_t low = "-1.0";
        c_t high = "1.0";
        randArray(dd_in1, numElement, low, high, 777);
	randArray(dd_in2, numElement, low, high, 888);
        g_t* gdd_in1 = new g_t[numElement];
	g_t* gdd_in2 = new g_t[numElement];
       	g_t* gdd_out = new g_t[numElement];
        qd2gqd(dd_in1, gdd_in1, numElement);
	qd2gqd(dd_in2, gdd_in2, numElement);


        unsigned int numBlock = 128;
        unsigned int numThread = 128;
        device_basic(gdd_in1, gdd_in2, gdd_out, numElement, ADD, numBlock, numThread);
        c_t* gpu_out = new c_t[numElement];
        gqd2qd(gdd_out, gpu_out, numElement);

        INIT_TIMER;
        START_TIMER;
#pragma omp parallel for
        for(unsigned int i = 0; i < numElement; i++) {
                gold_out[i] = dd_in1[i] + dd_in2[i];
        }
        END_TIMER;
        PRINT_TIMER_SEC("CPU add");

        checkTwoArray(gold_out, gpu_out, numElement);

        delete[] dd_in1;
	delete[] dd_in2;
        delete[] gold_out;
        delete[] gdd_in1;
	delete[] gdd_in2;
        delete[] gdd_out;
        delete[] gpu_out;

        FUNC_END_MSG;
}

template<class c_t, class g_t>
void test_mul(const unsigned int numElement) {

        FUNC_START_MSG;

        c_t* dd_in1 = new c_t[numElement];
        c_t* dd_in2 = new c_t[numElement];
        c_t* gold_out = new c_t[numElement];
        c_t low = "-1.0";
        c_t high = "1.0";
        randArray(dd_in1, numElement, low, high, 777);
        randArray(dd_in2, numElement, low, high, 888);
        g_t* gdd_in1 = new g_t[numElement];
        g_t* gdd_in2 = new g_t[numElement];
        g_t* gdd_out = new g_t[numElement];
        qd2gqd(dd_in1, gdd_in1, numElement);
        qd2gqd(dd_in2, gdd_in2, numElement);


        unsigned int numBlock = 128;
        unsigned int numThread = 128;
        device_basic(gdd_in1, gdd_in2, gdd_out, numElement, MUL, numBlock, numThread);
        c_t* gpu_out = new c_t[numElement];
        gqd2qd(gdd_out, gpu_out, numElement);

        INIT_TIMER;
        START_TIMER;
#pragma omp parallel for
        for(unsigned int i = 0; i < numElement; i++) {
                gold_out[i] = dd_in1[i] * dd_in2[i];
        }
        END_TIMER;
        PRINT_TIMER_SEC("CPU mul");

        checkTwoArray(gold_out, gpu_out, numElement);

        delete[] dd_in1;
        delete[] dd_in2;
        delete[] gold_out;
        delete[] gdd_in1;
        delete[] gdd_in2;
        delete[] gdd_out;
        delete[] gpu_out;

        FUNC_END_MSG;
}

template<class c_t, class g_t>
void test_div(const unsigned int numElement) {

        FUNC_START_MSG;

        c_t* dd_in1 = new c_t[numElement];
        c_t* dd_in2 = new c_t[numElement];
        c_t* gold_out = new c_t[numElement];
        c_t low = "-1.0";
        c_t high = "1.0";
        randArray(dd_in1, numElement, low, high, 777);
        randArray(dd_in2, numElement, low, high, 888);
        g_t* gdd_in1 = new g_t[numElement];
        g_t* gdd_in2 = new g_t[numElement];
        g_t* gdd_out = new g_t[numElement];
        qd2gqd(dd_in1, gdd_in1, numElement);
        qd2gqd(dd_in2, gdd_in2, numElement);


        unsigned int numBlock = 128;
        unsigned int numThread = 128;
        device_basic(gdd_in1, gdd_in2, gdd_out, numElement, DIV, numBlock, numThread);
        c_t* gpu_out = new c_t[numElement];
        gqd2qd(gdd_out, gpu_out, numElement);

        INIT_TIMER;
        START_TIMER;
#pragma omp parallel for
        for(unsigned int i = 0; i < numElement; i++) {
                gold_out[i] = dd_in1[i] / dd_in2[i];
        }
        END_TIMER;
        PRINT_TIMER_SEC("CPU div");

        checkTwoArray(gold_out, gpu_out, numElement);

        delete[] dd_in1;
        delete[] dd_in2;
        delete[] gold_out;
        delete[] gdd_in1;
        delete[] gdd_in2;
        delete[] gdd_out;
        delete[] gpu_out;

        FUNC_END_MSG;
}


int main(int argc, char** argv) {
	const int omp_num_thread = 16;
	omp_set_num_threads(omp_num_thread);
	printf("omp_num_thread = %d\n", omp_num_thread);

	unsigned int old_cw;
	fpu_fix_start(&old_cw);

	unsigned int numElement = 1000000;

	
	printf("******************** double-double precision *********************\n");
	GDDStart();
	printf("numElement = %d\n", numElement);
	test_add<dd_real, gdd_real>(numElement);
	test_mul<dd_real, gdd_real>(numElement);
	test_div<dd_real, gdd_real>(numElement);
	test_sqrt<dd_real, gdd_real>(numElement);
	test_exp<dd_real, gdd_real>(numElement);
	test_log<dd_real, gdd_real>(numElement);
	test_sin<dd_real, gdd_real>(numElement);
	test_tan<dd_real, gdd_real>(numElement);
	//test_atan<dd_real, gdd_real>(numElement);
	GDDEnd();

	printf("\n\n");

	printf("******************** quad-double precision *********************\n");
	GQDStart();
	printf("numElement = %d\n", numElement);
	test_add<qd_real, gqd_real>(numElement);
        test_mul<qd_real, gqd_real>(numElement);
        test_div<qd_real, gqd_real>(numElement);
        test_sqrt<qd_real, gqd_real>(numElement);
        test_exp<qd_real, gqd_real>(numElement);
        test_log<qd_real, gqd_real>(numElement);
        test_sin<qd_real, gqd_real>(numElement);
	test_tan<qd_real, gqd_real>(numElement);
	//test_atan<dd_real, gdd_real>(numElement);

	printf("\n\n");
	printf("******************** triple-double precision *********************\n");
	/* GTDStart() truncates the QD __constant__ tables that GQDStart()
	 * just populated, so it must run before GQDEnd() (whose
	 * cudaDeviceReset() would wipe them). */
	GTDStart();
	printf("numElement = %d\n", numElement);
#ifdef HAVE_TDLIB
	test_add<TDLIB_TYPE, gtd_real>(numElement);
	test_mul<TDLIB_TYPE, gtd_real>(numElement);
	test_div<TDLIB_TYPE, gtd_real>(numElement);
	test_sqrt<TDLIB_TYPE, gtd_real>(numElement);
	test_exp<TDLIB_TYPE, gtd_real>(numElement);
	test_log<TDLIB_TYPE, gtd_real>(numElement);
	test_sin<TDLIB_TYPE, gtd_real>(numElement);
	test_tan<TDLIB_TYPE, gtd_real>(numElement);
#else
	/* No dedicated CPU triple-double library configured; use qd_real as
	 * the CPU reference.  The error reported includes truncation from
	 * the 4th qd limb, so the gtd path is exercised end-to-end even
	 * though the comparison is conservative. */
	test_add<qd_real, gtd_real>(numElement);
	test_mul<qd_real, gtd_real>(numElement);
	test_div<qd_real, gtd_real>(numElement);
	test_sqrt<qd_real, gtd_real>(numElement);
	test_exp<qd_real, gtd_real>(numElement);
	test_log<qd_real, gtd_real>(numElement);
	test_sin<qd_real, gtd_real>(numElement);
	test_tan<qd_real, gtd_real>(numElement);
#endif
	GTDEnd();

	printf("\n\n");
	printf("******************** double-single (gds_real, ~14 digits) precision *********************\n");
	GDSStart();
	printf("numElement = %d\n", numElement);
	test_add<dd_real, gds_real>(numElement);
	test_mul<dd_real, gds_real>(numElement);
	test_div<dd_real, gds_real>(numElement);
	test_sqrt<dd_real, gds_real>(numElement);
	test_exp<dd_real, gds_real>(numElement);
	test_log<dd_real, gds_real>(numElement);
	test_sin<dd_real, gds_real>(numElement);
	test_tan<dd_real, gds_real>(numElement);
	GDSEnd();

	printf("\n\n");
	printf("******************** quad-single (gqs_real, ~28 digits) precision *********************\n");
	GQSStart();
	printf("numElement = %d\n", numElement);
	test_add<qd_real, gqs_real>(numElement);
	test_mul<qd_real, gqs_real>(numElement);
	test_div<qd_real, gqs_real>(numElement);
	test_sqrt<qd_real, gqs_real>(numElement);
	test_exp<qd_real, gqs_real>(numElement);
	test_log<qd_real, gqs_real>(numElement);
	test_sin<qd_real, gqs_real>(numElement);
	test_tan<qd_real, gqs_real>(numElement);

	printf("\n\n");
	printf("******************** triple-single (gts_real, ~21 digits) precision *********************\n");
	/* GTSStart() shares the device with the QS tables that GQSStart()
	 * just populated; it must run before any GxSEnd() calls cudaDeviceReset(). */
	GTSStart();
	printf("numElement = %d\n", numElement);
	test_add<qd_real, gts_real>(numElement);
	test_mul<qd_real, gts_real>(numElement);
	test_div<qd_real, gts_real>(numElement);
	test_sqrt<qd_real, gts_real>(numElement);
	test_exp<qd_real, gts_real>(numElement);
	test_log<qd_real, gts_real>(numElement);
	test_sin<qd_real, gts_real>(numElement);
	test_tan<qd_real, gts_real>(numElement);
	GTSEnd();

	fpu_fix_end(&old_cw);
	return EXIT_SUCCESS;
}

