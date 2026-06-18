

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <omp.h>
#include "test_util.h"

using namespace std;

/* `tdrand()` is provided by the user's TD CPU library (declared in
 * TDLIB_HEADER, e.g. <qd/td_real.h>).  No local shim is needed. */

void randArray(dd_real* data, const unsigned numElement, 
               dd_real low, dd_real high, int seed) {

	assert(high > low);
	srand(seed);
	dd_real band = high - low;

	for(unsigned int i = 0; i < numElement; i++) {
		data[i] = low + ddrand()*band;
	}	
}


void randArray(qd_real* data, const unsigned numElement,
               qd_real low, qd_real high, int seed) {

        assert(high > low);
        srand(seed);
        qd_real band = high - low;

        for(unsigned int i = 0; i < numElement; i++) {
                data[i] = low + qdrand()*band;
        }
}

void qd2gqd(dd_real* dd_data, gdd_real* gdd_data, const unsigned int numElement) {
	for(unsigned int i = 0; i < numElement; i++) {
		gdd_data[i].x = dd_data[i].x[0];
		gdd_data[i].y = dd_data[i].x[1];
	}	
}


void qd2gqd(qd_real* qd_data, gqd_real* gqd_data, const unsigned int numElement) {
        for(unsigned int i = 0; i < numElement; i++) {
                gqd_data[i].x = qd_data[i].x[0];
                gqd_data[i].y = qd_data[i].x[1];
		gqd_data[i].z = qd_data[i].x[2];
		gqd_data[i].w = qd_data[i].x[3];
        }
}


void gqd2qd(gdd_real* gdd_data, dd_real* dd_data, const unsigned int numElement) {
        for(unsigned int i = 0; i < numElement; i++) {
                dd_data[i].x[0] = gdd_data[i].x;
                dd_data[i].x[1] = gdd_data[i].y;
        } 
}

void gqd2qd(gqd_real* gqd_data, qd_real* qd_data, const unsigned int numElement) {
        for(unsigned int i = 0; i < numElement; i++) {
                qd_data[i].x[0] = gqd_data[i].x;
                qd_data[i].x[1] = gqd_data[i].y;
		qd_data[i].x[2] = gqd_data[i].z;
		qd_data[i].x[3] = gqd_data[i].w;
        }
}


/* ---- float-based bridges (gds_real / gts_real / gqs_real) ----
 * Each conversion uses the staircase pattern: round the running double
 * residual down to a float, subtract that float (in double arithmetic)
 * and continue to the next float limb.  The reverse direction simply
 * widens each float to a double slot in the dd_real/qd_real container.
 */
void qd2gqd(dd_real* dd_data, gds_real* gds_data, const unsigned int numElement) {
        for (unsigned int i = 0; i < numElement; i++) {
                float f0 = (float)dd_data[i].x[0];
                double r  = (dd_data[i].x[0] - (double)f0) + dd_data[i].x[1];
                float f1 = (float)r;
                gds_data[i].x = f0;
                gds_data[i].y = f1;
        }
}

void gqd2qd(gds_real* gds_data, dd_real* dd_data, const unsigned int numElement) {
        for (unsigned int i = 0; i < numElement; i++) {
                dd_data[i].x[0] = (double)gds_data[i].x;
                dd_data[i].x[1] = (double)gds_data[i].y;
        }
}

void qd2gqd(qd_real* qd_data, gts_real* gts_data, const unsigned int numElement) {
        for (unsigned int i = 0; i < numElement; i++) {
                float f0 = (float)qd_data[i].x[0];
                double r0 = (qd_data[i].x[0] - (double)f0) + qd_data[i].x[1];
                float f1 = (float)r0;
                double r1 = (r0 - (double)f1) + qd_data[i].x[2];
                float f2 = (float)r1;
                gts_data[i].x = f0;
                gts_data[i].y = f1;
                gts_data[i].z = f2;
        }
}

void gqd2qd(gts_real* gts_data, qd_real* qd_data, const unsigned int numElement) {
        for (unsigned int i = 0; i < numElement; i++) {
                qd_data[i].x[0] = (double)gts_data[i].x;
                qd_data[i].x[1] = (double)gts_data[i].y;
                qd_data[i].x[2] = (double)gts_data[i].z;
                qd_data[i].x[3] = 0.0;
        }
}

void qd2gqd(qd_real* qd_data, gqs_real* gqs_data, const unsigned int numElement) {
        for (unsigned int i = 0; i < numElement; i++) {
                float f0 = (float)qd_data[i].x[0];
                double r0 = (qd_data[i].x[0] - (double)f0) + qd_data[i].x[1];
                float f1 = (float)r0;
                double r1 = (r0 - (double)f1) + qd_data[i].x[2];
                float f2 = (float)r1;
                double r2 = (r1 - (double)f2) + qd_data[i].x[3];
                float f3 = (float)r2;
                gqs_data[i].x = f0;
                gqs_data[i].y = f1;
                gqs_data[i].z = f2;
                gqs_data[i].w = f3;
        }
}

void gqd2qd(gqs_real* gqs_data, qd_real* qd_data, const unsigned int numElement) {
        for (unsigned int i = 0; i < numElement; i++) {
                qd_data[i].x[0] = (double)gqs_data[i].x;
                qd_data[i].x[1] = (double)gqs_data[i].y;
                qd_data[i].x[2] = (double)gqs_data[i].z;
                qd_data[i].x[3] = (double)gqs_data[i].w;
        }
}


/* qd_real -> gtd_real: drop the 4th limb. The qd CPU reference will be
 * slightly more accurate than the gtd GPU result, so the relative error
 * reported by checkTwoArray reflects the truncation to triple-double. */
void qd2gqd(qd_real* qd_data, gtd_real* gtd_data, const unsigned int numElement) {
        for(unsigned int i = 0; i < numElement; i++) {
                gtd_data[i].x = qd_data[i].x[0];
                gtd_data[i].y = qd_data[i].x[1];
                gtd_data[i].z = qd_data[i].x[2];
        }
}

void gqd2qd(gtd_real* gtd_data, qd_real* qd_data, const unsigned int numElement) {
        for(unsigned int i = 0; i < numElement; i++) {
                qd_data[i].x[0] = gtd_data[i].x;
                qd_data[i].x[1] = gtd_data[i].y;
                qd_data[i].x[2] = gtd_data[i].z;
                qd_data[i].x[3] = 0.0;
        }
}


int checkTwoArray( const dd_real* gold, const dd_real* ref, const int numElement ) {
        dd_real maxRelError = 0.0;
        dd_real avgRelError = 0.0;
        int maxId = 0;

        for( int i = 0; i < numElement; i++ ) {
                dd_real relError = abs((gold[i] - ref[i])/gold[i]);
                avgRelError += (relError/numElement);
                if( relError > maxRelError ) {
                        maxRelError = relError;
                        maxId = i;
                }
        }

        cout << "abs. of max. relative error: " << maxRelError << endl;
        cout << "abs. of avg. relative error: " << avgRelError << endl;
        if( maxRelError > 0.0 ) {
                cout << "max. relative error elements" << endl;
                cout << "i = " << maxId << endl;
                cout << "gold = " << gold[maxId].to_string() << endl;
                printf("Components(%.16e, %.16e)\n", gold[maxId].x[0], gold[maxId].x[1]);
                cout << "ref  = " << ref[maxId].to_string() << endl;
                printf("Components(%.16e, %.16e)\n", ref[maxId].x[0], ref[maxId].x[1]);

        } else {
                cout << "a sample:" << endl;
                const int i = rand()%numElement;
                cout << "i = " << i << endl;
                cout << "gold = " << gold[i].to_string() << endl;
                printf("Components(%.16e, %.16e)\n", gold[maxId].x[0], gold[maxId].x[1]);
                cout << "ref  = " << ref[i].to_string() << endl;
                printf("Components(%.16e, %.16e)\n", ref[maxId].x[0], ref[maxId].x[1]);
                maxId = i;
        }

        return maxId;
}


#ifdef HAVE_TDLIB
void randArray(TDLIB_TYPE* data, const unsigned numElement,
               TDLIB_TYPE low, TDLIB_TYPE high, int seed) {

	assert(high > low);
	srand(seed);
	TDLIB_TYPE band = high - low;

	for (unsigned int i = 0; i < numElement; i++) {
		data[i] = low + tdrand() * band;
	}
}

void qd2gqd(TDLIB_TYPE* td_data, gtd_real* gtd_data, const unsigned int numElement) {
	for (unsigned int i = 0; i < numElement; i++) {
		gtd_data[i].x = td_data[i].x[0];
		gtd_data[i].y = td_data[i].x[1];
		gtd_data[i].z = td_data[i].x[2];
	}
}

void gqd2qd(gtd_real* gtd_data, TDLIB_TYPE* td_data, const unsigned int numElement) {
	for (unsigned int i = 0; i < numElement; i++) {
		td_data[i].x[0] = gtd_data[i].x;
		td_data[i].x[1] = gtd_data[i].y;
		td_data[i].x[2] = gtd_data[i].z;
	}
}

int checkTwoArray( const TDLIB_TYPE* gold, const TDLIB_TYPE* ref, const int numElement ) {
	TDLIB_TYPE maxRelError = 0.0;
	TDLIB_TYPE avgRelError = 0.0;
	int maxId = 0;

	for (int i = 0; i < numElement; i++) {
		TDLIB_TYPE relError = abs((gold[i] - ref[i]) / gold[i]);
		avgRelError += (relError / numElement);
		if (relError > maxRelError) {
			maxRelError = relError;
			maxId = i;
		}
	}

	cout << "abs. of max. relative error: " << maxRelError << endl;
	cout << "abs. of avg. relative error: " << avgRelError << endl;
	if (maxRelError > 0.0) {
		cout << "max. relative error elements" << endl;
		cout << "i = " << maxId << endl;
		cout << "gold = " << (gold[maxId]).to_string() << endl;
		cout << "ref  = " << (ref[maxId]).to_string() << endl;
	} else {
		cout << "a sample:" << endl;
		const int i = rand() % numElement;
		cout << "i = " << i << endl;
		cout << "gold = " << (gold[i]).to_string() << endl;
		cout << "ref  = " << (ref[i]).to_string() << endl;
		maxId = i;
	}

	return maxId;
}
#endif /* HAVE_TDLIB */


int checkTwoArray( const qd_real* gold, const qd_real* ref, const int numElement ) {
        qd_real maxRelError = 0.0;
        qd_real avgRelError = 0.0;
        int maxId = 0;

        for( int i = 0; i < numElement; i++ ) {
                qd_real relError = abs((gold[i] - ref[i])/gold[i]);
                avgRelError += (relError/numElement);
                if( relError > maxRelError ) {
                        maxRelError = relError;
                        maxId = i;
                }
        }

        cout << "abs. of max. relative error: " << maxRelError << endl;
        cout << "abs. of avg. relative error: " << avgRelError << endl;
        if( maxRelError > 0.0 ) {
                cout << "max. relative error elements" << endl;
                cout << "i = " << maxId << endl;
                cout << "gold = " << (gold[maxId]).to_string() << endl;
                cout << "ref  = " << (ref[maxId]).to_string() << endl;
                cout << "rel. error = " << (abs((gold[maxId] - ref[maxId])/gold[maxId])).to_string() << endl;
        } else {
                cout << "a sample:" << endl;
                const int i = rand()%numElement;
                cout << "i = " << i << endl;
                cout << "gold = " << (gold[i]).to_string() << endl;
                cout << "ref  = " << (ref[i]).to_string() << endl;
                maxId = i;
        }

        return maxId;
}
