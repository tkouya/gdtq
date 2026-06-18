/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef _CUDA_HEADER_CU_
#define _CUDA_HEADER_CU_

#include <stdio.h>
#include <stdlib.h>



// cutilCheckMsg
//#include "helper_cuda.h"
#define cutilCheckMsg getLastCudaError

/* CUDA 13 replacement for the legacy CUDA Samples helper_cuda.h
 * `getLastCudaError` macro: print the last CUDA error and abort. */
static inline void getLastCudaError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error after %s: %s\n",
		        msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

/*
 * In recent CUDA toolkits (CUDA 12+ and especially CUDA 13), cudaError_t
 * return values are marked [[nodiscard]], so silently ignoring them is
 * reported as a warning. Wrap the call in an explicit (void) cast so the
 * programmer's intent to discard the status is clear to the compiler.
 * If real error checking is desired, replace these macros with an
 * error-checking helper function.
 */
#define CUT_SAFE_CALL(function)  ((void)(function))
#define CUDA_SAFE_CALL(function) ((void)(function))
#define cutilSafeCall(function)  ((void)(function))

// cuCreateTimer
//#include "helper_timer.h"

/* kernel macros */
#define NUM_TOTAL_THREAD (gridDim.x * blockDim.x)
#define GLOBAL_THREAD_OFFSET (blockDim.x * blockIdx.x + threadIdx.x)

/** macro utility */
#define GPUMALLOC(D_DATA, MEM_SIZE) cutilSafeCall(cudaMalloc(D_DATA, MEM_SIZE))
#define TOGPU(D_DATA, H_DATA, MEM_SIZE) cutilSafeCall(cudaMemcpy(D_DATA, H_DATA, MEM_SIZE, cudaMemcpyHostToDevice))
#define FROMGPU( H_DATA, D_DATA, MEM_SIZE ) cutilSafeCall(cudaMemcpy( H_DATA, D_DATA, MEM_SIZE, cudaMemcpyDeviceToHost))
#define GPUTOGPU( DEST, SRC, MEM_SIZE ) cutilSafeCall(cudaMemcpy( DEST, SRC, MEM_SIZE, cudaMemcpyDeviceToDevice ))
#define GPUFREE( MEM ) cutilSafeCall(cudaFree(MEM));

/* CUDA 13 replacement for the legacy CUDA Samples helper_timer.h
 * StopWatchInterface.  Implemented in terms of CUDA events.  Exposed
 * as a struct so existing code that does `StopWatchInterface *timer;`
 * compiles unchanged. */
struct StopWatchInterface {
	cudaEvent_t start;
	cudaEvent_t stop;
};

static inline void startTimer(StopWatchInterface **timer)
{
	*timer = new StopWatchInterface();
	cudaEventCreate(&((*timer)->start));
	cudaEventCreate(&((*timer)->stop));
	cudaEventRecord((*timer)->start, 0);
}

static inline float endTimer(StopWatchInterface **timer, const char *title)
{
	cudaEventRecord((*timer)->stop, 0);
	cudaEventSynchronize((*timer)->stop);
	float ms = 0.0f;
	cudaEventElapsedTime(&ms, (*timer)->start, (*timer)->stop);
	printf("*** %s processing time: %.3f sec ***\n", title, ms / 1000.0f);
	cudaEventDestroy((*timer)->start);
	cudaEventDestroy((*timer)->stop);
	delete *timer;
	*timer = NULL;
	return ms;
}

#endif // _CUDA_HEADER_CU_
