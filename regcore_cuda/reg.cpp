#include <stdio.h>
#include <stdlib.h>
#include <mutex>
#include <cuda_runtime.h>
#include "kernels.cuh"
#include <helper_cuda.h>

cudaStream_t get_stream()
{
	static int ptr = 0;
	static thread_local unsigned int myPtr = -1;
	static cudaStream_t streams[16];
	static std::mutex mtx;
	if (myPtr == -1)
	{
		mtx.lock();
		myPtr = ptr;
		printf("thread 0x%x create No.%d cuda stream.\n", std::this_thread::get_id(), ptr);
		checkCudaErrors(cudaStreamCreate(&streams[ptr]));
		ptr += 1;
		mtx.unlock();
	}
	return streams[myPtr];
}
