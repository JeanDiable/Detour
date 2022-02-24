/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

// extensively modified by artheru

#include <device_launch_parameters.h>
#include "../reg.h"
#include "../kernels.cuh"
#if !defined CUDA_DISABLER

#include "common.hpp"
#include "functional.hpp"
#include "emulation.hpp"
#include "scan.hpp"
#include "reduce.hpp"
#include "saturate_cast.hpp"

#include <helper_cuda.h>


__global__ void
vectorAdd(const float* A, const float* B, float* C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = A[i] + B[i];
	}
}
int
test(void)
{
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	int numElements = 50000;
	size_t size = numElements * sizeof(float);
	printf("[Vector addition of %d elements]\n", numElements);

	// Allocate the host input vector A
	float* h_A = (float*)malloc(size);

	// Allocate the host input vector B
	float* h_B = (float*)malloc(size);

	// Allocate the host output vector C
	float* h_C = (float*)malloc(size);

	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}

	// Allocate the device input vector A
	float* d_A = NULL;
	err = cudaMalloc((void**)&d_A, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	float* d_B = NULL;
	err = cudaMalloc((void**)&d_B, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	float* d_C = NULL;
	err = cudaMalloc((void**)&d_C, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, numElements);
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i)
	{
		if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}

	printf("Test PASSED\n");

	// Free device global memory
	err = cudaFree(d_A);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_B);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_C);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);

	printf("Done\n");
	return 0;
}

char* cv::String::allocate(size_t len)
{
	size_t totalsize = len + 1; // alignSize(len + 1, (int)sizeof(int));
	int* data = (int*)malloc(totalsize + sizeof(int));
	data[0] = 1;
	cstr_ = (char*)(data + 1);
	len_ = len;
	cstr_[len] = 0;
	return cstr_;
}


void cv::String::deallocate()
{
	int* data = (int*)cstr_;
	len_ = 0;
	cstr_ = 0;

	if (data && 1 == CV_XADD(data - 1, -1))
	{
		free(data - 1);
	}
}
void cv::error(int _code, const cv::String& _err, const char* _func, const char* _file, int _line)
{
	//todo
}

using namespace cv::cuda;
using namespace cv::cuda::device;

namespace clahe
{
    __global__ void calcLutKernel(const PtrStepb src, PtrStepb lut,
                                  const int2 tileSize, const int tilesX,
                                  const int clipLimit, const float lutScale)
    {
        __shared__ int smem[512];

        const int tx = blockIdx.x;
        const int ty = blockIdx.y;
    	const int lane = threadIdx.x % 32;
        const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

        smem[tid] = 0;
        __syncthreads();

        for (int i = threadIdx.y; i < tileSize.y; i += blockDim.y)
        {
            const uchar* srcPtr = src.ptr(ty * tileSize.y + i) + tx * tileSize.x;
            for (int j = threadIdx.x; j < tileSize.x; j += blockDim.x)
            {
                const int data = srcPtr[j];
                atomicAdd(&smem[data], 1);
            }
        }
        __syncthreads();

        int tHistVal = smem[tid];

        __syncthreads();

        if (clipLimit > 0)
        {
            // clip histogram bar

            int clipped = 0;
            if (tHistVal > clipLimit)
            {
                clipped = tHistVal - clipLimit;
                tHistVal = clipLimit;
            }

            // find number of overall clipped samples

            reduce<256>(smem, clipped, tid, plus<int>());

            // broadcast evaluated value

            __shared__ int totalClipped;

            if (tid == 0)
                totalClipped = clipped;
            __syncthreads();

            // redistribute clipped samples evenly

            int redistBatch = totalClipped / 256;
            tHistVal += redistBatch;

            int residual = totalClipped - redistBatch * 256;
            if (tid < residual)
                ++tHistVal;
        }


		int val = tHistVal;
#pragma unroll
		for (int i = 1; i <= 16; i *= 2)
		{
			int n = __shfl_up_sync(0xffffffff, val, i);
			if (lane >= i)
				val += n;
		}
		smem[tid] = val;
		__syncthreads(); // now 32 batches are good.
    	
    	for (int i=1; i<8; ++i)
    	{
			smem[tid % 32 + i * 32] += smem[tid % 32 + (i - 1) * 32];
			__syncthreads();
    	}
    	
        const int lutVal = smem[tid];
		
        lut(ty * tilesX + tx, tid) = saturate_cast<uchar>(__float2int_rn(lutScale * lutVal*0.1));
    }

    void calcLut(PtrStepSzb src, PtrStepb lut, int tilesX, int tilesY, int2 tileSize, int clipLimit, float lutScale)
    {
        const dim3 block(32, 8);
        const dim3 grid(tilesX, tilesY);

        calcLutKernel<<<grid, block>>>(src, lut, tileSize, tilesX, clipLimit, lutScale);
		//test();
		
        checkCudaErrors( cudaDeviceSynchronize() );
    }

    __global__ void transformKernel(const PtrStepSzb src, PtrStepSzb dst, const PtrStepb lut, const int2 tileSize, const int tilesX, const int tilesY)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= src.cols || y >= src.rows)
            return;

        const float tyf = (static_cast<float>(y) / tileSize.y) - 0.5f;
        int ty1 = __float2int_rd(tyf);
        int ty2 = ty1 + 1;
        const float ya = tyf - ty1;
        ty1 = ::max(ty1, 0);
        ty2 = ::min(ty2, tilesY - 1);

        const float txf = (static_cast<float>(x) / tileSize.x) - 0.5f;
        int tx1 = __float2int_rd(txf);
        int tx2 = tx1 + 1;
        const float xa = txf - tx1;
        tx1 = ::max(tx1, 0);
        tx2 = ::min(tx2, tilesX - 1);

        const int srcVal = src(y, x);

        float res = 0;

        res += lut(ty1 * tilesX + tx1, srcVal) * ((1.0f - xa) * (1.0f - ya));
        res += lut(ty1 * tilesX + tx2, srcVal) * ((xa) * (1.0f - ya));
        res += lut(ty2 * tilesX + tx1, srcVal) * ((1.0f - xa) * (ya));
        res += lut(ty2 * tilesX + tx2, srcVal) * ((xa) * (ya));

        dst(y, x) = saturate_cast<uchar>(__float2int_rn(res));
    }

    void transform(PtrStepSzb src, PtrStepSzb dst, PtrStepb lut, int tilesX, int tilesY, int2 tileSize)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

//        cudaFuncSetCacheConfig(transformKernel, cudaFuncCachePreferL1);
        transformKernel<<<grid, block>>>(src, dst, lut, tileSize, tilesX, tilesY);

        //checkCudaErrors( cudaDeviceSynchronize() );
    }
}

#endif // CUDA_DISABLER

#define lut_wnd_num 20
#define lut_wnd_sz 16

extern "C" void initCLAHE(ImProcessStruct& ip_struct)
{
	cudaMalloc((void**)&ip_struct.lut, lut_wnd_num * lut_wnd_num * 256);
}



extern "C" void applyCLAHE(ImProcessStruct& ip_struct, unsigned char* _src, unsigned char* _dst)
{
	PtrStepSzb src(algo_sz, algo_sz, _src, algo_sz);
	PtrStepSzb dst(algo_sz, algo_sz, _dst, algo_sz);
	PtrStepSzb lut(lut_wnd_num * lut_wnd_num, 256, ip_struct.lut, 256);

	const int histSize = 256;
	const int tileSizeTotal = lut_wnd_sz*lut_wnd_sz;
	const float lutScale = static_cast<float>(histSize - 1) / tileSizeTotal;

	int clipLimit = static_cast<int>(2.2 * tileSizeTotal / histSize);
	clipLimit = std::max(clipLimit, 1);

	clahe::calcLut(src, lut, lut_wnd_num, lut_wnd_num, make_int2(lut_wnd_sz, lut_wnd_sz), clipLimit, lutScale);
	clahe::transform(src, dst, lut, lut_wnd_num, lut_wnd_num, make_int2(lut_wnd_sz, lut_wnd_sz));
}