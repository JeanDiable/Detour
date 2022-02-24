
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <cufft.h>

#include "kernels.cuh";
#include "reg.h"
#include <stdio.h>
#include <stdlib.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <ctime>


__global__ void phaseCorr(Complex* in, Complex* templ, Complex* dest)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	//a.*b^/abs(a)/abs(b)

	Complex a = in[i];
	Complex b = templ[i];
	float newx = a.x*b.x + a.y*b.y;
	float newy = a.x*b.y - a.y*b.x;
	float l = sqrt(newx*newx + newy*newy) + 0.000001f;
	dest[i].x = newx / l;
	dest[i].y = newy / l;
}

__global__ void pickMax(Complex* ffted, float* out)
{
	int id = blockIdx.x * (algo_sz / 2) + threadIdx.x;
	float t2 = ffted[id].x*ffted[id].x + ffted[id].y*ffted[id].y;

	__shared__ float values[5];
#pragma unroll
	for (int offset = 16; offset > 0; offset /= 2)
		t2 = max(t2, __shfl_down_sync(0xffffffff, t2, offset));
	//160 -> 5
	if (threadIdx.x%warpSize == 0)
		values[threadIdx.x / warpSize] = 0;
	__syncthreads();

	if (threadIdx.x != 0 && blockIdx.x != 0) return;
	// x32
	if (threadIdx.x == 0) {
#pragma unroll
		for (int i = 1; i < 5; ++i)
			t2 = max(t2, values[i]);
		out[blockIdx.x] = t2;
	}
	// x16

}

extern "C" void initThetaFinding(ThetaFunStruct& tf_struct)
{
	cufftPlan2d(&tf_struct.fftPlan, theta_num, algo_sz / 2, CUFFT_C2C);
	cufftSetStream(tf_struct.fftPlan, get_stream());
	cudaMalloc(&tf_struct.fftAbs, theta_num * sizeof(float));
	cudaMalloc((void**)&tf_struct.pc_dest, theta_num*algo_sz / 2 * sizeof(Complex));
}

__global__ void toComplex(float* what, Complex* fftTmp)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	fftTmp[i].x = what[i];
	fftTmp[i].y = 0;
}


extern "C" void computeRotFFT(ThetaFunStruct& tf_struct, float* roted, Complex* &target)
{
	//i1p/i2p -> i1pf, i2pf.
	toComplex << <algo_sz / 2, theta_num, 0, get_stream()>> >(roted, target);
	cufftExecC2C(tf_struct.fftPlan, target, target, CUFFT_FORWARD);
}

inline int cmpfunc(const void * a, const void * b) {
	return -(**(float**)a - **(float**)b >= 0 ? 1 : -1);
}

//#include <algorithm>    // std::nth_element, std::random_shuffle

extern "C" void thetaFinding(ThetaFunStruct& tf_struct, Complex* i1pf, Complex* i2pf)
{
	phaseCorr << <algo_sz / 2, theta_num, 0, get_stream() >> > (i2pf, i1pf, tf_struct.pc_dest);
	cufftExecC2C(tf_struct.fftPlan, tf_struct.pc_dest, tf_struct.pc_dest, CUFFT_FORWARD);
	pickMax << <theta_num, 160, 0, get_stream() >> >(tf_struct.pc_dest, tf_struct.fftAbs); //0.39ms

	float peaks[theta_num];
	float* idx[theta_num];
	for (int i = 0; i < theta_num; ++i)
		idx[i] = &peaks[i];
	cudaMemcpyAsync(peaks, tf_struct.fftAbs, theta_num * sizeof(float), cudaMemcpyDeviceToHost, get_stream());
	cudaStreamSynchronize(get_stream());
	qsort(idx, theta_num, sizeof(float*), cmpfunc);

	for (int i = 0; i < rot_check_num; ++i) 
		tf_struct.theta_results[i] = (idx[i] - peaks);

	float e = 0, e2 = 0;
	for (int i = 0; i<theta_num; ++i)
	{
		e += peaks[i];
		e2 += peaks[i] * peaks[i];
	}
	e /= theta_num;
	e2 /= theta_num;

	tf_struct.tconf = (*idx[0] - e) / sqrt(e2 - e*e);
	tf_struct.tconf_m = (*idx[rot_check_num - 1] - e) / sqrt(e2 - e*e);
}