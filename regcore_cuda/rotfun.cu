
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cufftXt.h>

#include <math.h>

#include "kernels.cuh";
#include "reg.h"
#include <stdio.h>
#include <stdlib.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <ctime>


#define SWAP(arr, i,j) if (arr[i]>arr[j]) {float tmp=arr[i];arr[i]=arr[j];arr[j]=tmp;}
__global__ void absrot(cudaTextureObject_t ffted, float* dest, ABSRotParm params) { //todo: tex2d + 
																					//return;
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (y >= theta_num) return;
	const int id = x + y*algo_sz / 2;

	//x[0~sz/2+1], y[0~theta_num]
	float rho = ((float)x) / (algo_sz / 2)*(algo_sz / 2 - 3) + 1;
	float bcos = params.baseCos[y] * rho;
	float bsin = params.baseSin[y] * rho;
	int bias = 0;
	float tmp = 0;

#pragma unroll
	float nums[rotfun_num];
	for (int i = 0; i < rotfun_num; ++i) {// todo: optimize this....
										  //xx~algo_sz/2+1, yy~algo_sz.
		float xx = (bcos);
		float yy = (bsin);
		if (yy < 0) yy += algo_sz;
		if (xx < 0) { xx = -xx;  yy = algo_sz - yy; };
		nums[i] = tex2D<float>(ffted, xx + bias + 0.5, yy + 0.5);

		bias += algo_sz / 2;
		float ncos = bcos*params.stepCos - bsin*params.stepSin;
		float nsin = bsin*params.stepCos + bcos*params.stepSin;
		bcos = ncos;
		bsin = nsin;
	}
	//~5 rots.
	SWAP(nums, 0, 1);
	SWAP(nums, 3, 4);
	SWAP(nums, 2, 4);
	SWAP(nums, 2, 3);
	SWAP(nums, 0, 3);
	SWAP(nums, 0, 2);
	SWAP(nums, 1, 4);
	SWAP(nums, 1, 3);
	SWAP(nums, 1, 2);

	dest[id] = (sqrt(nums[4] + nums[3]))*(1/(1+exp(-0.1f*(x-8.0f))));// nums[0] + nums[1] + nums[2] + nums[3] + nums[4];
}

__global__ void rotate_many(cudaTextureObject_t tex, float* dest,
	int sz, RotParams<rotfun_num> params) {
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	for (int n = 0; n < rotfun_num; ++n) {
		const float sx = x*params.cosTheta[n] - y*params.sinTheta[n] + params.biasX[n];
		const float sy = x*params.sinTheta[n] + y*params.cosTheta[n] + params.biasY[n];
		const int id = y * sz + x + n*algo_sz*algo_sz;

		const int rborder = 8;
		float m = tex2D<float>(tex, sx + 0.5, sy + 0.5);
		if (x < rborder) m = m / rborder*(x);
		if (y < rborder) m = m / rborder*(y);
		if (x >= algo_sz - rborder) m = m / rborder*(algo_sz - x);
		if (y >= algo_sz - rborder) m = m / rborder*(algo_sz - y);

		dest[id] = m;
	}
}

extern "C" void initRotFun(RotfunStruct& rot_struct)
{
	checkCudaErrors(cudaMalloc((void**)&rot_struct.interm, algo_sz*algo_sz * sizeof(float)*rotfun_num));
	checkCudaErrors(cudaMalloc((void**)&rot_struct.fft_out_data, algo_sz*(algo_sz / 2 + 1) * sizeof(Complex)*rotfun_num));

	int n[2] = { algo_sz, algo_sz };
	checkCudaErrors(cufftPlanMany(&rot_struct.fftPlan, 2, n,
		nullptr, 1, 0,
		nullptr, 1, 0,
		CUFFT_R2C, rotfun_num));
	cufftSetStream(rot_struct.fftPlan, get_stream());

	for (int i = 0; i < rotfun_num; ++i) {
		const float theta = 3.1415926 / 2 / (rotfun_num + 1) *i;
		rot_struct.params1.sinTheta[i] = sin(theta), rot_struct.params1.cosTheta[i] = cos(theta);
		rot_struct.params1.biasX[i] = -algo_sz / 2 * rot_struct.params1.cosTheta[i] + algo_sz / 2 * rot_struct.params1.sinTheta[i] + algo_sz / 2;
		rot_struct.params1.biasY[i] = -algo_sz / 2 * rot_struct.params1.sinTheta[i] - algo_sz / 2 * rot_struct.params1.cosTheta[i] + algo_sz / 2;
	}

	rot_struct.params2.stepSin = sin(-3.1415926 / 2 / (rotfun_num + 1));
	rot_struct.params2.stepCos = cos(3.1415926 / 2 / (rotfun_num + 1));
	for (int i = 0; i < theta_num; ++i) {
		const float theta = 3.1415926 / theta_num *i;
		rot_struct.params2.baseSin[i] = sin(theta);
		rot_struct.params2.baseCos[i] = cos(theta);
	}

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray* srcArray, *fftArray;
	cudaMallocArray(&srcArray, &channelDesc, algo_sz, algo_sz, cudaArraySurfaceLoadStore);
	cudaMallocArray(&fftArray, &channelDesc, (algo_sz / 2)*(rotfun_num), algo_sz + 1, cudaArraySurfaceLoadStore);

	cudaResourceDesc resDesc, fftResDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = srcArray;
	cudaCreateSurfaceObject(&rot_struct.pSurfObject, &resDesc);

	memset(&fftResDesc, 0, sizeof(fftResDesc));
	fftResDesc.resType = cudaResourceTypeArray;
	fftResDesc.res.array.array = fftArray;
	cudaCreateSurfaceObject(&rot_struct.pFFTSurfObject, &fftResDesc);

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	checkCudaErrors(cudaCreateTextureObject(&rot_struct.croppedTex, &resDesc, &texDesc, NULL));

	cudaTextureDesc texDesc2;
	memset(&texDesc2, 0, sizeof(texDesc2));
	texDesc2.readMode = cudaReadModeElementType;
	texDesc2.filterMode = cudaFilterModeLinear;
	texDesc2.addressMode[0] = cudaAddressModeBorder;
	texDesc2.addressMode[1] = cudaAddressModeBorder;
	checkCudaErrors(cudaCreateTextureObject(&rot_struct.FFTcroppedTex, &fftResDesc, &texDesc2, NULL));

}

__global__ void toAbs(Complex* src, cudaSurfaceObject_t surf) {
	//src: 161:320 => 160:321
	unsigned int n = blockIdx.z;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x; //1~algo_sz/2*rotfun_num
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y; //1~algo_sz

	int id = x % (algo_sz / 2) + y*(algo_sz / 2 + 1) + n*((algo_sz / 2 + 1)*algo_sz);

	float v = (src[id].x*src[id].x + src[id].y*src[id].y);

	surf2Dwrite(v, surf, (x + n*algo_sz / 2) * 4, y, cudaBoundaryModeTrap);

	if (y == 0)
		surf2Dwrite(v, surf, (x + n*algo_sz / 2) * 4, algo_sz, cudaBoundaryModeTrap);

}

__global__ void surfaceWriteKernel(float *gIData, int width, int height, cudaSurfaceObject_t surf)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	surf2Dwrite(gIData[y * width + x],
		surf, x * 4, y, cudaBoundaryModeTrap);
}

extern "C" void genTexture(RotfunStruct& rot_struct, float* src) {
	surfaceWriteKernel << <dim3(10, 10), dim3(32, 32), 0, get_stream() >> > (src, algo_sz, algo_sz, rot_struct.pSurfObject);
}

extern "C" void rotfun(RotfunStruct& rot_struct, float* src, float* dest) {
	rotate_many << <dim3(10, 10), dim3(32, 32), 0, get_stream() >> > (rot_struct.croppedTex, rot_struct.interm, algo_sz, rot_struct.params1);
	checkCudaErrors(cufftExecR2C(rot_struct.fftPlan, rot_struct.interm, rot_struct.fft_out_data));
	toAbs << <dim3(10, 20, rotfun_num), dim3(16, 16), 0, get_stream() >> > (rot_struct.fft_out_data, rot_struct.pFFTSurfObject);
	absrot << <dim3(10, ceil(theta_num / 16.0)), dim3(16, 16), 0, get_stream() >> > (rot_struct.FFTcroppedTex, dest, rot_struct.params2);
	checkCudaErrors(cudaStreamSynchronize(get_stream()));
}