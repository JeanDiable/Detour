
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


__global__ void phaseCorrN(const Complex* __restrict__  i1f, Complex* __restrict__  i2rfN, int multip)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	const Complex a = i1f[i];
	//a.*b^/abs(a)/abs(b)
	for (int j = 0; j < multip; ++j, i += algo_sz*algo_sz) {
		Complex b = i2rfN[i];

		float newx = a.x*b.x + a.y*b.y;
		float newy = a.x*b.y - a.y*b.x;
		float l = sqrt(newx*newx + newy*newy) + 0.0000001f;
		i2rfN[i].x = newx / l;
		i2rfN[i].y = newy / l;
	}
}

__global__ void phaseCorr2(const Complex* i1f, Complex* i2f, Complex* i1f2, Complex* i2f2)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= algo_sz_2*algo_sz_2) return;

	Complex a2 = i1f2[i];
	Complex b2 = i2f2[i];

	float newx2 = a2.x*b2.x + a2.y*b2.y;
	float newy2 = a2.x*b2.y - a2.y*b2.x;
	float l2 = sqrt(newx2*newx2 + newy2*newy2) + 0.0000001f;

	i2f2[i].x = newx2 / l2;
	i2f2[i].y = newy2 / l2;

	if (i >= algo_sz*algo_sz) return;

	Complex a = i1f[i];
	Complex b = i2f[i];

	float newx = a.x*b.x + a.y*b.y;
	float newy = a.x*b.y - a.y*b.x;
	float l = sqrt(newx*newx + newy*newy) + 0.0000001f;

	i2f[i].x = newx / l;
	i2f[i].y = newy / l;

}

__global__ void i2rfCoarse(cudaTextureObject_t tex, Complex* fftResult, RotParams<rot_check_num> params)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	for (int n = 0; n<rot_check_num; ++n)
	{
		const float sx = x*params.cosTheta[n] - y*params.sinTheta[n] + params.biasX[n];
		const float sy = x*params.sinTheta[n] + y*params.cosTheta[n] + params.biasY[n];

		float v = tex2D<float>(tex, sx + 0.5f, sy + 0.5f);
		int id1 = y*algo_sz + x + n * 2 * algo_sz*algo_sz;
		fftResult[id1].x = v;
		fftResult[id1].y = 0;

		int id2 = (algo_sz - 1 - y)*algo_sz + (algo_sz - 1 - x) + (n * 2 + 1)*algo_sz*algo_sz;
		fftResult[id2].x = v;
		fftResult[id2].y = 0;
	}
}

template<int N>
__global__ void i2rfFine(cudaTextureObject_t tex, Complex* fftResult, RotParams<N> params)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int id = y*algo_sz + x;

	for (int n = 0; n < N; ++n)
	{
		const float sx = x*params.cosTheta[n] - y*params.sinTheta[n] + params.biasX[n];
		const float sy = x*params.sinTheta[n] + y*params.cosTheta[n] + params.biasY[n];

		float v = tex2D<float>(tex, sx + 0.5f, sy + 0.5f);
		fftResult[id].x = v;
		fftResult[id].y = 0;
		id += algo_sz*algo_sz;
	}
}



__global__ void i1_kernel(float* what, Complex* i1f, Complex* i1f2)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	i1f[i].x = what[i];

	int j = (blockIdx.x + (algo_sz_2 - algo_sz) / 2) * algo_sz_2 + threadIdx.x + (algo_sz_2 - algo_sz) / 2;
	i1f2[j].x = what[i];
}

extern "C" void i1fft(PhaseCorrelationStruct& pc_struct, float* i1)
{
	cudaMemset(pc_struct.i1f, 0, sizeof(Complex)*algo_sz*algo_sz);
	cudaMemset(pc_struct.i1f2, 0, sizeof(Complex)*algo_sz_2*algo_sz_2);

	// i1-> i1f/i1f2
	i1_kernel << <algo_sz, algo_sz, 0, get_stream() >> > (i1, pc_struct.i1f, pc_struct.i1f2);

	checkCudaErrors(cufftExecC2C(pc_struct.fftPlan_i$f, pc_struct.i1f, pc_struct.i1f, CUFFT_FORWARD));
	checkCudaErrors(cufftExecC2C(pc_struct.fftPlan_i$f2, pc_struct.i1f2, pc_struct.i1f2, CUFFT_FORWARD));
}


extern "C" void initPhaseCorrelation(PhaseCorrelationStruct& pc_struct)
{
	int n[2] = { algo_sz, algo_sz };
	checkCudaErrors(cufftPlanMany(&pc_struct.fftPlanCoarse, 2, n,
		nullptr, 1, 0,
		nullptr, 1, 0,
		CUFFT_C2C, rot_check_num * 2));
	checkCudaErrors(cufftPlanMany(&pc_struct.fftPlanFine, 2, n,
		nullptr, 1, 0,
		nullptr, 1, 0,
		CUFFT_C2C, 2));
	checkCudaErrors(cufftPlanMany(&pc_struct.fftPlanFine2, 2, n,
		nullptr, 1, 0,
		nullptr, 1, 0,
		CUFFT_C2C, 3));
	cufftSetStream(pc_struct.fftPlanCoarse, get_stream());
	cufftSetStream(pc_struct.fftPlanFine, get_stream());
	cufftSetStream(pc_struct.fftPlanFine2, get_stream());

	checkCudaErrors(cufftPlan2d(&pc_struct.fftPlan_i$f, algo_sz, algo_sz, CUFFT_C2C));
	checkCudaErrors(cufftPlan2d(&pc_struct.fftPlan_i$f2, algo_sz_2, algo_sz_2, CUFFT_C2C));
	cufftSetStream(pc_struct.fftPlan_i$f, get_stream());
	cufftSetStream(pc_struct.fftPlan_i$f2, get_stream());


	checkCudaErrors(cudaMalloc((void**)&pc_struct.interm, algo_sz*algo_sz * sizeof(Complex)*rot_check_num * 2));
	checkCudaErrors(cudaMalloc((void**)&pc_struct.i1f, algo_sz*algo_sz * sizeof(Complex) + 1));
	checkCudaErrors(cudaMalloc((void**)&pc_struct.i1f2, algo_sz_2*algo_sz_2 * sizeof(Complex) + 1));
	checkCudaErrors(cudaMalloc((void**)&pc_struct.conf_coarse, 100 * rot_check_num * 2 * sizeof(conf_pair)));
	checkCudaErrors(cudaMalloc((void**)&pc_struct.interm2, algo_sz*algo_sz * sizeof(Complex) * 2));
	checkCudaErrors(cudaMalloc((void**)&pc_struct.conf_fine, 100 * 2 * sizeof(conf_pair)));
	checkCudaErrors(cudaMalloc((void**)&pc_struct.interm3, algo_sz*algo_sz * sizeof(Complex) * 3));
	checkCudaErrors(cudaMalloc((void**)&pc_struct.conf_fine2, 100 * 3 * sizeof(conf_pair)));
	checkCudaErrors(cudaMalloc((void**)&pc_struct.i2f, algo_sz*algo_sz * sizeof(Complex) + 1));
	checkCudaErrors(cudaMalloc((void**)&pc_struct.i2f2, algo_sz_2*algo_sz_2 * sizeof(Complex) + 1));
	checkCudaErrors(cudaMalloc((void**)&pc_struct.g1, algo_sz_2*algo_sz_2 * sizeof(float) + 1));
//	checkCudaErrors(cudaMalloc((void**)&pc_struct.g2, algo_sz_2*algo_sz_2 * sizeof(float) + 1));
	checkCudaErrors(cudaMalloc((void**)&pc_struct.absed, algo_sz*algo_sz * sizeof(float) + 1));
	checkCudaErrors(cudaMalloc((void**)&pc_struct.absed2, algo_sz_2*algo_sz_2 * sizeof(float) + 1));
	checkCudaErrors(cudaMalloc((void**)&pc_struct.block_maxima1, 100 * sizeof(maxima_struct)));
	checkCudaErrors(cudaMalloc((void**)&pc_struct.block_maxima2, 107 * sizeof(maxima_struct)));

	pc_struct.peaks = new Complex[algo_sz*algo_sz];
}

__global__ void conf_reduce(Complex *peaks, conf_pair* block_conf)
{
	__shared__ float sums2[32], sums[32], maximas[32];

	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int lane_id = threadIdx.x % warpSize;
	int warp_id = threadIdx.x / warpSize;

	float sum2 = peaks[id].x*peaks[id].x + peaks[id].y*peaks[id].y;
	float sum = sqrt(sum2);
	float maxima = sum;

#pragma unroll
	for (int offset = 16; offset > 0; offset /= 2)
	{
		sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
		sum += __shfl_down_sync(0xffffffff, sum, offset);
		float tmp = __shfl_down_sync(0xffffffff, maxima, offset);
		maxima = max(tmp, maxima);
	}

	if (lane_id == 0)
	{
		sums2[warp_id] = sum2;
		sums[warp_id] = sum;
		maximas[warp_id] = maxima;
	}
	__syncthreads();

	if (warp_id == 0) //use one warp to reduce all.
	{
		sum2 = sums2[lane_id];
		sum = sums[lane_id];
		maxima = maximas[lane_id];
#pragma unroll
		for (int offset = 16; offset > 0; offset /= 2)
		{
			sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
			sum += __shfl_down_sync(0xffffffff, sum, offset);
			float tmp = __shfl_down_sync(0xffffffff, maxima, offset);
			maxima = max(tmp, maxima);
		}
	}

	if (threadIdx.x != 0) return;

	block_conf[blockIdx.x].max_val = maxima;
	block_conf[blockIdx.x].sum2_val = sum2;
	block_conf[blockIdx.x].sum_val = sum;
}

__global__ void toABS(Complex* raw, float* absed, int N) {
	const int p0 = blockIdx.x * blockDim.x + threadIdx.x;
	if (p0 < N)
		absed[p0] = sqrt(raw[p0].x*raw[p0].x + raw[p0].y*raw[p0].y);
}

__global__ void gaussian(float* raw, float* filtered, int sz) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= sz) return;
	const int xu = x > 0 ? x - 1 : sz - 1;
	const int xd = x < sz - 1 ? x + 1 : 0;

	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (y >= sz) return;
	const int yu = y > 0 ? y - 1 : sz - 1;
	const int yd = y < sz-1 ? y + 1 : 0;

	const int p0 = y * sz + x;

	float p00 = raw[yu *sz + xu],
		p01 = raw[yu *sz + x],
		p02 = raw[yu *sz + xd],
		p10 = raw[y *sz + xu],
		p11 = raw[y *sz + x],
		p12 = raw[y *sz + xd],
		p20 = raw[yd *sz + xu],
		p21 = raw[yd *sz + x],
		p22 = raw[yd *sz + xd];

	filtered[p0] = 0.0751 * p00 +
		0.1238 * p01 +
		0.0751 * p02 +
		0.1238 * p10 +
		0.2042 * p11 +
		0.1238 * p12 +
		0.0751 * p20 +
		0.1238 * p21 +
		0.0751 * p22;
}


__global__ void maxima_reduce_with_mask(float *peaks, int N, maxima_struct* block_maxima, float4 xyd)
{
	__shared__ float  maximas[32];
	__shared__ int indxs[32];
	__shared__ int xx, yy;
	__shared__ float frF[9];

	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int lane_id = threadIdx.x % warpSize;
	int warp_id = threadIdx.x / warpSize;

	int indx = id;
	float maxima = id < N*N ? peaks[id] : 0;

	int mx = indx%N;
	int my = indx / N;
	float dx = min((xyd.x - mx)*(xyd.x - mx), (xyd.x - mx + N)*(xyd.x - mx + N));
	float dy = min((xyd.y - my)*(xyd.y - my), (xyd.y - my + N)*(xyd.y - my + N));
	maxima = maxima *exp(-(dx + dy) / (2 * xyd.z*xyd.z));

	//32->1
#pragma unroll
	for (int offset = 16; offset > 0; offset /= 2)
	{
		float tmp = __shfl_down_sync(0xffffffff, maxima, offset);
		float tmpIdx = __shfl_down_sync(0xffffffff, indx, offset);
		indx = (tmp > maxima)*tmpIdx + (tmp <= maxima)*indx;
		maxima = max(tmp, maxima);
	}

	if (lane_id == 0)
	{
		indxs[warp_id] = indx;
		maximas[warp_id] = maxima;
	}
	__syncthreads();

	//32->1
	if (warp_id == 0) //use one warp to reduce all.
	{
		maxima = maximas[lane_id];
		indx = indxs[lane_id];
#pragma unroll
		for (int offset = 16; offset > 0; offset /= 2)
		{
			float tmp = __shfl_down_sync(0xffffffff, maxima, offset);
			float tmpIdx = __shfl_down_sync(0xffffffff, indx, offset);
			indx = (tmp > maxima)*tmpIdx + (tmp <= maxima)*indx;
			maxima = max(tmp, maxima);
		}
	}
	if (threadIdx.x == 0)
	{
		xx = indx%N;
		yy = indx / N;
	}

	__syncthreads();

	if (threadIdx.x <9)
	{
		int xxx = xx + lane_id / 3 - 1;
		xxx = (xxx == -1)*(N - 1) + (xxx != -1 && xxx != N)*xxx + (xxx == N) * 0;
		int yyy = yy + lane_id % 3 - 1;
		yyy = (yyy == -1)*(N - 1) + (yyy != -1 && yyy != N)*yyy + (yyy == N) * 0;
		id = xxx + yyy*N;
		frF[threadIdx.x] = peaks[id];
	};

	__syncthreads();

	if (threadIdx.x == 0)
	{
		block_maxima[blockIdx.x].max_val = maxima;
		float frSum = frF[0] + frF[1] + frF[2] + frF[3] + frF[4] + frF[5] + frF[6] + frF[7] + frF[8] + 0.0001f;
		block_maxima[blockIdx.x].x = xx + (frF[2] - frF[0] + frF[5] - frF[3] + frF[8] - frF[6]) / frSum;
		block_maxima[blockIdx.x].y = yy + (frF[6] - frF[0] + frF[7] - frF[1] + frF[8] - frF[2]) / frSum;
	}
}




template<int N>
void calcTheta(float theta, RotParams<N>* p, int idx)
{
	p->sinTheta[idx] = sin(theta), p->cosTheta[idx] = cos(theta);
	p->biasX[idx] = -algo_sz / 2 * p->cosTheta[idx] + algo_sz / 2 * p->sinTheta[idx] + algo_sz / 2;
	p->biasY[idx] = -algo_sz / 2 * p->sinTheta[idx] - algo_sz / 2 * p->cosTheta[idx] + algo_sz / 2;
}

struct stat_struct
{
	float e;
	float std;
};

template<int N>
stat_struct testConf(float* confsF, Complex* interm, conf_pair* gpu_pair)
{
	conf_reduce << <100 * N, 1024, 0, get_stream() >> > (interm, gpu_pair);

	conf_pair pair[100 * N];
	cudaMemcpyAsync(pair, gpu_pair, 100 * N * sizeof(conf_pair), cudaMemcpyDeviceToHost, get_stream());
	cudaStreamSynchronize(get_stream());
	float e = 0, e2 = 0, maxima = 0;
	for (int n = 0; n<N; ++n)
	{
		e = e2 = maxima = 0;
		for (int i = 100 * n; i<100 * (n + 1); ++i)
		{
			e += pair[i].sum_val;
			e2 += pair[i].sum2_val;
			maxima = max(maxima, pair[i].max_val);
		}
		e = e / algo_sz / algo_sz;
		e2 = e2 / algo_sz / algo_sz;
		confsF[n] = (maxima - e) / sqrt(e2 - e*e);
	}
	return{ e,sqrt(e2 - e*e) };
}

extern "C" bool coarseTheta(PhaseCorrelationStruct& pc_struct, RotfunStruct& rot_struct, ThetaFunStruct& tf_struct)
{
	RotParams<rot_check_num> p;
	for (int i = 0; i < rot_check_num; ++i)
		calcTheta<rot_check_num>(tf_struct.theta_results[i] / theta_num* 3.1415926f, &p, i);
	// 0.67
	i2rfCoarse << <dim3(10, 10), dim3(32, 32), 0, get_stream() >> > (rot_struct.croppedTex, pc_struct.interm, p); //0.73
																												 //

	checkCudaErrors(cufftExecC2C(pc_struct.fftPlanCoarse, pc_struct.interm, pc_struct.interm, CUFFT_FORWARD)); //0.94
	phaseCorrN << <100, 1024, 0, get_stream() >> > (pc_struct.i1f, pc_struct.interm, rot_check_num * 2); //1.04 bound by mem bandwidth?!... consider callbacks.
	checkCudaErrors(cufftExecC2C(pc_struct.fftPlanCoarse, pc_struct.interm, pc_struct.interm, CUFFT_FORWARD)); //1.28

	float thConfs[rot_check_num * 2];
	testConf<rot_check_num * 2>(thConfs, pc_struct.interm, pc_struct.conf_coarse);

	pc_struct.coarseMaxConf = 0;
	int thIdx = 0;
	//printf("coarse:[");
	for (int n = 0; n < rot_check_num * 2; ++n)
	{
		//printf("%.2f:%.2f, ", (tf_struct.theta_results[n / 2] + (n % 2)* theta_num) / theta_num * 180, thConfs[n]);
		if (thConfs[n] > pc_struct.coarseMaxConf)
		{
			pc_struct.coarseMaxConf = thConfs[n];
			thIdx = n;
		}
	}
	//printf("]\n");
	//system("pause");

	int leftTh = tf_struct.theta_results[thIdx / 2] - 1; if (leftTh == -1) leftTh = theta_num - 1;
	int rightTh = tf_struct.theta_results[thIdx / 2] + 1; if (rightTh == theta_num) rightTh = 0;
	float leftConf = 5, rightConf = 5;
	pc_struct.scope = init_scope;
	for (int n = 0; n<rot_check_num; ++n)
	{
		if (tf_struct.theta_results[n] == leftTh) {
			leftConf = thConfs[thIdx % 2 + n * 2];
			pc_struct.scope = better_scope;
		}
		if (tf_struct.theta_results[n] == rightTh) {
			rightConf = thConfs[thIdx % 2 + n * 2];
			pc_struct.scope = better_scope;
		}
	}
	//printf("coarseTheta:%d, fix:%f\n", tf_struct.theta_results[thIdx / 2] + theta_num*(thIdx % 2), (-(rightConf - leftConf) / 2.0f / (leftConf + rightConf - 2.0f * pc_struct.coarseMaxConf + 0.00001f)));
	pc_struct.coarseTheta = (tf_struct.theta_results[thIdx / 2] + theta_num*(thIdx % 2) +
		(-(rightConf - leftConf) / 2.0f / (leftConf + rightConf - 2.0f * pc_struct.coarseMaxConf + 0.00001f)))
		/ theta_num * 180;
	if (isnan(pc_struct.coarseTheta))
		return false;
	return true;
	//1.34ms
}

float thetaInterp(float* confsF)
{
	if (confsF[0] > confsF[1] && confsF[0] > confsF[2]) {
		//printf("left overflow...\n");
		return -1;
	}
	else if (confsF[2] > confsF[0] && confsF[2] > confsF[1]) {
		return (-(confsF[1] - confsF[0]) / 2.0f / (confsF[0] + confsF[1] - 2.0f * confsF[2] + 0.0001f));
	}
	else if (confsF[1] > confsF[0] && confsF[1] > confsF[2]) {
		//printf("right overflow...\n");
		return 1;
	}
}

extern "C" void fineTheta(PhaseCorrelationStruct& pc_struct, RotfunStruct& rot_struct, ThetaFunStruct& tf_struct, bool twice)
{
	float confsF[3];

	if (twice) {
		RotParams<2> p;
		calcTheta<2>((pc_struct.coarseTheta - pc_struct.scope) / 180 * 3.1415926f, &p, 0);
		calcTheta<2>((pc_struct.coarseTheta + pc_struct.scope) / 180 * 3.1415926f, &p, 1);
		
		i2rfFine<2> << <dim3(10, 10), dim3(32, 32), 0, get_stream() >> >(rot_struct.croppedTex, pc_struct.interm2, p);
		
		checkCudaErrors(cufftExecC2C(pc_struct.fftPlanFine, pc_struct.interm2, pc_struct.interm2, CUFFT_FORWARD));
		phaseCorrN << <200, 512, 0, get_stream() >> > (pc_struct.i1f, pc_struct.interm2, 2);
		checkCudaErrors(cufftExecC2C(pc_struct.fftPlanFine, pc_struct.interm2, pc_struct.interm2, CUFFT_FORWARD));
		
		auto stat = testConf<2>(confsF, pc_struct.interm2, pc_struct.conf_fine);
		pc_struct.e = stat.e; pc_struct.std = stat.std;
		confsF[2] = pc_struct.coarseMaxConf;
		
		float bias = thetaInterp(confsF);
		pc_struct.coarseTheta = pc_struct.coarseTheta + pc_struct.scope*bias;
		pc_struct.scope /= 3;
	}
	//////////////////////////
	RotParams<3> p2;
	auto oldTheta = pc_struct.coarseTheta;
	calcTheta<3>((oldTheta - pc_struct.scope) / 180 * 3.1415926f, &p2, 0);
	calcTheta<3>((oldTheta + pc_struct.scope) / 180 * 3.1415926f, &p2, 1);
	calcTheta<3>((oldTheta) / 180 * 3.1415926f, &p2, 2);

	i2rfFine<3> << <dim3(10, 10), dim3(32, 32), 0, get_stream() >> >(rot_struct.croppedTex, pc_struct.interm3, p2);
	cufftExecC2C(pc_struct.fftPlanFine2, pc_struct.interm3, pc_struct.interm3, CUFFT_FORWARD);
	phaseCorrN << <200, 512, 0, get_stream() >> > (pc_struct.i1f, pc_struct.interm3, 3);
	cufftExecC2C(pc_struct.fftPlanFine2, pc_struct.interm3, pc_struct.interm3, CUFFT_FORWARD);

	auto stat = testConf<3>(confsF, pc_struct.interm3, pc_struct.conf_fine2);
	pc_struct.e = stat.e; pc_struct.std = stat.std;
	auto bias = thetaInterp(confsF);
	pc_struct.fineTheta = oldTheta + pc_struct.scope*bias;
	pc_struct.fineConf = confsF[(int)(round(bias) + 1)];
	//	printf("       fine theta, %.2f,%.2f,%.2f (%.2f, %.2f, %.2f) -> %f\n",
	//		oldTheta - pc_struct.scope, oldTheta, oldTheta + pc_struct.scope,
	//		confsF[0], confsF[2], confsF[1], pc_struct.fineTheta);
}

__global__ void i2rf1f2(cudaTextureObject_t tex, Complex* i2f1, Complex* i2f2, float cosTheta, float sinTheta, float biasX, float biasY)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	const int rborder = 8;

	if (!(x < algo_sz_2 && y < algo_sz_2))
		return;

	float sx = x*cosTheta - y*sinTheta + biasX;
	float sy = x*sinTheta + y*cosTheta + biasY;
	float value = tex2D<float>(tex, sx + 0.5f, sy + 0.5f);

	if (y >= (algo_sz_2 - algo_sz) / 2 && x >= (algo_sz_2 - algo_sz) / 2 && y < (algo_sz_2 + algo_sz) / 2 && x < (algo_sz_2 + algo_sz) / 2) {
		int px = (x - (algo_sz_2 - algo_sz) / 2), py = (y - (algo_sz_2 - algo_sz) / 2);
		float m = value;
		if (px < rborder) m = m / rborder*(px);
		if (py < rborder) m = m / rborder*(py);
		if (px >= algo_sz - rborder) m = m / rborder*(algo_sz - px);
		if (py >= algo_sz - rborder) m = m / rborder*(algo_sz - py);
		i2f1[py*algo_sz + px].x = value;
	}

	if (x < rborder) value = value / rborder*(x);
	if (y < rborder) value = value / rborder*(y);
	if (x >= algo_sz_2 - rborder) value = value / rborder*(algo_sz_2 - x);
	if (y >= algo_sz_2 - rborder) value = value / rborder*(algo_sz_2 - y);
	i2f2[y*algo_sz_2 + x].x = value;
}

#define sz2dim 106
extern "C" void phase(PhaseCorrelationStruct& pc_struct, RotfunStruct& rot_struct, ThetaFunStruct& tf_struct, bool masked, float4 xyd)
{
	cudaMemsetAsync(pc_struct.i2f, 0, algo_sz*algo_sz * sizeof(Complex), get_stream());
	cudaMemsetAsync(pc_struct.i2f2, 0, algo_sz_2*algo_sz_2 * sizeof(Complex), get_stream());
	checkCudaErrors(cudaStreamSynchronize(get_stream()));

	float theta = pc_struct.fineTheta / 180 * 3.1415926f;
	float sinTheta = sin(theta), cosTheta = cos(theta);
	float biasX = -algo_sz_2 / 2 * cosTheta + algo_sz_2 / 2 * sinTheta + algo_sz / 2;
	float biasY = -algo_sz_2 / 2 * sinTheta - algo_sz_2 / 2 * cosTheta + algo_sz / 2;

	i2rf1f2 << <dim3(41, 41), dim3(8, 8), 0, get_stream() >> > (rot_struct.croppedTex, pc_struct.i2f, pc_struct.i2f2,
		cosTheta, sinTheta, biasX, biasY);

	//float2 tmp[102400];
	//cudaMemcpyAsync(tmp, pc_struct.i2f, 102400 * sizeof(float2), cudaMemcpyDeviceToHost);
	//checkCudaErrors(cudaStreamSynchronize(get_stream()));
	//FILE* fd=fopen("absed", "w");
	//for (int i = 0; i < 320 * 320; ++i) fprintf(fd, "%f ", tmp[i].x);
	//fclose(fd);

	cufftExecC2C(pc_struct.fftPlan_i$f, pc_struct.i2f, pc_struct.i2f, CUFFT_FORWARD);
	cufftExecC2C(pc_struct.fftPlan_i$f2, pc_struct.i2f2, pc_struct.i2f2, CUFFT_FORWARD);
	phaseCorr2 << <sz2dim, 1024, 0, get_stream() >> > (pc_struct.i1f, pc_struct.i2f, pc_struct.i1f2, pc_struct.i2f2);
	cufftExecC2C(pc_struct.fftPlan_i$f, pc_struct.i2f, pc_struct.i2f, CUFFT_FORWARD);
	cufftExecC2C(pc_struct.fftPlan_i$f2, pc_struct.i2f2, pc_struct.i2f2, CUFFT_FORWARD);

	toABS << <100, 1024, 0, get_stream() >> > (pc_struct.i2f, pc_struct.absed, algo_sz*algo_sz);


	gaussian << <dim3(40, 40), dim3(8, 8), 0, get_stream() >> > (pc_struct.absed, pc_struct.g1, algo_sz);
	gaussian << <dim3(40, 40), dim3(8, 8), 0, get_stream() >> > (pc_struct.g1, pc_struct.absed, algo_sz);
	gaussian << <dim3(40, 40), dim3(8, 8), 0, get_stream() >> > (pc_struct.absed, pc_struct.g1, algo_sz);
	gaussian << <dim3(40, 40), dim3(8, 8), 0, get_stream() >> > (pc_struct.g1, pc_struct.absed, algo_sz);

	toABS << <sz2dim, 1024, 0, get_stream() >> > (pc_struct.i2f2, pc_struct.absed2, algo_sz_2*algo_sz_2);
	gaussian << <dim3(41, 41), dim3(8, 8), 0, get_stream() >> > (pc_struct.absed2, pc_struct.g1, algo_sz_2);
	gaussian << <dim3(41, 41), dim3(8, 8), 0, get_stream() >> > (pc_struct.g1, pc_struct.absed2, algo_sz_2);
	gaussian << <dim3(41, 41), dim3(8, 8), 0, get_stream() >> > (pc_struct.absed2, pc_struct.g1, algo_sz_2);
	gaussian << <dim3(41, 41), dim3(8, 8), 0, get_stream() >> > (pc_struct.g1, pc_struct.absed2, algo_sz_2);

	//maxima_reduce << <100, 1024, 0, get_stream() >> > (pc_struct.i2f, algo_sz, pc_struct.block_maxima1);
	//maxima_reduce << <sz2dim, 1024, 0, get_stream() >> > (pc_struct.i2f2, algo_sz_2, pc_struct.block_maxima2);
	auto xyd2 = xyd;
	if (xyd2.x < 0)xyd2.x += algo_sz;
	if (xyd2.y < 0)xyd2.y += algo_sz;
	maxima_reduce_with_mask << <100, 1024, 0, get_stream() >> > (pc_struct.absed, algo_sz, pc_struct.block_maxima1,xyd2);
	auto xyd3 = xyd;
	if (xyd3.x < 0)xyd3.x += algo_sz_2;
	if (xyd3.y < 0)xyd3.y += algo_sz_2;
	maxima_reduce_with_mask << <sz2dim, 1024, 0, get_stream() >> > (pc_struct.absed2, algo_sz_2, pc_struct.block_maxima2,xyd3);
	
	maxima_struct maxima1[100];
	maxima_struct maxima2[sz2dim];
	cudaMemcpyAsync(maxima1, pc_struct.block_maxima1, 100 * sizeof(maxima_struct), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(maxima2, pc_struct.block_maxima2, sz2dim * sizeof(maxima_struct), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaStreamSynchronize(get_stream()));

	float maxima = -9999, x1, y1, x2, y2;
	for (int i = 0; i<100; ++i)
	{
		if (maxima<maxima1[i].max_val)
		{
			maxima = maxima1[i].max_val;
			x1 = maxima1[i].x;
			y1 = maxima1[i].y;
		}
	}
	pc_struct.phConf = pc_struct.fineConf; //(sqrt(maxima) - pc_struct.e) / pc_struct.std;

	maxima = -9999;
	for (int i = 0; i<sz2dim; ++i)
	{
		if (maxima<maxima2[i].max_val)
		{
			maxima = maxima2[i].max_val;
			x2 = maxima2[i].x;
			y2 = maxima2[i].y;
		}
	}
	pc_struct.phConf2 = pc_struct.fineConf;// (sqrt(maxima) - pc_struct.e) / pc_struct.std;

	//printf("1(%f,%f), conf:%f \n2(%f,%f), conf:%f\n", x1, y1, pc_struct.phConf, x2, y2, (sqrt(maxima) - pc_struct.e) / pc_struct.std);

	float x = x1; float y = y1;

	// all these shit ballots a 
	float x1a = x1, x2a = x1 - algo_sz, x1b = x2, x2b = x2 - algo_sz_2;
	float c1 = abs(x1a - x1b), c2 = abs(x1a - x2b), c3 = abs(x2a - x1b), c4 = abs(x2a - x2b);
	if (x1b > algo_sz) { // prevent fake peak
		if (abs(x1a - x2b) > abs(x2a - x2b)) x = x2a;
	}
	else if (x2b < -algo_sz) {
		if (abs(x1a - x1b) > abs(x2a - x1b)) x = x2a;
	}
	else if (c1 < c2 && c1 < c3 && c1 < c4) x = x1a;
	else if (c2 < c1 && c2 < c3 && c2 < c4) x = x1a;
	else if (c3 < c1 && c3 < c2 && c3 < c4) x = x2a;
	else if (c4 < c1 && c4 < c2 && c4 < c3) x = x2a;
	int sdx = min(c1, min(c2, min(c3, c4)));

	float y1a = y1, y2a = y1 - algo_sz, y1b = y2, y2b = y2 - algo_sz_2;
	c1 = abs(y1a - y1b), c2 = abs(y1a - y2b), c3 = abs(y2a - y1b), c4 = abs(y2a - y2b);
	if (y1b > algo_sz) {
		if (abs(y1a - y2b) > abs(y2a - y2b)) y = y2a;
	}
	else if (y2b < -algo_sz) {
		if (abs(y1a - y1b) > abs(y2a - y1b)) y = y2a;
	}
	else if (c1 < c2 && c1 < c3 && c1 < c4) y = y1a;
	else if (c2 < c1 && c2 < c3 && c2 < c4) y = y1a;
	else if (c3 < c1 && c3 < c2 && c3 < c4) y = y2a;
	else if (c4 < c1 && c4 < c2 && c4 < c3) y = y2a;
	int sdy = min(c1, min(c2, min(c3, c4)));

	if (sdx > 3 || sdy > 3) pc_struct.phConf = 0;
	pc_struct.finalX = x;
	pc_struct.finalY = y;

}