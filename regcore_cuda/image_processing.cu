
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include "kernels.cuh"
#include <helper_cuda.h>
 
#include "clahe/saturate_cast.hpp"

using namespace cv::cuda::device;


extern "C" void initImageProcessing(ImProcessStruct& ip_struct)
{
	initCLAHE(ip_struct);
	cudaMalloc((void**)&ip_struct.tmp1, algo_sz*algo_sz);
	cudaMalloc((void**)&ip_struct.tmp2, algo_sz*algo_sz);
	cudaMalloc((void**)&ip_struct.hist, 256 * sizeof(int));
}

#define pd_sz algo_sz*2

extern "C" void applyMesh(ImProcessStruct& ip_struct, float2* mesh) {
	cudaMemcpy(ip_struct.mesh, mesh, croping_mesh * croping_mesh * sizeof(float2), cudaMemcpyHostToDevice);
}

extern "C" void initCropping(ImProcessStruct& ip_struct, int width, int height, float2* mesh)
{
	cudaMalloc((void**)&ip_struct.mesh, croping_mesh * croping_mesh * sizeof(float2));
	cudaMalloc((void**)&ip_struct.filt_tmp, (pd_sz + 5) * (pd_sz + 5) * sizeof(unsigned char));
	ip_struct.filt_tmp += pd_sz * 2;
	cudaMemcpy(ip_struct.mesh, mesh, croping_mesh * croping_mesh * sizeof(float2), cudaMemcpyHostToDevice);



	ip_struct.height = height;
	ip_struct.width = width;

	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	checkCudaErrors(cudaMallocArray(&ip_struct.cuArray, &channelDesc, ip_struct.width, ip_struct.height));

	memset(&ip_struct.resDesc, 0, sizeof(ip_struct.resDesc));
	ip_struct.resDesc.resType = cudaResourceTypeArray;
	ip_struct.resDesc.res.array.array = ip_struct.cuArray;
	cudaCreateSurfaceObject(&ip_struct.pSurfObject, &ip_struct.resDesc);

	memset(&ip_struct.texDesc, 0, sizeof(ip_struct.texDesc));
	ip_struct.texDesc.readMode = cudaReadModeElementType;
	ip_struct.texDesc.filterMode = cudaFilterModePoint;
	ip_struct.texDesc.addressMode[0] = cudaAddressModeClamp;
	ip_struct.texDesc.addressMode[1] = cudaAddressModeClamp;

	cudaCreateTextureObject(&ip_struct.rawTex, &ip_struct.resDesc, &ip_struct.texDesc, NULL);
}

__global__ void crop_kernel(cudaTextureObject_t rawTex, unsigned char* gpu_cropped, float2* mesh)
{
	const float mesh_sz = ((float)pd_sz / (croping_mesh - 1));
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int mesh_x = floorf(x / mesh_sz);
	const int mesh_y = floorf(y / mesh_sz);

	const float ax = x / mesh_sz - mesh_x;
	const float ay = y / mesh_sz - mesh_y;
	const float bx = 1 - ax;
	const float by = 1 - ay;
	
	float xx = mesh[mesh_x + mesh_y*croping_mesh].x*bx*by +
		mesh[mesh_x + (mesh_y + 1)*croping_mesh].x*bx*ay +
		mesh[(mesh_x + 1) + mesh_y*croping_mesh].x*ax*by +
		mesh[(mesh_x + 1) + (mesh_y + 1)*croping_mesh].x*ax*ay;

	float yy = mesh[mesh_x + mesh_y*croping_mesh].y*bx*by +
		mesh[mesh_x + (mesh_y + 1)*croping_mesh].y*bx*ay +
		mesh[(mesh_x + 1) + mesh_y*croping_mesh].y*ax*by +
		mesh[(mesh_x + 1) + (mesh_y + 1)*croping_mesh].y*ax*ay;

	gpu_cropped[x * pd_sz + y] = tex2D<unsigned char>(rawTex, xx, yy); //todo: reverse this?
} 

__global__ void filter_kernel_2(unsigned char* raw, unsigned char* filtered){

	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y2 = y * 2;
	const int x2 = x * 2;
	const int p0 = y2 * pd_sz + x2; 

	float p00 = p0 >= 1 + pd_sz ? raw[p0 - 1 - pd_sz] : raw[p0],
		p01 = p0 >= pd_sz ? raw[p0 - pd_sz] : raw[p0],
		p02 = p0 >= pd_sz - 1 ? raw[p0 + 1 - pd_sz] : raw[p0],
		p10 = p0 >= 1 ? raw[p0 - 1] : raw[p0],
		p11 = raw[p0],
		p12 = p0 + 1 < pd_sz*pd_sz ? raw[p0 + 1] : raw[p0],
		p20 = p0 - 1 + pd_sz < pd_sz*pd_sz ? raw[p0 - 1 + pd_sz] : raw[p0],
		p21 = p0 + pd_sz < pd_sz*pd_sz ? raw[p0 + pd_sz] : raw[p0],
		p22 = p0 + 1 + pd_sz < pd_sz*pd_sz ? raw[p0 + 1 + pd_sz] : raw[p0];

	const float sig = 20;
	float
		w00 = 0.0751 * exp(-abs(p00 - p11) / sig),
		w01 = 0.1238 * exp(-abs(p01 - p11) / sig),
		w02 = 0.0751 * exp(-abs(p02 - p11) / sig),
		w10 = 0.1238 * exp(-abs(p10 - p11) / sig),
		w12 = 0.1238 * exp(-abs(p12 - p11) / sig),
		w20 = 0.0751 * exp(-abs(p20 - p11) / sig),
		w21 = 0.1238 * exp(-abs(p21 - p11) / sig),
		w22 = 0.0751 * exp(-abs(p22 - p11) / sig);

	float v = (p00*w00
		+ p01*w01
		+ p02*w02
		+ p10*w10
		+ p11*0.2042
		+ p12*w12
		+ p20*w20
		+ p21*w21
		+ p22*w22) / (w00 + w01 + w02 + w10 + w12 + w20 + w21 + w22 + 0.2042);

	filtered[y*algo_sz + x] = v;
}

__global__ void bilateral_filter(unsigned char* raw, unsigned char* filtered) {

	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int p0 = y  * algo_sz + x;
	
	float p00 = p0 >= 1 + algo_sz ? raw[p0 - 1 - algo_sz] : raw[p0],
		p01 = p0 >= algo_sz ? raw[p0 - algo_sz] : raw[p0],
		p02 = p0 >= algo_sz - 1 ? raw[p0 + 1 - algo_sz] : raw[p0],
		p10 = p0 >= 1 ? raw[p0 - 1] : raw[p0],
		p11 = raw[p0],
		p12 = p0 + 1 < algo_sz*algo_sz ? raw[p0 + 1] : raw[p0],
		p20 = p0 - 1 + algo_sz < algo_sz*algo_sz ? raw[p0 - 1 + algo_sz] : raw[p0],
		p21 = p0 + algo_sz < algo_sz*algo_sz ? raw[p0 + algo_sz] : raw[p0],
		p22 = p0 + 1 + algo_sz < algo_sz*algo_sz ? raw[p0 + 1 + algo_sz] : raw[p0];

	const float sig = 10;
	float
		w00 = 0.0751 * exp(-abs(p00 - p11) / sig),
		w01 = 0.1238 * exp(-abs(p01 - p11) / sig),
		w02 = 0.0751 * exp(-abs(p02 - p11) / sig),
		w10 = 0.1238 * exp(-abs(p10 - p11) / sig),
		w12 = 0.1238 * exp(-abs(p12 - p11) / sig),
		w20 = 0.0751 * exp(-abs(p20 - p11) / sig),
		w21 = 0.1238 * exp(-abs(p21 - p11) / sig),
		w22 = 0.0751 * exp(-abs(p22 - p11) / sig);

	float v = (p00*w00
		+ p01*w01
		+ p02*w02
		+ p10*w10
		+ p11*0.2042
		+ p12*w12
		+ p20*w20
		+ p21*w21
		+ p22*w22) / (w00 + w01 + w02 + w10 + w12 + w20 + w21 + w22 + 0.2042);

	filtered[y*algo_sz + x] = v;
}


__global__ void surfaceWriteKernelB(unsigned char *gIData, int width, int height, cudaSurfaceObject_t surf)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
	surf2Dwrite(gIData[y * width + x],
		surf, x, y, cudaBoundaryModeTrap);
}

extern "C" void cropIm(ImProcessStruct& ip_struct, unsigned char* raw_input, unsigned char* gpu_cropped)
{
	int size = ip_struct.width * ip_struct.height * sizeof(char);
	// checkCudaErrors(cudaMemcpyToArray(ip_struct.cuArray, 0, 0, raw_input, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy2DToArray(ip_struct.cuArray, 0, 0, raw_input, ip_struct.width, ip_struct.width, ip_struct.height, cudaMemcpyHostToDevice));
	crop_kernel << <dim3(20, 20), dim3(32, 32), 0, get_stream() >> >(ip_struct.rawTex, ip_struct.filt_tmp, ip_struct.mesh);
	filter_kernel_2 << <dim3(10, 10), dim3(32, 32), 0, get_stream() >> > (ip_struct.filt_tmp, gpu_cropped);
	checkCudaErrors(cudaStreamSynchronize(get_stream()));
}

__global__ void edge_kernel(const unsigned char* pix, unsigned char* filtered)
{
	const int border = 4;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || y < 0 || x >= algo_sz || y >= algo_sz) return;
	if (x<border || y <border || x >= algo_sz - border || y >= algo_sz - border) {
		filtered[x*algo_sz + y] = 0;
		return;
	}
	const int p0 = y * algo_sz + x;
	
	if (pix[p0 + 1] == 0 || pix[p0 - 1] == 0 || pix[p0 + algo_sz] == 0 || pix[p0 - algo_sz] == 0) {
		filtered[y*algo_sz + x] = 0;
		return;
	}
	float valx = pix[p0 + 1] - pix[p0 - 1];
	float valy = pix[p0 + algo_sz] - pix[p0 - algo_sz];

	float m = sqrt(valx*valx + valy*valy);

	const int rborder = 10;
	if (x < rborder) m = m / rborder*(x);
	if (y < rborder) m = m / rborder*(y);
	if (x >= algo_sz - rborder) m = m / rborder*(algo_sz - x);
	if (y >= algo_sz - rborder) m = m / rborder*(algo_sz - y);

	filtered[y*algo_sz + x] = m;
}

#define rnditer(seed) (((seed + blockIdx.x) * 9301 + 49297) % 233280) % (algo_sz*algo_sz - 256)
__global__ void hist_calc(int *Histograms, uchar *im, int seed)
{
	__shared__ unsigned int bHist[256];
	bHist[threadIdx.x] = 0;
	__syncthreads();

	unsigned int idx = rnditer(seed);
	for (int i = 0; i < 4; ++i) {
		idx = rnditer(idx);
		uchar tmp = im[idx + threadIdx.x];
		//if (tmp >= 256) tmp = 255; if (tmp < 0) tmp = 0;
		atomicAdd(&bHist[tmp], 1); //a warp
	}
	__syncthreads();
	//access 1024 elements. 1%.
	atomicAdd(&Histograms[threadIdx.x], bHist[threadIdx.x]); //10 blocks write this.
}

struct hist_struct
{
	float hist[256];
};
__global__ void hist_apply(hist_struct table, uchar*im, float* out)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	out[x] = table.hist[im[x]];
}



extern "C" void preProcess(ImProcessStruct& ip_struct, unsigned char* gpu_cropped, float* ci)
{
	dim3 tiles = dim3(10, 10);
	dim3 threads = dim3(32, 32);
	//applyCLAHE(ip_struct, gpu_cropped, ip_struct.tmp1);
	// test();
	edge_kernel << <tiles, threads, 0, get_stream() >> > (gpu_cropped, ip_struct.tmp2);



	cudaMemsetAsync(ip_struct.hist, 0, 256 * sizeof(int), get_stream());
	checkCudaErrors(cudaStreamSynchronize(get_stream()));

	hist_calc << <10, 256, 0, get_stream() >> > (ip_struct.hist, ip_struct.tmp2, clock());
	int hist[256];
	cudaMemcpyAsync(hist, ip_struct.hist, 256 * 4, cudaMemcpyDeviceToHost, get_stream());
	checkCudaErrors(cudaStreamSynchronize(get_stream()));
	for (int i = 1; i < 256; ++i)
		hist[i] += hist[i - 1];
	hist_struct hs = { 0 };
	for (int i = 0; i < 256; ++i) {
		float v = ((float)(hist[i])) / (float)(hist[255]);
		hs.hist[i] = pow(v, 8.5);
	}
	hist_apply << <100, 1024, 0, get_stream() >> > (hs, ip_struct.tmp2, ci);
}

#define SWAP(arr, i,j) if (arr[i]>arr[j]) {float tmp=arr[i];arr[i]=arr[j];arr[j]=tmp;}
__global__ void median(const unsigned char* pix, unsigned char* out) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 1 || y < 1 || x >= algo_sz - 1 || y >= algo_sz - 1) return;
	const int p0 = y * algo_sz + x;

	unsigned char p[9];
	p[0] = pix[p0 - algo_sz * 1 - 1], p[3] = pix[p0 - algo_sz * 1 - 0], p[6] = pix[p0 - algo_sz * 1 + 1],
	p[1] = pix[p0 - algo_sz * 0 - 1], p[4] = pix[p0 - algo_sz * 0 - 0], p[7] = pix[p0 - algo_sz * 0 + 1],
	p[2] = pix[p0 + algo_sz * 1 - 1], p[5] = pix[p0 + algo_sz * 1 - 0], p[8] = pix[p0 + algo_sz * 1 + 1];

	SWAP(p, 0, 1);
	SWAP(p, 2, 3);
	SWAP(p, 0, 2);
	SWAP(p, 1, 3);
	SWAP(p, 1, 2);
	SWAP(p, 4, 5);
	SWAP(p, 7, 8);
	SWAP(p, 6, 8);
	SWAP(p, 6, 7);
	SWAP(p, 4, 7);
	SWAP(p, 4, 6);
	SWAP(p, 5, 8);
	SWAP(p, 5, 7);
	SWAP(p, 5, 6);
	SWAP(p, 0, 5);
	SWAP(p, 0, 4);
	SWAP(p, 1, 6);
	SWAP(p, 1, 5);
	SWAP(p, 1, 4);
	SWAP(p, 2, 7);
	SWAP(p, 3, 8);
	SWAP(p, 3, 7);
	SWAP(p, 2, 5);
	SWAP(p, 2, 4);
	SWAP(p, 3, 6);
	SWAP(p, 3, 5);
	SWAP(p, 3, 4);

	out[p0] = p[4];
}

#define rg 1 
__global__ void mask_kernel(unsigned char* dest, unsigned char* mask, float biasX, float biasY, float sinTheta, float cosTheta, float dx1, float dy1,
	float dx2, float dy2)
{
	const int border = 4;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int p0 = y * algo_sz + x;

	float target = 0; 

	for (int i=-rg; i<=rg; ++i)
		for (int j = -rg; j <= rg; ++j) {
			int sx = x*cosTheta - y*sinTheta + biasX + dx1*i + dx2*j;
			int sy = x*sinTheta + y*cosTheta + biasY - dy1*i - dy2*j;
			if (sx >= 3 && sx < algo_sz-2 && sy >= 3 && sy < algo_sz-2) {
				if (target < mask[sy*algo_sz + algo_sz - sx])
					target = mask[sy*algo_sz + algo_sz - sx];
			}
		}
	//if (target > 128) dest[y*algo_sz + x] = 0;
	dest[y*algo_sz + x] *= (1 - target / 255);
}

extern "C" void applyMask(unsigned char* dest, unsigned char* mask, float r_x, float r_y, float r_th, float dx1, float dy1,
	float dx2, float dy2) {
	float theta = -r_th / 180 * 3.1415926f;
	float sinTheta = sin(theta), cosTheta = cos(theta);

	float nxT =  cosTheta * r_x - sinTheta * r_y;
	float nyT =  sinTheta * r_x + cosTheta * r_y;

	float biasX = -algo_sz / 2 * cosTheta + algo_sz / 2 * sinTheta + algo_sz / 2 + nxT;
	float biasY = -algo_sz / 2 * sinTheta - algo_sz / 2 * cosTheta + algo_sz / 2 + nyT;

	mask_kernel << <dim3(10, 10), dim3(32, 32), 0, get_stream() >> > (dest, mask, biasX, biasY, sinTheta, cosTheta, dx1, dy1, dx2, dy2);
	checkCudaErrors(cudaStreamSynchronize(get_stream()));
}