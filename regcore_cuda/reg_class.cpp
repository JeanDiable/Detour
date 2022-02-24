#pragma once
#include <cuda_runtime.h>
#include "kernels.cuh"
#include <helper_cuda.h>
#include "reg_class.h"
#include <ctime>
#include <thread>
#include <mutex>
#include <chrono>
#include "build_def.h"

void reg_core::checkThread()
{
	if (thisThread != std::this_thread::get_id())
	{
		printf("[regcore] called from another thread violation...\n");
		exit(-1);
	}
}
void reg_core::cropImage(unsigned char* rawIn, int reg_im_idx)
{
	cropIm(ip_struct, rawIn, reg_pool[reg_im_idx]);
}

void reg_core::mesh(float2* mesh)
{
	applyMesh(ip_struct, mesh);
}

void reg_core::init(int width, int height, float2* mesh)
{
	init_regonly();
	initCropping(ip_struct, width, height, mesh);
	printf("image processing inited...\n");
}

bool deviceInit = false;
std::mutex global;
void reg_core::init_regonly()
{
	global.lock();
	if (!deviceInit)
	{
		deviceInit = true;
		findCudaDevice(0, 0);
	}
	global.unlock();

	get_stream();
	initRotFun(rotfun_struct);
	initThetaFinding(tf_struct);
	initPhaseCorrelation(pc_struct);
	initImageProcessing(ip_struct);
	checkCudaErrors(cudaMalloc((void**)&i1rBuf, theta_num*algo_sz / 2 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&i1pf, theta_num*algo_sz / 2 * sizeof(Complex)));
	checkCudaErrors(cudaMalloc((void**)&i2rBuf, theta_num*algo_sz / 2 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&i2pf, theta_num*algo_sz / 2 * sizeof(Complex)));
	for (int i = 0; i < max_pool; ++i)
		cudaMalloc((void**)&reg_pool[i], algo_sz*algo_sz * sizeof(unsigned char));
	for (int i = 0; i < max_pool; ++i)
		cudaMalloc((void**)&algo_pool[i], algo_sz*algo_sz * sizeof(float));
	printf("regcore inited...\n");
	thisThread = std::this_thread::get_id();
}

void reg_core::preprocess(int reg_im_idx, int algo_im_idx)
{
	checkThread();
	preProcess(ip_struct, reg_pool[reg_im_idx], algo_pool[algo_im_idx]);
}


void reg_core::set(int algo_im_idx)
{
	checkThread();
	genTexture(rotfun_struct, algo_pool[algo_im_idx]);
	rotfun(rotfun_struct, algo_pool[algo_im_idx], i1rBuf);
	computeRotFFT(tf_struct, i1rBuf, i1pf);
	i1fft(pc_struct, algo_pool[algo_im_idx]);
}

//


void reg_core::mask(int dest, int mask, float r_x, float r_y, float r_th, float dx1, float dy1,
	float dx2, float dy2) {
	applyMask(reg_pool[dest], reg_pool[mask], r_x, r_y, r_th, dx1, dy1, dx2, dy2);
}

reg_result reg_core::reg(int algo_im_idx, bool pos, float x, float y, float th, float th_range, float xy_range)
{
	checkThread();
	frames += 1;
	genTexture(rotfun_struct, algo_pool[algo_im_idx]);
	if (!pos && th_range < 1.5) {
		rotfun(rotfun_struct, algo_pool[algo_im_idx], i2rBuf);
		computeRotFFT(tf_struct, i2rBuf, i2pf);
		thetaFinding(tf_struct, i1pf, i2pf);
		if (!coarseTheta(pc_struct, rotfun_struct, tf_struct))
			return{ 0,0,0,-1 };
	}
	else {
		pc_struct.coarseTheta = th;
		pc_struct.scope = 1.5;
	}
	fineTheta(pc_struct, rotfun_struct, tf_struct, pos);
	phase(pc_struct, rotfun_struct, tf_struct, pos, { x,y,xy_range });
	if (pc_struct.fineTheta > 180) pc_struct.fineTheta -= 360;

	//printf("%d>(%.2f,%.2f,%.2f,%.2f)\t thC:%.2f(%.2f)\t tF:%.2f<>%.2f\n", frames, pc_struct.finalX, pc_struct.finalY,
	//	pc_struct.fineTheta, pc_struct.phConf, pc_struct.coarseTheta, pc_struct.coarseMaxConf, tf_struct.tconf, tf_struct.tconf_m);

	return{ -pc_struct.finalX, -pc_struct.finalY, -pc_struct.fineTheta, pc_struct.phConf };
}

reg_core* reg_core::getInstance()
{
	return new reg_core();
} 

extern "C" __declspec(dllexport) reg_core* CreateRegCore(int id = 0) {
	return reg_core::getInstance();
}

extern "C" __declspec(dllexport) void Reg(reg_core* core, int algo_idx, reg_result* result)
{
	*result = core->reg(algo_idx);
}

extern "C"  __declspec(dllexport) void Mask(reg_core* core, int dest, int mask, float rX, float rY, float rTh, float dx1, float dy1,
	float dx2, float dy2) {
	core->mask(dest, mask, rX, rY, rTh, dx1, dy1, dx2, dy2);
}

extern "C" __declspec(dllexport) void PosReg(reg_core* core, int algo_idx, reg_result* result, float x, float y, float th, float th_range, float xy_range)
{
	*result = core->reg(algo_idx, true, x, y, th, th_range, xy_range);
}

extern "C" __declspec(dllexport) void Set(reg_core* core, int algo_idx)
{
	core->set(algo_idx);
}
extern "C" __declspec(dllexport) void Preprocess(reg_core* core, int reg_idx, int algo_idx)
{
	core->preprocess(reg_idx, algo_idx);
}


extern "C" __declspec(dllexport) void DumpRegImage(reg_core* im, int reg_idx, unsigned char* dest)
{
	cudaMemcpy(dest, im->reg_pool[reg_idx], algo_sz*algo_sz, cudaMemcpyDeviceToHost);
}


extern "C" __declspec(dllexport) void DebugDumpAlgoImage(reg_core* im, int algo_idx)
{
    float out[algo_sz*algo_sz];
    char a[20];
    cudaMemcpy(out, im->algo_pool[algo_idx], 320 * 320 * 4, cudaMemcpyDeviceToHost);
	sprintf(a, "algo%d", algo_idx);
	auto ofp = fopen(a, "w");
	if (ofp == NULL) {
		perror("opening output file");
		return;
	}
	for (int i = 0; i < algo_sz*algo_sz; ++i)
		fprintf(ofp, "%f ", (float)out[i]);
	fclose(ofp);
}

extern "C" __declspec(dllexport) void LoadRegImage(reg_core* core, unsigned char* src, int reg_idx)
{
	cudaMemcpy(core->reg_pool[reg_idx], src, algo_sz*algo_sz * sizeof(unsigned char), cudaMemcpyHostToDevice);
}

extern "C" __declspec(dllexport) void LoadAlgoImage(reg_core* core, float* src, int algo_idx)
{
	cudaMemcpy(core->algo_pool[algo_idx], src, algo_sz*algo_sz * sizeof(float), cudaMemcpyHostToDevice);
}

extern "C" __declspec(dllexport) void ApplyMesh(reg_core* core, float* meshX, float* meshY)
{
	float2 mesh[64];
	for (int k = 0; k<64; ++k) {
		mesh[k].x = meshX[k] * core->width;
		mesh[k].y = meshY[k] * core->height;
	}
	core->mesh(mesh);
}

extern "C" __declspec(dllexport) void Init(reg_core* core, int width, int height, float* meshX, float* meshY)
{
	float2 mesh[64];
	for (int k = 0; k<64; ++k) {
		mesh[k].x = meshX[k] * width;
		mesh[k].y = meshY[k] * height;
	}
	core->width = width;
	core->height = height;
	core->init(width, height, mesh);
}
extern "C" __declspec(dllexport) void InitRegOnly(reg_core* core)
{
	core->init_regonly();
}

extern "C" __declspec(dllexport) void Crop(reg_core* core, unsigned char* raw, int reg_idx)
{
	core->cropImage(raw, reg_idx);
}

extern "C" __declspec(dllexport) unsigned const char* Version()
{
	return completeVersion;
}