#pragma once
#include "reg.h"
#include "kernels.cuh"
#include <thread>
#include <mutex>

#define max_pool 256

struct reg_result
{
	float x, y, theta, conf;
};

struct xy
{
	float x;
	float y;
};

class reg_core
{
private:

	RotfunStruct rotfun_struct;
	ImProcessStruct ip_struct;
	ThetaFunStruct tf_struct;
	PhaseCorrelationStruct pc_struct;
	float *i1rBuf, *i2rBuf;
	Complex *i1pf, *i2pf;
	std::thread::id thisThread;

	void checkThread();
public:
	int frames = 0;
	unsigned char* reg_pool[max_pool];
	float* algo_pool[max_pool];

	void cropImage(unsigned char* rawIn, int reg_im_idx);
	void init(int width, int height, float2* mesh);
	void init_regonly();
	void preprocess(int reg_im_idx, int algo_im_idx);
	void set(int algo_im_idx);
	void mesh(float2* mesh);
	reg_result reg(int algo_im_idx, bool pos = false, float x = 0, float y = 0, float th = 0, float th_range=1.0f, float xy_range=9999.0f);
	void mask(int dest, int mask, float r_x, float r_y, float r_th, float dx1, float dy1,
	         float dx2, float dy2);

	int width, height;
	static reg_core* getInstance();
};
