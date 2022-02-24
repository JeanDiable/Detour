#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "reg.h"
#include <cufft.h>

template<int N>
struct RotParams {
	float sinTheta[N], cosTheta[N], biasX[N], biasY[N];
};

struct ABSRotParm {
	float baseSin[theta_num], baseCos[theta_num], stepSin, stepCos;
};

struct RotfunStruct
{
	float *interm;
	cudaChannelFormatDesc channelDesc;
	Complex* fft_out_data;
	cufftHandle fftPlan;
	RotParams<rotfun_num> params1;
	ABSRotParm params2;
	cudaTextureObject_t croppedTex;
	cudaSurfaceObject_t pSurfObject;

	cudaTextureObject_t FFTcroppedTex;
	cudaSurfaceObject_t pFFTSurfObject;
};

struct ThetaFunStruct
{
	cufftHandle fftPlan;
	float* fftTmp;
	float* fftAbs;
	float theta_results[rot_check_num];

	Complex* pc_dest;
	float tconf;
	float tconf_m;
	//	int* tf_result;
	//	float* peaks;
	//	void* peak_finding_tmp;
	//	size_t temp_storage_bytes;
};

struct ImProcessStruct
{
	unsigned char* lut;
	unsigned char* tmp1;
	unsigned char* tmp2;
	float2* mesh;
	int height;
	int width;
	int* hist;
	cudaArray* cuArray;
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;
	cudaSurfaceObject_t pSurfObject;
	cudaTextureObject_t rawTex;
	unsigned char* filt_tmp;
};
struct maxima_struct
{
	float max_val;
	float x;
	float y;
};
struct conf_pair
{
	float sum_val;
	float sum2_val;
	float max_val;
};

struct PhaseCorrelationStruct
{
	cufftHandle fftPlan_i$f, fftPlan_i$f2;
	cufftHandle fftPlanCoarse, fftPlanFine, fftPlanFine2;

	RotParams<rot_check_num> p1; //coarse theta
	Complex *interm, *interm2, *interm3, *i2f, *i2f2;
	Complex *i1f, *i1f2;
	float *absed, *absed2, *g1;
	float coarseMaxConf;
	float coarseTheta;

	conf_pair *conf_coarse, *conf_fine, *conf_fine2, *conf1, *conf2;

	float fineTheta, fineConf;
	maxima_struct *block_maxima1, *block_maxima2;
	Complex* peaks;
	float finalY;
	float finalX;

	float scope;
	float e, std;
	float phConf;
	float phConf2;
};

extern "C" void genTexture(RotfunStruct& rot_struct, float* src);
extern "C" void rotfun(RotfunStruct& rot_struct, float* src, float* dest);
extern "C" void initRotFun(RotfunStruct& rot_struct);

extern "C" void applyMesh(ImProcessStruct& ip_struct, float2* mesh);
extern "C" void initImageProcessing(ImProcessStruct& ip_struct);
extern "C" void initCLAHE(ImProcessStruct& ip_struct);
extern "C" void applyCLAHE(ImProcessStruct& ip_struct, unsigned char* _src, unsigned char* _dst);
extern "C" void initCropping(ImProcessStruct& ip_struct, int width, int height, float2* mesh);
extern "C" void cropIm(ImProcessStruct& ip_struct, unsigned char* raw_input, unsigned char* gpu_cropped);
extern "C" void preProcess(ImProcessStruct& ip_struct, unsigned char* gpu_cropped, float* ci);
extern "C" void applyMask(unsigned char* str, unsigned char* text, float r_x, float r_y, float r_th, float dx1, float dy1,
	float dx2, float dy2);


extern "C" void initThetaFinding(ThetaFunStruct& tf_struct);
extern "C" void computeRotFFT(ThetaFunStruct& tf_struct, float* roted, Complex* &target);
extern "C" void thetaFinding(ThetaFunStruct& tf_struct, Complex* roted_ci_ffted, Complex* roted_templ);


extern "C" void initPhaseCorrelation(PhaseCorrelationStruct& pc_struct);
extern "C" void i1fft(PhaseCorrelationStruct& pc_struct, float* image);
extern "C" bool coarseTheta(PhaseCorrelationStruct& pc_struct, RotfunStruct& rot_struct, ThetaFunStruct& tf_struct);
extern "C" void fineTheta(PhaseCorrelationStruct& pc_struct, RotfunStruct& rot_struct, ThetaFunStruct& tf_struct, bool twice = false);
extern "C" void phase(PhaseCorrelationStruct& pc_struct, RotfunStruct& rot_struct, ThetaFunStruct& tf_struct, bool pos = false, float4 xyd = {0,0,99999});