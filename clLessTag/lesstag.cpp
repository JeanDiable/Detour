#include <cstdio>
#include <cstdlib>

// #include <boost/chrono.hpp>
#include <iostream>
#include <stdarg.h>
#include "lesstag_cl.hpp"
#include "lesstag.h"
#include <chrono>
#include <thread>

#include "tag_info.h"
#include "CImg.h"

#define MAXPLATFORMS 10
#define MAXDEVICES 10

#define group_sz 2048


void abortf(const char* mes, ...) {
	va_list ap;
	va_start(ap, mes);
	vfprintf(stderr, mes, ap);
	va_end(ap);
	exit(-1);
}

cl_device_id simpleGetDevice(char* dev) {
	cl_int ret;
	cl_uint nPlatforms, nTotalDevices = 0;
	cl_platform_id platformIDs[MAXPLATFORMS] ;
	cl_device_id devices[MAXDEVICES];

	char strbuf[1048]; 

	clGetPlatformIDs(MAXPLATFORMS, platformIDs, &nPlatforms); // select first platform
	if (nPlatforms == 0) std::exception("No platform available");

	int p;
	for (p = 0; p < nPlatforms; p++) {
		cl_uint nDevices;
		ret = clGetDeviceIDs(platformIDs[p], CL_DEVICE_TYPE_ALL, MAXDEVICES - nTotalDevices, &devices[nTotalDevices], &nDevices);
		if (ret != CL_SUCCESS) continue;
		nTotalDevices += nDevices;
	}

	fprintf(stderr, "Selecting %s\n", dev);

	for (int i = 0; i < nTotalDevices; i++) {
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 1024, strbuf, NULL);
		fprintf(stderr, "Device %d : %s\n", i, strbuf);
		if (std::string(strbuf).find(dev) != std::string::npos) {
			clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 1024, strbuf, NULL);
			printf("%s ", strbuf);
			clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, 1024, strbuf, NULL);
			printf("%s\n", strbuf);
			return devices[i];
		}
	}

	printf("Device %s not found...\n", dev);
	throw std::exception("Device not found");
}

void __stdcall openclErrorCallback(const char* errinfo, const void* private_info, size_t cb, void* user_data) {
	fprintf(stderr, "* Error callback called, info = %s\n", errinfo);
}

cl_context simpleCreateContext(cl_device_id device) {
	cl_int ret;
	cl_context hContext;

	hContext = clCreateContext(NULL, 1, &device, openclErrorCallback, NULL, &ret);
	if (ret != CL_SUCCESS) abortf("Could not create context : %d\n", ret);

	return hContext;
}

#define max_lines 1024
#define max_quads 256

FILE* fd;

lesstag* init_cllesstag(char* dev, char* type, int w, int h) {
	fd = stdout;// fopen("log.txt", "w"); 
	lesstag* ret = new lesstag();
	ret->w = w;
	ret->h = h;
	ret->device = simpleGetDevice(dev);
	ret->context = simpleCreateContext(ret->device);

	ret->queue = clCreateCommandQueue(ret->context, ret->device, 0, NULL);
	char* source;
	get_source(source, type);  //0:cl2.0, 1:cl1.2
	cl_program program = clCreateProgramWithSource(ret->context, 1, (const char**)&source, 0, NULL);


	cl_int rc = clBuildProgram(program, 1, &ret->device, "-cl-std=CL2.0", NULL, NULL);
	//cl_int rc = clBuildProgram(program, 1, &ret->device, "", NULL, NULL);
	if (rc != CL_SUCCESS) {
		fprintf(stderr, "Could not build program : %d\n", rc);
		if (rc == CL_BUILD_PROGRAM_FAILURE) fprintf(stderr, "CL_BUILD_PROGRAM_FAILURE\n");
		char strbuf[10010];
		if (clGetProgramBuildInfo(program, ret->device, CL_PROGRAM_BUILD_LOG, 10000, strbuf, NULL) == CL_SUCCESS) {
			fprintf(stderr, "Build log follows\n");
			fprintf(stderr, "%s\n", strbuf);
		}
		exit(-1);
	}


	//ret->kernel_bilateral = clCreateKernel(program, "bilateral", NULL);
	ret->kernel_gradient = clCreateKernel(program, "gradient", NULL);
	ret->kernel_prepare_uf = clCreateKernel(program, "prepare_uf", NULL);
	ret->kernel_perform_uf = clCreateKernel(program, "perform_uf", NULL);
	ret->kernel_relabel_points = clCreateKernel(program, "relabel_points", NULL);
	ret->kernel_first_count = clCreateKernel(program, "first_count", NULL);
	ret->kernel_reduce_count = clCreateKernel(program, "reduce_count", NULL);
	ret->kernel_merge_pos = clCreateKernel(program, "merge_pos", NULL);
	ret->kernel_propagate = clCreateKernel(program, "propagate", NULL);
	ret->kernel_calc_line_1 = clCreateKernel(program, "calc_line_1", NULL);
	ret->kernel_calc_line_2 = clCreateKernel(program, "calc_line_2", NULL);
	ret->kernel_reduce_line = clCreateKernel(program, "reduce_line", NULL);
	ret->kernel_line_merge_connect = clCreateKernel(program, "line_merge_connect", NULL);
	ret->kernel_line_merge_unionfind = clCreateKernel(program, "line_merge_unionfind", NULL);
	ret->kernel_line_merge_reverse_list = clCreateKernel(program, "line_merge_reverse_list", NULL);
	ret->kernel_line_merge_calc = clCreateKernel(program, "line_merge_calc", NULL);
	ret->kernel_line_merge_reduce = clCreateKernel(program, "line_merge_reduce", NULL);
	ret->kernel_quadrilateral_prepare = clCreateKernel(program, "quadrilateral_prepare", NULL);
	ret->kernel_quadrilateral_find = clCreateKernel(program, "quadrilateral_find", NULL);
	ret->kernel_homography_transform = clCreateKernel(program, "homography_transform", NULL);
	ret->kernel_rect_decoding = clCreateKernel(program, "rect_decoding", NULL);
	ret->kernel_error_correcting = clCreateKernel(program, "error_correcting", NULL);

	ret->memIm = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, w * h * sizeof(cl_char), NULL, NULL);
	//ret->memIm2 = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, w * h * sizeof(cl_char), NULL, NULL);
	ret->memFlags = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, w * h * sizeof(cl_char), NULL, NULL);
	ret->memGrad = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, w * h * sizeof(cl_uchar2), NULL, NULL);
	ret->memLabel = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, w * h * sizeof(cl_int), NULL, NULL);
	ret->memPos = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, (1 + w * h) * sizeof(cl_int), NULL, NULL);
	ret->memCount1 = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, w * h * sizeof(cl_int), NULL, NULL);
	ret->memLen = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, 2048 * sizeof(cl_int), NULL, NULL);
	ret->memStat = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, max_lines * 2 * 12 * sizeof(cl_int), NULL, NULL);
	ret->memLines = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, max_lines * 12 * sizeof(float), NULL, NULL);
	ret->memLines2 = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, max_lines * 12 * sizeof(float), NULL, NULL);
	ret->memConns = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, max_lines * max_lines * sizeof(cl_int), NULL, NULL);
	ret->memRConn1 = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, max_lines * 64 * sizeof(cl_int), NULL, NULL);
	ret->memRConn2 = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, max_lines * 64 * sizeof(cl_int), NULL, NULL);
	ret->memPoints = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, max_quads * 8 * sizeof(cl_float), NULL, NULL);
	ret->memH = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, max_quads * 8 * sizeof(cl_float), NULL, NULL);
	ret->memRect = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, max_quads * 1600 * sizeof(cl_uchar), NULL, NULL);
	ret->memCoding = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, max_quads * 36 * sizeof(cl_int), NULL, NULL);
	ret->memResults = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, max_quads * 8 * sizeof(cl_int), NULL, NULL);
	ret->memResultsE = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, max_quads * 3 * sizeof(cl_int), NULL, NULL);
	
	ret->cm = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, max_quads * 36 * sizeof(cl_int), NULL, NULL);

	ret->ordering = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, 144 * sizeof(cl_int), NULL, NULL);
	ret->constCodes = clCreateBuffer(ret->context, CL_MEM_READ_WRITE, 12512 * sizeof(cl_int), NULL, NULL);

	ret->bufH = new float[max_quads * 8];
	ret->bufResults = new int[max_quads * 3];

	printf("Initialized on %s\n", dev);
	return ret;
}

#define MAXPASS 5

#define DEBUG_OUTPUT
#define lane_width 32
#define group_comps 256
#define prepare_threads 8192

int frames = 0;

float* Lines = new float[1024 * 12]; 
int* Stats = new int[1024 * 12 * 2];
int* bufLabel = new int[1024 * 1024];

int perform_lesstag(lesstag* ret, unsigned char* bufIm, detected_code_im* result, int d_level) {
	frames += 1;
	bool debug = false, debug_deep = false;
	d_level = 0;
	if (d_level > 0) debug = true;
	if (d_level > 1) debug_deep = true;
	debug=debug_deep = false;

	if (debug) {
		fprintf(fd, ">>> starting %d...\n", frames);
		fflush(fd);
	}

	int N = ret->w * ret->h;
	int bufLen[1024];
	clEnqueueWriteBuffer(ret->queue, ret->memIm, false, 0, N, bufIm, 0, NULL, NULL);
	if (debug) {
		clFinish(ret->queue);
		fprintf(fd, "copy\n");
		fflush(fd);
	}

	int init_stat[] = { 0 };

	size_t work_size[2] = { ret->w,ret->h };
	size_t local_work_size[2] = { 16,16 };

	clEnqueueFillBuffer(ret->queue, ret->memCount1, init_stat, sizeof(cl_int), 0, ret->w * ret->h * sizeof(cl_int), 0, NULL, NULL);
	clEnqueueFillBuffer(ret->queue, ret->memStat, init_stat, sizeof(cl_int), 0, max_lines * 2 * 12 * sizeof(cl_int), 0, NULL, NULL);

	//clSetKernelArg(ret->kernel_bilateral, 0, sizeof(cl_mem), (void *)&ret->memIm);
	//clSetKernelArg(ret->kernel_bilateral, 1, sizeof(cl_mem), (void *)&ret->memIm2);
	//clSetKernelArg(ret->kernel_bilateral, 2, sizeof(cl_int), (int *)&ret->w);
	//clSetKernelArg(ret->kernel_bilateral, 3, sizeof(cl_int), (int *)&ret->h);
	//clEnqueueNDRangeKernel(ret->queue, ret->kernel_bilateral, 2, NULL, work_size, local_work_size, 0, NULL, NULL);


	//cv::Mat X(cv::Size(ret->w, ret->h), CV_8U);
	//clEnqueueReadBuffer(ret->queue, ret->memIm2, CL_TRUE, 0, ret->w*ret->h * sizeof(cl_char), X.data, 0, NULL, NULL);
	//cv::imwrite("grad.bmp", X);

	clSetKernelArg(ret->kernel_gradient, 0, sizeof(cl_mem), (void*)&ret->memIm);
	clSetKernelArg(ret->kernel_gradient, 1, sizeof(cl_mem), (void*)&ret->memGrad);
	clSetKernelArg(ret->kernel_gradient, 2, sizeof(cl_int), (int*)&ret->w);
	clSetKernelArg(ret->kernel_gradient, 3, sizeof(cl_int), (int*)&ret->h);
	clEnqueueNDRangeKernel(ret->queue, ret->kernel_gradient, 2, NULL, work_size, local_work_size, 0, NULL, NULL);

	//
	if (debug) {
		clFinish(ret->queue);
		fprintf(fd, "grad\n");
		fflush(fd);
	}

	clSetKernelArg(ret->kernel_prepare_uf, 0, sizeof(cl_mem), (void*)&ret->memLabel);
	clSetKernelArg(ret->kernel_prepare_uf, 1, sizeof(cl_mem), (void*)&ret->memGrad);
	clSetKernelArg(ret->kernel_prepare_uf, 2, sizeof(cl_mem), (void*)&ret->memFlags);
	clSetKernelArg(ret->kernel_prepare_uf, 3, sizeof(cl_int), (int*)&ret->w);
	clSetKernelArg(ret->kernel_prepare_uf, 4, sizeof(cl_int), (int*)&ret->h);
	clEnqueueNDRangeKernel(ret->queue, ret->kernel_prepare_uf, 2, NULL, work_size, local_work_size, 0, NULL, NULL);


	for (int i = 0; i < MAXPASS; i++) {
		clSetKernelArg(ret->kernel_perform_uf, 0, sizeof(cl_mem), (void*)&ret->memLabel);
		clSetKernelArg(ret->kernel_perform_uf, 1, sizeof(cl_mem), (void*)&ret->memFlags);
		clSetKernelArg(ret->kernel_perform_uf, 2, sizeof(cl_int), (int*)&ret->w);
		clSetKernelArg(ret->kernel_perform_uf, 3, sizeof(cl_int), (int*)&ret->h);
		clEnqueueNDRangeKernel(ret->queue, ret->kernel_perform_uf, 2, NULL, work_size, local_work_size, 0, NULL, NULL);
	}
	//
	size_t work_CP_G[1] = { ret->w * ret->h };

	//clFinish(ret->queue);
	if (debug) {
		clFinish(ret->queue);
		fprintf(fd, "uf\n");
		fflush(fd);
	}

	clSetKernelArg(ret->kernel_first_count, 0, sizeof(cl_mem), (void*)&ret->memLabel);
	clSetKernelArg(ret->kernel_first_count, 1, sizeof(cl_mem), (void*)&ret->memCount1);
	clSetKernelArg(ret->kernel_first_count, 2, sizeof(cl_int), (void*)&N);
	clEnqueueNDRangeKernel(ret->queue, ret->kernel_first_count, 1, NULL, work_CP_G, NULL, 0, NULL, NULL);



	clSetKernelArg(ret->kernel_reduce_count, 0, sizeof(cl_mem), (void*)&ret->memLabel);
	clSetKernelArg(ret->kernel_reduce_count, 1, sizeof(cl_mem), (void*)&ret->memCount1);
	clSetKernelArg(ret->kernel_reduce_count, 2, sizeof(cl_int), (void*)&N);
	clEnqueueNDRangeKernel(ret->queue, ret->kernel_reduce_count, 1, NULL, work_CP_G, NULL, 0, NULL, NULL);
	//2.7-3.0ms

	int group_n = (N / group_sz + (N % group_sz > 0));
	//nv patch:
	clSetKernelArg(ret->kernel_relabel_points, 0, sizeof(cl_mem), (void*)&ret->memLabel);
	clSetKernelArg(ret->kernel_relabel_points, 1, sizeof(cl_mem), (void*)&ret->memLen);
	clSetKernelArg(ret->kernel_relabel_points, 2, sizeof(cl_mem), (int*)&ret->memPos);
	clSetKernelArg(ret->kernel_relabel_points, 3, sizeof(cl_int), (int*)&N);
	size_t work_RP_G[1] = { group_n * lane_width };
	size_t work_MP_G[1] = { lane_width };
	clEnqueueNDRangeKernel(ret->queue, ret->kernel_relabel_points, 1, NULL, work_RP_G, work_MP_G, 0, NULL, NULL);

	if (debug_deep) {
		clFinish(ret->queue);
		clEnqueueReadBuffer(ret->queue, ret->memLen, CL_TRUE, 0, group_n * sizeof(cl_int), bufLen, 0, NULL, NULL);
		fprintf(fd, "relabel %d >>\n", group_n);
		for (int i = 0; i < group_n; ++i) //150items. 
			fprintf(fd, "%d ", bufLen[i]);
		fprintf(fd, "\n");
		fflush(fd);
	}


	clSetKernelArg(ret->kernel_merge_pos, 0, sizeof(cl_mem), (void*)&ret->memLabel);
	clSetKernelArg(ret->kernel_merge_pos, 1, sizeof(cl_mem), (void*)&ret->memLen);
	clSetKernelArg(ret->kernel_merge_pos, 2, sizeof(cl_mem), (int*)&ret->memPos);
	clSetKernelArg(ret->kernel_merge_pos, 3, sizeof(cl_int), (int*)&group_n);
	clEnqueueNDRangeKernel(ret->queue, ret->kernel_merge_pos, 1, NULL, work_MP_G, work_MP_G, 0, NULL, NULL);

	//3ms

	clSetKernelArg(ret->kernel_propagate, 0, sizeof(cl_mem), (void*)&ret->memLabel);
	clSetKernelArg(ret->kernel_propagate, 1, sizeof(cl_int), (void*)&N);
	clEnqueueNDRangeKernel(ret->queue, ret->kernel_propagate, 1, NULL, work_CP_G, NULL, 0, NULL, NULL);
	//3.1-3.2ms

	//clFinish(ret->queue);
	if (debug) {
		clEnqueueReadBuffer(ret->queue, ret->memLen, CL_TRUE, 0, 4 * sizeof(cl_int), bufLen, 0, NULL, NULL);
		clFinish(ret->queue);
		fprintf(fd, "propagate, l:%d\n", bufLen[0]);
		fflush(fd);
	}

	clSetKernelArg(ret->kernel_calc_line_1, 0, sizeof(cl_mem), (void*)&ret->memLabel);
	clSetKernelArg(ret->kernel_calc_line_1, 1, sizeof(cl_mem), (int*)&ret->memGrad);
	clSetKernelArg(ret->kernel_calc_line_1, 2, sizeof(cl_mem), (int*)&ret->memStat);
	clSetKernelArg(ret->kernel_calc_line_1, 3, sizeof(cl_int), (int*)&ret->w);
	clEnqueueNDRangeKernel(ret->queue, ret->kernel_calc_line_1, 2, NULL, work_size, NULL, 0, NULL, NULL);

	//3.4-3.6ms

	clSetKernelArg(ret->kernel_calc_line_2, 0, sizeof(cl_mem), (int*)&ret->memStat);
	clSetKernelArg(ret->kernel_calc_line_2, 1, sizeof(cl_mem), (int*)&ret->memLen);
	clSetKernelArg(ret->kernel_calc_line_2, 2, sizeof(cl_mem), (float*)&ret->memLines);
	size_t work_calc_2[1] = { max_lines };
	clEnqueueNDRangeKernel(ret->queue, ret->kernel_calc_line_2, 1, NULL, work_calc_2, work_MP_G, 0, NULL, NULL);


	clSetKernelArg(ret->kernel_reduce_line, 0, sizeof(cl_mem), (int*)&ret->memLen);
	clSetKernelArg(ret->kernel_reduce_line, 1, sizeof(cl_mem), (float*)&ret->memLines);
	clSetKernelArg(ret->kernel_reduce_line, 2, sizeof(cl_mem), (float*)&ret->memLines2);
	clEnqueueNDRangeKernel(ret->queue, ret->kernel_reduce_line, 1, NULL, work_MP_G, work_MP_G, 0, NULL, NULL);

	//clFinish(ret->queue);
	if (debug) {
		clFinish(ret->queue);
		clEnqueueReadBuffer(ret->queue, ret->memLen, CL_TRUE, 0, 4 * sizeof(cl_int), bufLen, 0, NULL, NULL);
		fprintf(fd, "calc line> %d %d %d %d\n", bufLen[0], bufLen[1], bufLen[2], bufLen[3]);
		fflush(fd);


		if (debug_deep) {
			clEnqueueReadBuffer(ret->queue, ret->memLines2, CL_TRUE, 0, max_lines * 12 * sizeof(float), Lines, 0, NULL, NULL);
			clEnqueueReadBuffer(ret->queue, ret->memStat, CL_TRUE, 0, max_lines * 12 * sizeof(float) * 2, Stats, 0, NULL, NULL);

			cimg_library::CImg<unsigned char> img(ret->w, ret->h, 1, 3);
			img.fill(0);

			for (int i = 0; i < bufLen[0]; ++i) {
				int rgb = (i * 1103515245 + 129389741) % 1300813;
				auto r = rgb & 0xff; rgb >>= 8;
				auto g = rgb & 0xff; rgb >>= 8;
				auto b = rgb & 0xff; rgb >>= 8;

				if (d_level > 3)
					printf("%d(%d): %d, %d, %d, %d, %d\n", Stats[i * 12], Stats[i * 12 + 7], Stats[i * 12 + 1], Stats[i * 12 + 2], Stats[i * 12 + 3], Stats[i * 12 + 4]);
				const unsigned char color[] = { r,g,b };
				img.draw_line(Lines[i * 12 + 8], Lines[i * 12 + 9], Lines[i * 12 + 10], Lines[i * 12 + 11], color);
				char buf[20];
				sprintf_s(buf, "L%d", i);
				// img.draw_text(Lines[i * 12 + 8], Lines[i * 12 + 9], buf, color, 0, 1, std::rand() % 2 ? 38 : 57);
			}
			char fn[20];
			sprintf(fn, "debug\\%d a.bmp", frames);
			img.save_bmp(fn);
		}

	}

	cl_mem in_lines = ret->memLines2;
	cl_mem out_lines = ret->memLines;

	size_t work_lmc2[1] = { prepare_threads };
	for (int p = 0; p < 2; ++p) {
		clEnqueueFillBuffer(ret->queue, ret->memRConn1, init_stat, sizeof(cl_int), 0, max_lines * 64 * sizeof(cl_int), 0, NULL, NULL);
		clEnqueueFillBuffer(ret->queue, ret->memRConn2, init_stat, sizeof(cl_int), 0, max_lines * 64 * sizeof(cl_int), 0, NULL, NULL);
		//clFinish(ret->queue);
		if (debug) {
			clFinish(ret->queue);
			fprintf(fd, "fill rconn\n");
			fflush(fd);
		}

		clSetKernelArg(ret->kernel_line_merge_connect, 0, sizeof(cl_mem), &in_lines);
		clSetKernelArg(ret->kernel_line_merge_connect, 1, sizeof(cl_mem), (void*)&ret->memLen);
		clSetKernelArg(ret->kernel_line_merge_connect, 2, sizeof(cl_mem), (void*)&ret->memRConn1);
		size_t work_lmc[2] = { max_lines, max_lines };
		clEnqueueNDRangeKernel(ret->queue, ret->kernel_line_merge_connect, 1, NULL, work_lmc2, NULL, 0, NULL, NULL);

		if (debug) {
			clFinish(ret->queue);
			fprintf(fd, "merge_connect\n");
			fflush(fd);
		}

		size_t work_klmuf[1] = { max_lines };
		for (int i = 0; i < 5; ++i) {
			clSetKernelArg(ret->kernel_line_merge_unionfind, 0, sizeof(cl_mem), (void*)&ret->memRConn1);
			clSetKernelArg(ret->kernel_line_merge_unionfind, 1, sizeof(cl_mem), (float*)&ret->memLen);
			clEnqueueNDRangeKernel(ret->queue, ret->kernel_line_merge_unionfind, 1, NULL, work_klmuf, NULL, 0, NULL, NULL);
		}

		//clFinish(ret->queue);
		if (debug) {
			clFinish(ret->queue);
			fprintf(fd, "union_find\n");
			fflush(fd);
		}

		clSetKernelArg(ret->kernel_line_merge_reverse_list, 0, sizeof(cl_mem), (void*)&ret->memRConn1);
		clSetKernelArg(ret->kernel_line_merge_reverse_list, 1, sizeof(cl_mem), (float*)&ret->memLen);
		clSetKernelArg(ret->kernel_line_merge_reverse_list, 2, sizeof(cl_mem), (float*)&ret->memRConn2);
		clEnqueueNDRangeKernel(ret->queue, ret->kernel_line_merge_reverse_list, 1, NULL, work_klmuf, NULL, 0, NULL, NULL);

		if (debug) {
			clFinish(ret->queue);
			fprintf(fd, "reverse\n");
			fflush(fd);
		}

		clSetKernelArg(ret->kernel_line_merge_calc, 0, sizeof(cl_mem), (void*)&ret->memRConn2);
		clSetKernelArg(ret->kernel_line_merge_calc, 1, sizeof(cl_mem), (float*)&ret->memLen);
		clSetKernelArg(ret->kernel_line_merge_calc, 2, sizeof(cl_mem), &in_lines);
		clEnqueueNDRangeKernel(ret->queue, ret->kernel_line_merge_calc, 1, NULL, work_klmuf, NULL, 0, NULL, NULL);


		if (debug) {
			clFinish(ret->queue);
			fprintf(fd, "merge_calc\n");
			fflush(fd);
		}

		clSetKernelArg(ret->kernel_line_merge_reduce, 0, sizeof(cl_mem), (void*)&ret->memConns);
		clSetKernelArg(ret->kernel_line_merge_reduce, 1, sizeof(cl_mem), &in_lines);
		clSetKernelArg(ret->kernel_line_merge_reduce, 2, sizeof(cl_mem), &out_lines);
		clSetKernelArg(ret->kernel_line_merge_reduce, 3, sizeof(cl_mem), (float*)&ret->memLen);
		clSetKernelArg(ret->kernel_line_merge_reduce, 4, sizeof(cl_mem), (void*)&ret->memRConn2);
		clEnqueueNDRangeKernel(ret->queue, ret->kernel_line_merge_reduce, 1, NULL, work_MP_G, work_MP_G, 0, NULL, NULL);

		if (debug) {
			clFinish(ret->queue);

			clEnqueueReadBuffer(ret->queue, ret->memLen, CL_TRUE, 0, 4 * sizeof(cl_int), bufLen, 0, NULL, NULL);
			int Rconn2[64 * max_lines];
			clEnqueueReadBuffer(ret->queue, ret->memRConn2, CL_TRUE, 0, max_lines * 64 * sizeof(cl_int), Rconn2, 0, NULL, NULL);
			fprintf(fd, "merge line> %d %d %d %d\n", bufLen[0], bufLen[1], bufLen[2], bufLen[3]);
			if (d_level > 10) {
				for (int i = 0; i < bufLen[3]; ++i) {
					fprintf(fd, "L%d %d>", i, Rconn2[64 * i]);
					for (int j = 0; j < Rconn2[64 * i]; ++j)
						fprintf(fd, "%d ", Rconn2[64 * i + j + 1]);
					fprintf(fd, "\n");
				}
				fflush(fd);
			}

			if (debug_deep) {
				clEnqueueReadBuffer(ret->queue, out_lines, CL_TRUE, 0, max_lines * 12 * sizeof(float), Lines, 0, NULL, NULL);
				cimg_library::CImg<unsigned char> img(ret->w, ret->h,1,3);
				img.fill(0);

				for (int i = 0; i < bufLen[0]; ++i) {
					int rgb = (i * 1103515245 + 129389741) % 1300813;
					auto r = rgb & 0xff; rgb >>= 8;
					auto g = rgb & 0xff; rgb >>= 8;
					auto b = rgb & 0xff; rgb >>= 8;
					const unsigned char color[] = { r,g,b };
					//printf("%f,%f -%f,%f\n", Lines[i * 12 + 8], Lines[i * 12 + 9], Lines[i * 12 + 10], Lines[i * 12 + 11]);
					img.draw_line(Lines[i * 12 + 8], Lines[i * 12 + 9], Lines[i * 12 + 10], Lines[i * 12 + 11], color);char buf[20];
					sprintf_s(buf, "L%d", i);
					// img.draw_text(Lines[i * 12 + 8], Lines[i * 12 + 9], buf, color, 0, 1, std::rand() % 2 ? 38 : 57);

				}
				char fn[20]; 
				sprintf(fn, "debug\\%d merged%d.bmp", frames, p);
				img.save_bmp(fn);
			}

		}

		auto tmp = in_lines;
		in_lines = out_lines;
		out_lines = in_lines;
	}

	clEnqueueFillBuffer(ret->queue, ret->memRConn1, init_stat, sizeof(cl_int), 0, max_lines * 64 * sizeof(cl_int), 0, NULL, NULL);

	clSetKernelArg(ret->kernel_quadrilateral_prepare, 0, sizeof(cl_mem), (void*)&ret->memConns);
	clSetKernelArg(ret->kernel_quadrilateral_prepare, 1, sizeof(cl_mem), &in_lines);
	clSetKernelArg(ret->kernel_quadrilateral_prepare, 2, sizeof(cl_mem), (void*)&ret->memLen);
	clSetKernelArg(ret->kernel_quadrilateral_prepare, 3, sizeof(cl_mem), (void*)&ret->memRConn1);
	clEnqueueNDRangeKernel(ret->queue, ret->kernel_quadrilateral_prepare, 1, NULL, work_lmc2, NULL, 0, NULL, NULL);

	size_t work_qr[] = { max_lines };
	clSetKernelArg(ret->kernel_quadrilateral_find, 0, sizeof(cl_mem), (void*)&ret->memConns);
	clSetKernelArg(ret->kernel_quadrilateral_find, 1, sizeof(cl_mem), &in_lines);
	clSetKernelArg(ret->kernel_quadrilateral_find, 2, sizeof(cl_mem), (void*)&ret->memLen);
	clSetKernelArg(ret->kernel_quadrilateral_find, 3, sizeof(cl_mem), (void*)&ret->memRConn1);
	clSetKernelArg(ret->kernel_quadrilateral_find, 4, sizeof(cl_mem), (void*)&ret->memPoints);
	clSetKernelArg(ret->kernel_quadrilateral_find, 5, sizeof(cl_mem), (void*)&ret->memH);
	clEnqueueNDRangeKernel(ret->queue, ret->kernel_quadrilateral_find, 1, NULL, work_qr, NULL, 0, NULL, NULL);

	//clFinish(ret->queue);
	if (debug) {
		clFinish(ret->queue);
		clEnqueueReadBuffer(ret->queue, ret->memLen, CL_TRUE, 0, 4 * sizeof(cl_int), bufLen, 0, NULL, NULL);
		fprintf(fd, "quads > %d %d %d %d\n", bufLen[0], bufLen[1], bufLen[2], bufLen[3]);
		fflush(fd);

	}

	size_t work_ht[] = { max_quads * lane_width };
	clSetKernelArg(ret->kernel_homography_transform, 0, sizeof(cl_mem), (void*)&ret->memIm);
	clSetKernelArg(ret->kernel_homography_transform, 1, sizeof(cl_mem), (int*)&ret->memH);
	clSetKernelArg(ret->kernel_homography_transform, 2, sizeof(cl_mem), (void*)&ret->memLen);
	clSetKernelArg(ret->kernel_homography_transform, 3, sizeof(cl_mem), (void*)&ret->memRect);
	clSetKernelArg(ret->kernel_homography_transform, 4, sizeof(cl_int), (void*)&ret->w);
	clSetKernelArg(ret->kernel_homography_transform, 5, sizeof(cl_int), (void*)&ret->h);
	clEnqueueNDRangeKernel(ret->queue, ret->kernel_homography_transform, 1, NULL, work_ht, work_MP_G, 0, NULL, NULL);

	// mapping
	clEnqueueWriteBuffer(ret->queue, ret->ordering, true, 0, 144 * sizeof(cl_int), ordering, 0, NULL, NULL);

	clSetKernelArg(ret->kernel_rect_decoding, 0, sizeof(cl_mem), (void*)&ret->memRect);
	clSetKernelArg(ret->kernel_rect_decoding, 1, sizeof(cl_mem), (void*)&ret->memLen);
	clSetKernelArg(ret->kernel_rect_decoding, 2, sizeof(cl_int), (void*)&ret->w);
	clSetKernelArg(ret->kernel_rect_decoding, 3, sizeof(cl_mem), (void*)&ret->memResults);
	clSetKernelArg(ret->kernel_rect_decoding, 4, sizeof(cl_mem), (void*)&ret->ordering);
	clSetKernelArg(ret->kernel_rect_decoding, 5, sizeof(cl_mem), (void*)&ret->cm);
	clEnqueueNDRangeKernel(ret->queue, ret->kernel_rect_decoding, 1, NULL, work_ht, work_MP_G, 0, NULL, NULL);

	clEnqueueWriteBuffer(ret->queue, ret->constCodes, true, 0, 12512 * sizeof(cl_int), codes, 0, NULL, NULL);

	clSetKernelArg(ret->kernel_error_correcting, 0, sizeof(cl_mem), (void*)&ret->memResults);
	clSetKernelArg(ret->kernel_error_correcting, 1, sizeof(cl_mem), (void*)&ret->memLen);
	clSetKernelArg(ret->kernel_error_correcting, 2, sizeof(cl_mem), (int*)&ret->memResultsE);
	clSetKernelArg(ret->kernel_error_correcting, 3, sizeof(cl_mem), (int*)&ret->constCodes);
	size_t work_ht2[] = { 12512 };
	clEnqueueNDRangeKernel(ret->queue, ret->kernel_error_correcting, 1, NULL, work_ht2, NULL, 0, NULL, NULL);


	clFinish(ret->queue);

	clEnqueueReadBuffer(ret->queue, ret->memH, CL_TRUE, 0, max_quads * 8 * sizeof(cl_int), ret->bufH, 0, NULL, NULL);
	clEnqueueReadBuffer(ret->queue, ret->memResultsE, CL_TRUE, 0, max_quads * 3 * sizeof(cl_int), ret->bufResults, 0, NULL, NULL);
	clEnqueueReadBuffer(ret->queue, ret->memLen, CL_TRUE, 0, 4 * sizeof(cl_int), bufLen, 0, NULL, NULL);

	if (debug) {
		clFinish(ret->queue);
		fprintf(fd, "fin> %d %d %d %d\n", bufLen[0], bufLen[1], bufLen[2], bufLen[3]);
		fflush(fd); 

		// if (debug_deep) {
			int* R = new int[1024 * 3];
			clEnqueueReadBuffer(ret->queue, ret->memResults, CL_TRUE, 0, max_quads * 8 * sizeof(cl_int), R, 0, NULL, NULL);
			for (int i = 0; i < bufLen[1]; ++i)
				printf("..>> %x,%d %x,%d %x,%d %x,%d\n", R[i * 8], R[i * 8 + 1], R[i * 8 + 2], R[i * 8 + 3], R[i * 8 + 4], R[i * 8 + 5], R[i * 8 + 6], R[i * 8 + 7]);

			clEnqueueReadBuffer(ret->queue, ret->memLen, CL_TRUE, 0, (5 + 24) * sizeof(cl_int), bufLen, 0, NULL, NULL);
			//printf("orderings:");
			//for (int i = 0; i < 24; ++i)
			//	printf("%d ", bufLen[i + 5]);
			//printf("\n");
			//
			clEnqueueReadBuffer(ret->queue, ret->cm, CL_TRUE, 0, 64 * sizeof(cl_int), R, 0, NULL, NULL);
			for (int i=0; i<8; ++i)
			{
				for (int j = 0; j < 8; ++j)
					printf("%d\t", R[i * 8 + j]);
				printf("\n");
			}
			
		// }
	}

	for (int i = 0; i < bufLen[2]; ++i) {
		int id = ret->bufResults[i * 3 + 0];
		int rot = ret->bufResults[i * 3 + 1];
		int H = ret->bufResults[i * 3 + 2];

		float x = ret->bufH[H * 8 + 2];
		float y = ret->bufH[H * 8 + 5];
		float th = -rot * 3.1415926 / 2.0;

		float c = cos(th), s = sin(th); // this is tag left dir.
		float lX = (ret->bufH[H * 8 + 0] * c + ret->bufH[H * 8 + 1] * s + ret->bufH[H * 8 + 2]) /
			(ret->bufH[H * 8 + 6] * c + ret->bufH[H * 8 + 7] * s + 1);
		float lY = (ret->bufH[H * 8 + 3] * c + ret->bufH[H * 8 + 4] * s + ret->bufH[H * 8 + 5]) /
			(ret->bufH[H * 8 + 6] * c + ret->bufH[H * 8 + 7] * s + 1);

		th = th + 3.1415926 / 4;
		c = cos(th) * 1.414, s = sin(th) * 1.414;
		float x1 = (ret->bufH[H * 8 + 0] * c + ret->bufH[H * 8 + 1] * s + ret->bufH[H * 8 + 2]) /
			(ret->bufH[H * 8 + 6] * c + ret->bufH[H * 8 + 7] * s + 1);
		float y1 = (ret->bufH[H * 8 + 3] * c + ret->bufH[H * 8 + 4] * s + ret->bufH[H * 8 + 5]) /
			(ret->bufH[H * 8 + 6] * c + ret->bufH[H * 8 + 7] * s + 1);
		th += 3.1415926 / 2;
		c = cos(th) * 1.414, s = sin(th) * 1.414;
		float x2 = (ret->bufH[H * 8 + 0] * c + ret->bufH[H * 8 + 1] * s + ret->bufH[H * 8 + 2]) /
			(ret->bufH[H * 8 + 6] * c + ret->bufH[H * 8 + 7] * s + 1);
		float y2 = (ret->bufH[H * 8 + 3] * c + ret->bufH[H * 8 + 4] * s + ret->bufH[H * 8 + 5]) /
			(ret->bufH[H * 8 + 6] * c + ret->bufH[H * 8 + 7] * s + 1);
		th += 3.1415926 / 2;
		c = cos(th) * 1.414, s = sin(th) * 1.414;
		float x3 = (ret->bufH[H * 8 + 0] * c + ret->bufH[H * 8 + 1] * s + ret->bufH[H * 8 + 2]) /
			(ret->bufH[H * 8 + 6] * c + ret->bufH[H * 8 + 7] * s + 1);
		float y3 = (ret->bufH[H * 8 + 3] * c + ret->bufH[H * 8 + 4] * s + ret->bufH[H * 8 + 5]) /
			(ret->bufH[H * 8 + 6] * c + ret->bufH[H * 8 + 7] * s + 1);
		th += 3.1415926 / 2;
		c = cos(th) * 1.414, s = sin(th) * 1.414;
		float x4 = (ret->bufH[H * 8 + 0] * c + ret->bufH[H * 8 + 1] * s + ret->bufH[H * 8 + 2]) /
			(ret->bufH[H * 8 + 6] * c + ret->bufH[H * 8 + 7] * s + 1);
		float y4 = (ret->bufH[H * 8 + 3] * c + ret->bufH[H * 8 + 4] * s + ret->bufH[H * 8 + 5]) /
			(ret->bufH[H * 8 + 6] * c + ret->bufH[H * 8 + 7] * s + 1);

		float roll = atan2(y - lY, x - lX);

		//x1 = bufPoints[H * 8 + 0], y1 = bufPoints[H * 8 + 1];
		//x2 = bufPoints[H * 8 + 2], y2 = bufPoints[H * 8 + 3];
		//x3 = bufPoints[H * 8 + 4], y3 = bufPoints[H * 8 + 5];
		//x4 = bufPoints[H * 8 + 6], y4 = bufPoints[H * 8 + 7];

		result[i].id = id;
		result[i].x = x;
		result[i].y = y;
		result[i].x1 = x1;
		result[i].y1 = y1;
		result[i].x2 = x2;
		result[i].y2 = y2;
		result[i].x3 = x3;
		result[i].y3 = y3;
		result[i].x4 = x4;
		result[i].y4 = y4;
		result[i].roll2d = roll / 3.1415926 * 180;
		if (debug) {
			printf("$ Tag %d %d: (%f,%f)->(%f,%f) r %d\n", i, id, x, y, lX, lY, rot);
		}
	}

	if (debug) {
		printf("$ CLLessTag> detected %d lesstags, %d quads, %d lines, %d segs ...\n", bufLen[2], bufLen[1], bufLen[0], bufLen[3]);
		fprintf(fd, "$ CLLessTag> detected %d lesstags, %d quads, %d lines, %d segs ...\n", bufLen[2], bufLen[1], bufLen[0], bufLen[3]);
	}

	return bufLen[2];
	// everything's done.
}
