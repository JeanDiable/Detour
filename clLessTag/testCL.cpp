/*****************************************************************************
* Copyright (c) 2013-2016 Intel Corporation
* All rights reserved.
*
* WARRANTY DISCLAIMER
*
* THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
* MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* Intel Corporation is the author of the Materials, and requests that all
* problem reports or change requests be submitted to it directly
*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <tchar.h>
#include <memory.h>
#include <vector>

#include "CL\cl.h"
#include "utils.h"

//for perf. counters
#include <Windows.h>
#include "helpers.hpp"

#define MSTRINGIFY(A) #A
char* cl_scan_algo =
#include "scan_algo.cl"
;

int CreateAndBuildProgram(opencl_common *ocl, char* source)
{
	cl_int err = CL_SUCCESS;

	// And now after you obtained a regular C string call clCreateProgramWithSource to create OpenCL program object.
	ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**)&source, 0, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateProgramWithSource returned %s.\n", TranslateOpenCLError(err));
		goto Finish;
	}

	err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));

		if (err == CL_BUILD_PROGRAM_FAILURE)
		{
			size_t log_size = 0;
			clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

			std::vector<char> build_log(log_size);
			clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], NULL);

			LogError("Error happened during the build of OpenCL program.\nBuild log:%s", &build_log[0]);
		}
	}

Finish:
	return err;
}

#define MAXPASS 10

int _tmain(int argc, TCHAR* argv[])
{

	int w = 320, h = 320;
	cl_int *bufPix = (cl_int *)calloc(w * h, sizeof(cl_int));
	cl_int *bufLabel = (cl_int *)calloc(w * h, sizeof(cl_int));
	cl_int *bufFlags = (cl_int *)calloc(MAXPASS + 1, sizeof(cl_int));

	{
		int x, y;
		for (y = 0; y<h; y++) {
			for (x = 0; x<w; x++) {
				bufPix[y * w + x] = abs(x-160)+abs(y-160)<100 ? 1: 0;
			}
		}
	}

	
	{
		const int hh = 320, ww = 320;
		for (int y = 0; y<h; y++) {
			for (int x = 0; x<w; x++) {
				if (bufPix[y * w + x] >0)
				{
					int color = bufPix[y*w + x];
					int Q[99999];
					int minP[hh], maxP[hh], minQ[hh], maxQ[hh];
					memset(minP, -1, sizeof(minP));
					memset(maxP, -1, sizeof(maxP));
					memset(minQ, -1, sizeof(minQ));
					memset(maxQ, -1, sizeof(maxQ));
					Q[0] = y*w + x;
					int Ql = 1;

					for (int i = 0; i < Ql; ++i)
					{
						int tx = Q[i] % w;
						int ty = Q[i] / w;

						for (int yy = -1; yy <= 1; ++yy)
							for (int xx = -1; xx <= 1; ++xx)
							{
								int nY = ty + yy;
								int nX = tx + xx;
								if (nY < 0 || nY >= h || nX < 0 || nX >= w) continue;
								int id2 = nY*w + nX;
								if (color == bufPix[id2])
								{
									bufPix[id2] = 0;
									Q[Ql] = id2;
									Ql += 1;
									if (minP[nY] == -1 || minP[nY] % w > nX)
										minP[nY] = id2;
									if (maxP[nY] == -1 || maxP[nY] % w < nX)
										maxP[nY] = id2;
									if (minQ[nX] == -1 || minQ[nX] / w > nY)
										minQ[nX] = id2;
									if (maxQ[nX] == -1 || maxQ[nX] / w < nY)
										maxQ[nX] = id2;
								}
								bufPix[id2] = -1; // label : already matched.
							}
					}

					int startMinP = -1, stopMinP = -1; //[i=start, i<=stop]
					for (int i = 0; i<h; ++i)
						if (minP[i] != -1) { startMinP = i; break; };
					for (int i = h-1; i>=0; --i)
						if (minP[i] != -1) { stopMinP = i; break; };

					int startMaxP = -1, stopMaxP = -1;
					for (int i = 0; i<h; ++i)
						if (maxP[i] != -1) { startMaxP = i; break; };
					for (int i = h - 1; i >= 0; --i)
						if (maxP[i] != -1) { stopMaxP = i; break; };

					int startMinQ = -1, stopMinQ = -1;
					for (int i = 0; i<w; ++i)
						if (minQ[i] != -1) { startMinQ = i; break; };
					for (int i = h - 1; i >= 0; --i)
						if (minQ[i] != -1) { stopMinQ = i; break; };

					int startMaxQ = -1, stopMaxQ = -1;
					for (int i = 0; i<w; ++i)
						if (maxP[i] != -1) { startMaxQ = i; break; };
					for (int i = h - 1; i >= 0; --i)
						if (maxP[i] != -1) { stopMaxQ = i; break; };

					int l = 0;
					int lid[4 * (hh + ww)];
					int t1 = -1;

					for (int i = startMinQ; i <= stopMaxQ; ++i)
						if (minQ[i] == minP[startMinP]) { t1 = i-1; lid[l++] = minQ[startMinP]; break; };
					for (int i=startMinP+1; i<=stopMinP; ++i)
					{
						if (minP[i] == maxQ[startMaxQ])
						{
							t1 = i+1; lid[l++] = maxQ[startMaxQ]; break;
						};
						if (t1>=startMinQ)
						{
							while (minP[i]!=minQ[t1])
								lid[l++] = minQ[t1--];
						}
						lid[l++] = minP[i];
						t1--;
					}
					for (int i = startMaxQ+1; i<=stopMaxQ; ++i)
					{
						if (maxQ[i] == maxP[stopMaxP])
						{
							t1 = i+1; lid[l++] = maxP[stopMaxP]; break;
						};
						if (t1 <= stopMinP)
						{
							while (maxQ[i] != minP[t1])
								lid[l++] = minP[t1++];
						}
						lid[l++] = maxQ[i];
						t1++;
					}
					for (int i = stopMaxP-1; i >= startMaxP; --i)
					{
						if (maxP[i] == minQ[stopMinQ])
						{
							t1 = i-1; lid[l++] = minQ[stopMinQ]; break;
						};
						if (t1 <= stopMaxQ)
						{
							while (maxP[i] != maxQ[t1])
								lid[l++] = maxQ[t1++];
						}
						lid[l++] = maxP[i];
						t1++;
					}
					for (int i = stopMinQ-1; i >= startMinQ; --i)
					{
						if (minQ[i] == minP[startMinP]) 
							break; //fin.
						if (t1 >= stopMaxP)
						{
							while (minQ[i] != maxP[t1])
								lid[l++] = maxP[t1--];
						}
						lid[l++] = minQ[i];
						t1--;
					}

					// pigeonhole: 5 segments, 1 segment must be totally on line.
					for (int i=0; i<5; ++i)
					{
						float x2 = 0, y2 = 0, x = 0, y = 0, xy = 0, ln = l / 5;
						for (int j = i*l / 5; j < (i + 1)*l / 5; ++j)
						{
							int xx = lid[j] % w;
							int yy = lid[j] / w;
							x += xx; y += yy;
							x2 += xx*xx;
							y2 += yy*yy;
							xy += xx*yy;
						}
						x /= ln; y /= ln; x2 /= ln; y2 /= ln; xy /= ln;
						float a = x2 - x*x, b = xy - x*y, c = y2 - y*y;
						float l1 = a + c + sqrt((a - c)*(a - c) + 4 * b*b), l2 = a + c - sqrt((a - c)*(a - c) + 4 * b*b);
						if (sqrt(abs(l2/l1))<0.1)
						{
							// line found.
							float dx = l1/2 - c, dy = b;
							float A = dy, B = -dx, C = dx*y - dy*x;

						}
					}


				}
			}
		}
	}

	/*
	 
	cl_int err;
	opencl_common ocl;
	cl_device_type deviceType = CL_DEVICE_TYPE_GPU;

	//initialize Open CL objects (context, queue, etc.)
	if (CL_SUCCESS != SetupOpenCL(&ocl, deviceType))
	{
		return -1;
	}

	// Create and build the OpenCL program
	if (CL_SUCCESS != CreateAndBuildProgram(&ocl, cl_scan_algo))
	{
		return -1;
	}

	cl_kernel kernel_prepare = clCreateKernel(ocl.program, "labelxPreprocess_int_int", NULL);
	cl_kernel kernel_propagate = clCreateKernel(ocl.program, "label8xMain_int_int", NULL);

	cl_mem memPix = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, w * h * sizeof(cl_int), bufPix, NULL);
	cl_mem memLabel = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, w * h * sizeof(cl_int), bufLabel, NULL);
	cl_mem memFlags = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (MAXPASS + 1) * sizeof(cl_int), bufFlags, NULL);


	size_t work_size[2] = { (size_t)((w + 31) & ~31), (size_t)((h + 31) & ~31) };

	int i;

	cl_event events[MAXPASS + 1];
	for (i = 0; i <= MAXPASS; i++) {
		events[i] = clCreateUserEvent(ocl.context, NULL);
	}


	clSetKernelArg(kernel_prepare, 0, sizeof(cl_mem), (void *)&memLabel);
	clSetKernelArg(kernel_prepare, 1, sizeof(cl_mem), (void *)&memPix);
	clSetKernelArg(kernel_prepare, 2, sizeof(cl_mem), (void *)&memFlags);
	i = MAXPASS; clSetKernelArg(kernel_prepare, 3, sizeof(cl_int), (void *)&i);
	i = 0; clSetKernelArg(kernel_prepare, 4, sizeof(cl_int), (void *)&i);
	clSetKernelArg(kernel_prepare, 5, sizeof(cl_int), (int *)&w);
	clSetKernelArg(kernel_prepare, 6, sizeof(cl_int), (int *)&h);

	clEnqueueNDRangeKernel(ocl.commandQueue, kernel_prepare, 2, NULL, work_size, NULL, 0, NULL, &events[0]);

	for (i = 1; i <= MAXPASS; i++) {
		clSetKernelArg(kernel_propagate, 0, sizeof(cl_mem), (void *)&memLabel);
		clSetKernelArg(kernel_propagate, 1, sizeof(cl_mem), (void *)&memPix);
		clSetKernelArg(kernel_propagate, 2, sizeof(cl_mem), (void *)&memFlags);
		clSetKernelArg(kernel_propagate, 3, sizeof(cl_int), (void *)&i);
		clSetKernelArg(kernel_propagate, 4, sizeof(cl_int), (int *)&w);
		clSetKernelArg(kernel_propagate, 5, sizeof(cl_int), (int *)&h);

		clEnqueueNDRangeKernel(ocl.commandQueue, kernel_propagate, 2, NULL, work_size, NULL, 0, NULL, &events[i]);
	}

	clEnqueueReadBuffer(ocl.commandQueue, memLabel, CL_TRUE, 0, w * h * sizeof(cl_int), bufLabel, 0, NULL, NULL);
	clEnqueueReadBuffer(ocl.commandQueue, memFlags, CL_TRUE, 0, (MAXPASS + 1) * sizeof(cl_int), bufFlags, 0, NULL, NULL);

	clFinish(ocl.commandQueue);

	*/

	FILE *f;
	unsigned char *img = NULL;
	int filesize = 54 + 3 * w*h;  //w is your image width, h is image height, both int

	img = (unsigned char *)malloc(3 * w*h);
	memset(img, 0, 3 * w*h);

	for (int i = 0; i<w; i++)
	{
		for (int j = 0; j<h; j++)
		{
			int rgb = bufLabel[i * w + j] == -1 ? 0 : (bufLabel[i * w + j] * 1103515245 + 12345);
			img[(i + j*w) * 3 + 2] = rgb & 0xff; rgb >>= 8;
			img[(i + j*w) * 3 + 1] = rgb & 0xff; rgb >>= 8;
			img[(i + j*w) * 3 + 0] = rgb & 0xff; rgb >>= 8;
		}
	}

	unsigned char bmpfileheader[14] = { 'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0 };
	unsigned char bmpinfoheader[40] = { 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0 };
	unsigned char bmppad[3] = { 0,0,0 };

	bmpfileheader[2] = (unsigned char)(filesize);
	bmpfileheader[3] = (unsigned char)(filesize >> 8);
	bmpfileheader[4] = (unsigned char)(filesize >> 16);
	bmpfileheader[5] = (unsigned char)(filesize >> 24);

	bmpinfoheader[4] = (unsigned char)(w);
	bmpinfoheader[5] = (unsigned char)(w >> 8);
	bmpinfoheader[6] = (unsigned char)(w >> 16);
	bmpinfoheader[7] = (unsigned char)(w >> 24);
	bmpinfoheader[8] = (unsigned char)(h);
	bmpinfoheader[9] = (unsigned char)(h >> 8);
	bmpinfoheader[10] = (unsigned char)(h >> 16);
	bmpinfoheader[11] = (unsigned char)(h >> 24);

	f = fopen("img.bmp", "wb");
	fwrite(bmpfileheader, 1, 14, f);
	fwrite(bmpinfoheader, 1, 40, f);
	for (int i = 0; i<h; i++)
	{
		fwrite(img + (w*(h - i - 1) * 3), 3, w, f);
		fwrite(bmppad, 1, (4 - (w * 3) % 4) % 4, f);
	}

	free(img);
	fclose(f);

	return 0;
}

