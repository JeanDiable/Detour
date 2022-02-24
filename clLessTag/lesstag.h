#pragma once
#include "CL\cl.h"

struct detected_code_im {
	unsigned int id;
	float x, y, x1, y1, x2, y2, x3, y3, x4, y4, roll2d;
};

struct detected_code_phy {
	float phy_x, phy_y, phy_z, roll, pitch, yaw;
};


class lesstag { 
public:
	cl_device_id device;
	cl_context context;
	cl_kernel kernel_gradient;
	cl_kernel kernel_prepare_uf;
	cl_kernel kernel_perform_uf;
	cl_kernel kernel_relabel_points;
	cl_kernel kernel_first_count;
	cl_kernel kernel_reduce_count;
	cl_kernel kernel_merge_pos;
	cl_kernel kernel_propagate;
	cl_kernel kernel_calc_line_1;
	cl_kernel kernel_calc_line_2;
	cl_kernel kernel_reduce_line;
	cl_kernel kernel_line_merge_connect;
	cl_kernel kernel_line_merge_unionfind;
	cl_kernel kernel_line_merge_reverse_list;
	cl_kernel kernel_line_merge_calc;
	cl_kernel kernel_line_merge_reduce;
	cl_kernel kernel_quadrilateral_prepare;
	cl_kernel kernel_quadrilateral_find;
	cl_kernel kernel_homography_transform;
	cl_kernel kernel_rect_decoding;
	cl_kernel kernel_error_correcting;
	cl_mem memIm;
	cl_mem memFlags;
	cl_mem memGrad;
	cl_mem memLabel;
	cl_mem memCount1;
	cl_mem memLen; 
	cl_mem memStat;
	cl_mem memLines;
	cl_mem memConns;
	cl_mem memPos;
	cl_mem memRConn1;
	cl_mem memRConn2;
	cl_mem memPoints;
	cl_mem memH;
	cl_mem memRect;
	cl_mem memCoding;
	cl_mem memResults;
	cl_mem memResultsE;
	cl_mem constCodes;
	int w;
	int h;
	cl_command_queue queue;
	float* bufH;
	int* bufResults;
	cl_mem memNorm;
	cl_kernel kernel_cropping;
	cl_mem memLines2;
	int inW, inH;
	cl_mem memRaw;
	cl_mem memMesh;
	cl_mem ordering;
	cl_mem debug;
	cl_mem cm;
};

lesstag* init_cllesstag(char* dev, char* type, int w, int h);

int perform_lesstag(lesstag* ret, unsigned char* bufIm, detected_code_im* result, int d_lvl = 0);

void apply_mesh(lesstag* ret, float* meshX, float* meshY);

void perform_crop(lesstag* ret, unsigned char* bufIm, unsigned char* bufOut);