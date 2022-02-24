#pragma once

typedef unsigned char uchar;
typedef float2 Complex;

#define rho_num  160
#define theta_num   192
// 320*pi/2

#define algo_sz  320
//
#define algo_sz_2 328
#define rotfun_num  7
#define rot_check_num 4
// 6:1.53, 4:1.25 8:1.85  ->0:0.64
//  above is number of thetas, not including +180.

#define croping_mesh 8

#define init_scope 0.7f / theta_num*180.0f
#define better_scope 0.3f / theta_num*180.0f


cudaStream_t get_stream();