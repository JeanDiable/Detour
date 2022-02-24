#include <cuda_runtime_api.h>
#include <algorithm>
#include <helper_cuda.h>
#include <ctime>
#include <iostream>
#include "reg_class.h"

unsigned char* readBMP(char* filename)
{
	int i;
	FILE* f = fopen(filename, "rb");
	unsigned char info[31798];
	fread(info, sizeof(unsigned char), 31798, f); // read the 54-byte header
	int width = *(int*)&info[18];
	int height = *(int*)&info[22];
	//printf("w:%d, h:%d\n", width, height);
	int size = width * height;
	unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
	fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
	fclose(f);
	return data;
}

template<typename type> void dumpVars(const char*filename, type* what, int sz)
{
	auto ofp = fopen(filename, "w");
	if (ofp == NULL) {
		perror("opening output file");
		exit(-1);
	}
	for (int i = 0; i < sz; ++i)
		fprintf(ofp, "%f ", (float)what[i]);

	fclose(ofp);
}


void test_reg()
{
	reg_core* core = reg_core::getInstance();

	FILE *fp;
	if ((fp = fopen("caliberation.values", "r+")) == NULL) {
		printf("No such file\n");
		return;
	}
	float2 mesh[64];
	int k = 0;
	for (int i = 0; i < 8; ++i)
		for (int j = 0; j < 8; ++j) {
			float x, y;
			fscanf(fp, "%f,%f\n", &x, &y);
			mesh[k].x = x * 1280;
			mesh[k].y = y * 1024;
			k += 1;
		}
	fclose(fp);

	core->init(1280, 1024, mesh);

	int err = 0;
	long tn = 0;
	for (int i = 1; i<1200; ++i)
	{
		using namespace std::chrono;
		char tmp[256];
		sprintf(tmp, "D:\\work\\robot\\frames\\2\\(%d).bmp", i);
		auto bmpT = readBMP(tmp);
		sprintf(tmp, "D:\\work\\robot\\frames\\2\\(%d).bmp", i + 1);
		auto bmpC = readBMP(tmp);
		core->cropImage(bmpT,0);
		core->preprocess(0,0);
	
	
		auto start = high_resolution_clock::now();
		core->cropImage(bmpC,1);
		core->preprocess(1,1);
		core->set(0);
		auto result = core->reg(1);
		tn += duration_cast<microseconds>(high_resolution_clock::now() - start).count();
	
	
		if (result.conf < 13) {
			err++;
			std::cout << "*******bad*********" << std::endl;
		}
		delete bmpT;
		delete bmpC;
	
		//system("pause");
	}

	printf("error: %d\n", err);
	std::cout << "time:" << tn / 1200 << "us" << std::endl;
}

void main() {
	std::thread t1(test_reg);
	//std::thread t2(test_reg);
	t1.join();
	//t2.join();
	system("pause");
}