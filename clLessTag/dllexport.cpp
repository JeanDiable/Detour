#include "lesstag.h"

struct detector_interface {
	lesstag *td;
	int width; 
	int height;
    
    int n;
	detected_code_im* result;
	detected_code_phy* result2;
};

extern "C" __declspec(dllexport) detector_interface* init_lesstag(char* dev, char* type, unsigned int width, unsigned int height) {
	detector_interface* ret = new detector_interface();
	ret->td = init_cllesstag(dev, type, width, height);
	ret->width = width;
	ret->height = height;
	ret->result = new detected_code_im[50];
	ret->result2 = new detected_code_phy[50];
	return ret;
}

extern "C" __declspec(dllexport) detected_code_im * detect_lesstag(detector_interface * detector, unsigned char* im, int* len, int d_lvl) {
	detector->n = *len = perform_lesstag(detector->td, im, detector->result, d_lvl);
	return detector->result;
}