#pragma once
#include <windows.h>
#include <map>

HMODULE DllHandle = NULL; 

std::map<std::string, std::string> code;

BOOL APIENTRY DllMain(HMODULE hModule, DWORD dwReason, LPVOID lpReserved) {
	if (dwReason == DLL_PROCESS_ATTACH) DllHandle = hModule;
	code.insert(std::make_pair("cl2.0", "OPENCL_SOURCE_A"));
	code.insert(std::make_pair("cl1.2", "OPENCL_SOURCE_B"));
	return TRUE;
}
 
void get_source(char*& data, char* src)
{
	HRSRC resource = FindResource(DllHandle, "CCL_SOURCE", code[std::string(src)].c_str());
	HGLOBAL resMem = LoadResource(DllHandle, resource);
	data = (char*)LockResource(resMem);
}