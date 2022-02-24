#include <iostream>
#include <ctime>
#include <future>
#include <random>

#pragma warning(disable:4996)

// 2^27 + 9bit padding.
#define scope 134217728
#define correction 2
#define padding 1
#define dist (correction*2+padding)
#define filename "codes_c2p0.txt"

const int map[] = { 6,19,21,2,24,7,16,22,8,12,26,11,31,35,27,15,23,1,33,32,29,9,25,0,18,13,3 };
const int ordering[] = { 23,17,3,26,-1,-1,0,5,8,21,-1,11,9,25,-1,15,6,-1,24,1,-1,2,7,16,4,22,10,14,-1,20,-1,12,19,18,-1,13 };
int rot(int val)
{
	char bh[36];
	for (int i = 0; i < 36; ++i)
		bh[i] = 1;
	for (int i = 0; i < 27; ++i)
		bh[map[i]] = (val&(1 << i)) != 0;

	int ret = 0;
	for (int i = 0; i<36; ++i)
	{
		if (ordering[i] >= 0)
		{
			int xx = ordering[i] % 6;
			int yy = ordering[i] / 6;
			ret |= bh[6 * xx + 5 - yy] << i;
		}
	}
	return ret;
}

void gen_code()
{
	std::mt19937 mt_rand(time(0));
	auto fd = fopen(filename, "w");
	fprintf(fd, "int codes[]={");

	int outxor = mt_rand() % (scope - 1);
	printf("outxor:%x\n", outxor);

	char* code = new char[scope]; //12 bit
	for (int i = 0; i < scope; ++i)
		code[i] = 0;
	int codes = 0;
	//int rem = scope;
	while (true) {
		bool found = false;
		for (int i = 0; i < scope; ++i)
		{
			int nCode = i^outxor;

			if (code[nCode] == 0) {
				//rem -= 1;
				found = true;
				codes += 1;
				printf("%d> %x\n", codes, nCode);
				fprintf(fd, "0x%x,", nCode);
				if (codes % 10 == 9) fprintf(fd, "\n");
				int c2 = rot(nCode);
				int c3 = rot(nCode);
				int c4 = rot(nCode);
				for (int j = 0; j < scope; ++j)
					if (__popcnt(j^nCode) <= dist ||
						__popcnt(j^c2) <= dist ||
						__popcnt(j^c3) <= dist ||
						__popcnt(j^c4) <= dist) {
						code[j] = 1;
					}
			}
			if (found) break;
		}
		if (!found) break;
		if (codes % 4096 == 4095)
			fprintf(fd, "\n //=====\n");
	}
	fprintf(fd, "-1}");
	fprintf(fd, "\n#define n_codes %d", codes);
	printf("total %d codes\n", codes);
	fclose(fd);
}

void genIm()
{
	int n = 0;
	int id;
	auto fd = fopen(filename, "r");
	while (fscanf(fd, "%x", &id) != EOF)
	{
		char bh[36];
		for (int i = 0; i < 36; ++i)
			bh[i] = 1;
		for (int i = 0; i < 27; ++i)
			bh[map[i]] = (id&(1 << i)) != 0;
		char buf[50];
		sprintf(buf, "code_ims/im%d.html", n);
		auto tfd = fopen(buf, "w");
		fprintf(tfd, R"head(
<svg width = "400" height = "400">
<rect width = "320" height = "320" x="40" y="40" style="fill:black;" />
<rect x="80" y="80" width = "240" height = "240" style="fill:white;" />
<text x="200" y="30" style="text-anchor: middle">%d</text>
         )head", n);
		for (int i = 0; i < 36; ++i) {
			int x = i % 6;
			int y = 5 - i / 6;
			if (bh[i])
				fprintf(tfd, R"rect(<rect x="%d" y="%d" width = "40" height = "40" style = "fill:black;" />)rect", (x + 2) * 40, (y + 2) * 40);
		}
		fprintf(tfd, R"(</svg>)");
		fclose(tfd);

		printf("%d>%s\n", n, buf);
		n += 1;
	}
}

int main3(int argc, char** argv)
{
	//gen_code();
	genIm();
	system("pause");
	return 0;
}