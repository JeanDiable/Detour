__kernel void gradient(global const unsigned char* pix, global char2* grad, int iw, int ih) {
	const int x = get_global_id(0), y = get_global_id(1);
	const int p0 = y * iw + x;
	if (x >= iw - 5 || y >= ih - 5 || x < 5 || y < 5) {
		grad[p0].x = 0;
		grad[p0].y = 0;
		return;
	}

	unsigned char
		p00 = pix[p0 - iw * 2 - 2], p01 = pix[p0 - iw * 2 - 1], p02 = pix[p0 - iw * 2 - 0], p03 = pix[p0 - iw * 2 + 1],
		p10 = pix[p0 - iw * 1 - 2], p11 = pix[p0 - iw * 1 - 1], p12 = pix[p0 - iw * 1 - 0], p13 = pix[p0 - iw * 1 + 1],
		p20 = pix[p0 - iw * 0 - 2], p21 = pix[p0 - iw * 0 - 1], p22 = pix[p0 - iw * 0 - 0], p23 = pix[p0 - iw * 0 + 1],
		p30 = pix[p0 + iw * 1 - 2], p31 = pix[p0 + iw * 1 - 1], p32 = pix[p0 + iw * 1 - 0], p33 = pix[p0 + iw * 1 + 1];

	float valx =
		p00 * 0.1605 + p01 * 0.2284 + p02 * -0.2284 + p03 * -0.1605
		+ p10 * 0.4595 + p11 * 0.7666 + p12 * -0.7666 + p13 * -0.4595
		+ p20 * 0.4595 + p21 * 0.7666 + p22 * -0.7666 + p23 * -0.4595
		+ p30 * 0.1605 + p31 * 0.2284 + p32 * -0.2284 + p33 * -0.1605;

	float valy =
		p00 * 0.16059 + p01 * 0.4595 + p02 * 0.4595 + p03 * 0.1605
		+ p10 * 0.22842 + p11 * 0.7666 + p12 * 0.7666 + p13 * 0.2284
		+ p20 * -0.2284 + p21 * -0.766 + p22 * -0.766 + p23 * -0.228
		+ p30 * -0.1605 + p31 * -0.459 + p32 * -0.459 + p33 * -0.160;

	float fx = 125 / fabs(valx), fy = 125 / fabs(valy);
	float mf = fmin(fx, fy);
	float fac = fmin(1.0f, mf);

	grad[p0].x = valx * fac;
	grad[p0].y = valy * fac;
}


#define theta_thresh 0.983
#define g_thresh 24
__kernel void prepare_uf(global int* label, global const char2* grad, global unsigned char* flags, int iw, int ih) {
	const int x = get_global_id(0), y = get_global_id(1);
	const int p0 = y * iw + x;
	if (x >= iw || y >= ih) return;

	unsigned char flag = 0;
	unsigned int val = p0;

	char2 gd = grad[p0];

	float norm = sqrt((float)(gd.x * gd.x + gd.y * gd.y));
	if (norm < g_thresh) {
		val = -1;
		flag = 3; // bad pix.
	}
	else {
		char2 cmp;
		float cs;

		if (x > 0) {
			cmp = grad[p0 - 1];
			float norm2 = sqrt((float)(cmp.x * cmp.x + cmp.y * cmp.y));
			cs = (cmp.x * gd.x + cmp.y * gd.y) / (norm2 * norm);
			if (norm2 > g_thresh && cs > theta_thresh && norm2 * 1.3 > norm && norm * 1.3 > norm2) {
				val = p0 - 1;
				flag += 1;
			}
		}

		if (y > 0) {
			cmp = grad[p0 - iw];
			float norm2 = sqrt((float)(cmp.x * cmp.x + cmp.y * cmp.y));
			cs = (cmp.x * gd.x + cmp.y * gd.y) / (norm2 * norm);
			if (norm2 > g_thresh && cs > theta_thresh && norm2 * 1.3 > norm && norm * 1.3 > norm2) {
				val = p0 - iw;
				flag += 1;
			}
		}
	}
	label[p0] = val;
	flags[p0] = flag;
}

__kernel void perform_uf(global int* label, const global unsigned char* flags, int iw, int ih) {
	const int x = get_global_id(0), y = get_global_id(1);

	if (x >= iw || y >= ih || x < 1 || y < 1) return;
	const int p0 = y * iw + x;

	if (flags[p0] == 3) return;

	//propagate: //needs filtering.
	int tmp, ori = label[p0];
	if ((tmp = label[ori]) != ori) {
		ori = tmp;
		if ((tmp = label[ori]) != ori) ori = tmp;
	} //label[p0]=label[label[label[p0]]]

	if (flags[p0] == 2) {
		int up_v = label[p0 - iw], left_v = label[p0 - 1];
		if (up_v < left_v && label[left_v]>up_v) atomic_min(&label[left_v], up_v);
		else if (left_v < up_v && label[up_v]>left_v) atomic_min(&label[up_v], left_v);
	}

	if ((tmp = label[ori]) != ori) {
		ori = tmp;
		if ((tmp = label[ori]) != ori) ori = tmp;
	}
	label[p0] = ori;
}


__kernel void first_count(global int* label, global int* count1, int N) {
	const int x = get_global_id(0);
	if (x >= N) return;
	int lab = label[x];
	if (lab < 0) return;
	if (count1[lab] < 10)
		atomic_inc(&count1[lab]);
}

__kernel void reduce_count(global int* label, global int* count1, int N) {
	const int id = get_global_id(0);
	if (id >= N) return;
	int lbl = label[id];
	if (lbl < 0) return;
	if (count1[lbl] < 10)
		label[id] = -1;
}

#define reduce_sz 32 
#define group_sz 2048 
#define group_comps 256u
#define max_lines 1024
#define max_quads 256


// filter out connected component labels.
// todo: rename to "extract component id"
__attribute__((reqd_work_group_size(reduce_sz, 1, 1)))
__kernel void relabel_points(global int* label, global unsigned int* len, global int* pos, int N) {
	// len: contains component number for each 2048 pix, pos: contains component id.
	const int x = get_local_id(0), gid = get_group_id(0);
	int group_st = gid * group_sz;
	int out_st = group_comps * gid;

	local unsigned int sz_boundaries;

	if (get_local_id(0) == 0)
		sz_boundaries = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	// filter out valid points:
	for (int i = 0; i < group_sz / reduce_sz; ++i) {
		int x0 = group_st + x + (i * reduce_sz);
		bool is_point = (x0 < N) && (label[x0] == x0);

		int p_sum = sub_group_scan_inclusive_add(is_point); // simd 32
		if (is_point) {
			int p = sz_boundaries + p_sum - 1;
			if (p < group_comps)
				pos[out_st + p] = x0;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (get_local_id(0) == reduce_sz - 1)
			sz_boundaries += p_sum;
	}

	if (x == 0) {
		len[gid] = min(sz_boundaries, group_comps);
	}
}

__kernel void merge_pos(global int* label, global int* len, global int* pos, int group_n) {
	int x = get_global_id(0);
	local int pstart;

	if (get_local_id(0) == 0)
		pstart = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < group_n / reduce_sz + (group_n % reduce_sz > 0); ++i) {
		int x0 = x + (i * reduce_sz);
		int mylen = (x0 < group_n) * len[x0];
		int p_sum = sub_group_scan_inclusive_add(mylen);

		for (int j = 0; j < mylen; ++j) {
			int nid = pstart + p_sum - mylen + j;
			if (nid < max_lines)
				label[pos[x0 * group_comps + j]] = -nid - 4;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		if (get_local_id(0) == reduce_sz - 1)
			pstart += p_sum;
	}

	if (get_local_id(0) == reduce_sz - 1) {
		len[0] = min(pstart, max_lines);
	}
}
// 

__kernel void propagate(global int* label, int N) {
	const int gid = get_global_id(0);
	if (gid >= N) return;
	if (label[gid] == -1) return;

	int lbl = label[gid];
	int count = 0;
	while (lbl > -1 && lbl < gid && count < 5) { // leaf
		lbl = label[lbl];
		label[gid] = lbl;
		count += 1;
	}
	if (lbl > -1 && lbl < gid)
		label[gid] = -1;
}

// todo: optimize this: 
#define gd_thresh 16

//todo: optimize this.
__kernel void calc_line_1(global int* label, global const char2* grad, global unsigned int* stat, int iw) {
	const unsigned int x = get_global_id(0), y = get_global_id(1);
	const unsigned int p0 = iw * y + x;
	int lbl = label[p0];
	if (lbl == -1) return;
	lbl = -lbl - 4;
	//if (num[lbl] < 20) return; 
	int norm = ceil(sqrt((float)(grad[p0].x * grad[p0].x + grad[p0].y * grad[p0].y)) / gd_thresh); //40
	//todo: use some numerical stable methods for variance computing.

	//lbl += (p0 % 2)*fract;
	unsigned int rnd = ((p0 + grad[p0].x + grad[p0].y) * 15485863 + 32452867) * 86028157;
	int plc = lbl + (rnd % 5) * max_lines; // 0,1 ok, 2-7 drop.
	if (plc >= 2 * max_lines || plc < 0)
		plc = lbl;
	atomic_add(&stat[plc * 12], (unsigned int)(x * norm));
	atomic_add(&stat[plc * 12 + 1], (unsigned int)(y * norm));
	atomic_add(&stat[plc * 12 + 2], (unsigned int)(x * x * norm));
	atomic_add(&stat[plc * 12 + 3], (unsigned int)(y * y * norm));
	atomic_add(&stat[plc * 12 + 4], (unsigned int)(x * y * norm));
	atomic_add(&stat[plc * 12 + 5], (unsigned int)((256 + grad[p0].x) * norm));
	atomic_add(&stat[plc * 12 + 6], (unsigned int)((256 + grad[p0].y) * norm));
	atomic_add(&stat[plc * 12 + 7], (unsigned int)(norm));
	atomic_max(&stat[plc * 12 + 8], 10000u - x);
	atomic_max(&stat[plc * 12 + 9], 10000u - y);
	atomic_max(&stat[plc * 12 + 10], x);
	atomic_max(&stat[plc * 12 + 11], y);
}

__kernel void calc_line_2(global unsigned int* stat, global int* len, global float* lines) {
	//num: per component pix num, len[0]: n of components
	const int p0 = get_global_id(0);
	if (p0 >= len[0] || p0 >= max_lines) return;

	unsigned int x = stat[p0 * 12] + stat[(max_lines + p0) * 12];
	unsigned int y = stat[p0 * 12 + 1] + stat[(max_lines + p0) * 12 + 1];
	unsigned int xx = stat[p0 * 12 + 2] + stat[(max_lines + p0) * 12 + 2];
	unsigned int yy = stat[p0 * 12 + 3] + stat[(max_lines + p0) * 12 + 3];
	unsigned int xy = stat[p0 * 12 + 4] + stat[(max_lines + p0) * 12 + 4];
	unsigned int n = stat[p0 * 12 + 7] + stat[(max_lines + p0) * 12 + 7];

	float fx = ((float)(x % n)) / n + x / n;
	float fy = ((float)(y % n)) / n + y / n;
	float fxx = ((float)(xx % n)) / n + xx / n;
	float fyy = ((float)(yy % n)) / n + yy / n;
	float fxy = ((float)(xy % n)) / n + xy / n;

	float a = fxx - fx * fx, b = fxy - fx * fy, c = fyy - fy * fy;
	float sqt = sqrt((a - c) * (a - c) + 4 * b * b);
	float l1 = a + c + sqt, l2 = a + c - sqt;
	float dx, dy;
	if (fabs(a - l1 / 2) > fabs(c - l1 / 2)) {
		dy = l1 / 2 - a; dx = b;
	}
	else
	{
		dx = l1 / 2 - c; dy = b;
	}
	float norm = sqrt(dx * dx + dy * dy);
	dx /= norm; dy /= norm;

	float gx = stat[p0 * 12 + 5] + stat[(p0 + max_lines) * 12 + 5] - 256.0f * n, gy = stat[p0 * 12 + 6] + stat[(p0 + max_lines) * 12 + 6] - 256.0f * n;
	norm = sqrt(gx * gx + gy * gy);
	gx /= norm; gy /= norm;

	float A = dy, B = -dx, C = dx * fy - dy * fx;
	lines[p0 * 12] = A;
	lines[p0 * 12 + 1] = B;
	lines[p0 * 12 + 2] = C;
	lines[p0 * 12 + 5] = gx;
	lines[p0 * 12 + 6] = gy;
	float minx = 10000 - max(stat[p0 * 12 + 8], stat[(p0 + max_lines) * 12 + 8]),
		miny = 10000 - max(stat[p0 * 12 + 9], stat[(p0 + max_lines) * 12 + 9]),
		maxx = max(stat[p0 * 12 + 10], stat[(p0 + max_lines) * 12 + 10]),
		maxy = max(stat[p0 * 12 + 11], stat[(p0 + max_lines) * 12 + 11]);
	if (fabs(minx * A + miny * B + C) > fabs(minx * A + maxy * B + C)) {
		float t = miny;
		miny = maxy;
		maxy = t;
	}

	// pedal point.
	lines[p0 * 12 + 8] = (B * B * minx - A * B * miny - A * C) / (A * A + B * B);
	lines[p0 * 12 + 9] = (A * A * miny - A * B * minx - B * C) / (A * A + B * B);
	minx = lines[p0 * 12 + 8]; miny = lines[p0 * 12 + 9];

	lines[p0 * 12 + 10] = (B * B * maxx - A * B * maxy - A * C) / (A * A + B * B);
	lines[p0 * 12 + 11] = (A * A * maxy - A * B * maxx - B * C) / (A * A + B * B);
	maxx = lines[p0 * 12 + 10]; maxy = lines[p0 * 12 + 11];

	if (l2 / l1 <= 0.333) { // should be 0
		lines[p0 * 12 + 7] = (maxx - minx) * (maxx - minx) + (maxy - miny) * (maxy - miny); //length, should be >20px
	}
	else
		lines[p0 * 12 + 7] = -1; // not a line

	lines[p0 * 12 + 3] = (minx + maxx) / 2;
	lines[p0 * 12 + 4] = (miny + maxy) / 2;
}


// filter out lines with length <7px.
__kernel void reduce_line(global int* len, global float* lines, global float* rlines) {
	const int me = get_local_id(0);

	local int sz_lines;
	if (me == 0)
		sz_lines = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	// filter out valid points:
	for (int i = 0; i < len[0] / reduce_sz + (len[0] % reduce_sz > 0); ++i) {
		int x0 = me + (i * reduce_sz);
		bool is_point = x0 < len[0] && lines[x0 * 12 + 7] >= 20;
		int p_sum = sub_group_scan_inclusive_add(is_point); // simd 32
		int pos = sz_lines + p_sum - 1;
		if (is_point && pos < max_lines && pos >= 0) {
			rlines[pos * 12] = lines[x0 * 12];
			rlines[pos * 12 + 1] = lines[x0 * 12 + 1];
			rlines[pos * 12 + 2] = lines[x0 * 12 + 2];
			rlines[pos * 12 + 3] = lines[x0 * 12 + 3];
			rlines[pos * 12 + 4] = lines[x0 * 12 + 4];
			rlines[pos * 12 + 5] = lines[x0 * 12 + 5];
			rlines[pos * 12 + 6] = lines[x0 * 12 + 6];
			rlines[pos * 12 + 7] = lines[x0 * 12 + 7];
			rlines[pos * 12 + 8] = lines[x0 * 12 + 8];
			rlines[pos * 12 + 9] = lines[x0 * 12 + 9];
			rlines[pos * 12 + 10] = lines[x0 * 12 + 10];
			rlines[pos * 12 + 11] = lines[x0 * 12 + 11];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (me == reduce_sz - 1)
			sz_lines += p_sum;
	}

	if (me == reduce_sz - 1) {
		len[0] = min(sz_lines, max_lines);
	}
}

// todo: use bit operations.
#define max_conn 64
#define prepare_threads 8192
// 1/4 l-> join.
#define thresh_gap 1/4.0f
// pixel range.
#define thresh_whole 20
__kernel void line_merge_connect(global float* lines, global int* len, global int* rconn1) {
	int p0 = get_global_id(0);
	int nlines = len[0];
	int np = (nlines) * (nlines - 1) / 2;
	for (int i = p0; i < np; i += prepare_threads) {
		int px = np - i - 1;
		int y = floor((sqrt(1.0 + 8 * px) - 1) / 2);
		int x = px - y * (y + 1) / 2;
		x = nlines - x - 1;

		if (y < 0 || y >= nlines || x < 0 || x >= nlines) continue;


		float A1 = lines[x * 12], B1 = lines[x * 12 + 1], C1 = lines[x * 12 + 2];
		float mx1 = lines[x * 12 + 3], my1 = lines[x * 12 + 4];
		float l1 = lines[x * 12 + 7];
		float minx = lines[x * 12 + 8], miny = lines[x * 12 + 9],
			maxx = lines[x * 12 + 10], maxy = lines[x * 12 + 11];

		float A2 = lines[y * 12], B2 = lines[y * 12 + 1], C2 = lines[y * 12 + 2];
		float minx2 = lines[y * 12 + 8], miny2 = lines[y * 12 + 9],
			maxx2 = lines[y * 12 + 10], maxy2 = lines[y * 12 + 11];
		float l2 = lines[y * 12 + 7];

		float pA = A1, pB = B1, pC = C1, tx1 = minx2, tx2 = maxx2, ty1 = miny2, ty2 = maxy2;
		if (l2 > l1) {
			pA = A2; pB = B2; pC = C2;
			tx1 = minx; tx2 = maxx; ty1 = miny; ty2 = maxy;
		}
		float dL1 = fabs(tx1 * pA + ty1 * pB + pC), dL2 = fabs(tx2 * pA + ty2 * pB + pC);
		float dL = max(dL1, dL2);

		float d1 = (minx - minx2) * (minx - minx2) + (miny - miny2) * (miny - miny2);
		float d2 = (minx - maxx2) * (minx - maxx2) + (miny - maxy2) * (miny - maxy2);
		float d3 = (maxx - minx2) * (maxx - minx2) + (maxy - miny2) * (maxy - miny2);
		float d4 = (maxx - maxx2) * (maxx - maxx2) + (maxy - maxy2) * (maxy - maxy2);

		float md = d1;
		int type = 0;
		if (d2 < md)
			md = d2;
		if (d3 < md)
			md = d3;
		if (d4 < md)
			md = d4;

		float gx1 = lines[x * 12 + 5], gy1 = lines[x * 12 + 6];
		float gx2 = lines[y * 12 + 5], gy2 = lines[y * 12 + 6];

		bool ok =
			(md < l1* thresh_gap || md < l2* thresh_gap || md < thresh_whole) && dL < 4.0f &&
			gx1 * gx2 + gy1 * gy2>0.1;

		if (ok) { //todo: debug
			// rconn: line N connects to M lines, label is W, connecting: A,B,C....
			int old1 = atomic_inc(&rconn1[y * max_conn]);
			if (old1 >= max_conn - 3) {
				rconn1[y * max_conn] = max_conn - 3;
				continue;
			}
			atomic_max(&rconn1[y * max_conn + 1], 10000 - x); // init value: 0, aka. id 10000.
			rconn1[y * max_conn + old1 + 2] = x;

			int old2 = atomic_inc(&rconn1[x * max_conn]);
			if (old2 >= max_conn - 3) {
				rconn1[x * max_conn] = max_conn - 3;
				continue;
			}
			atomic_max(&rconn1[x * max_conn + 1], 10000 - x);
			rconn1[x * max_conn + old2 + 2] = x;
		}
		// y>x.
	}

}

// perform 5 times enough.
__kernel void line_merge_unionfind(global int* rconn, global int* len) {
	int id = get_global_id(0);
	int nlines = len[0];
	if (id >= nlines) return;
	for (int i = 0; i < rconn[id * max_conn]; ++i) {
		int om = rconn[rconn[id * max_conn + i + 2] * max_conn + 1]; // connected line's label.
		atomic_max(&rconn[id * max_conn + 1], om);
	}
}

__kernel void line_merge_reverse_list(global const int* rconn1, global int* len, global int* rconn2) {
	int id = get_global_id(0);
	int nlines = len[0];
	if (id >= nlines) return;

	int tid = rconn1[id * max_conn + 1];
	if (tid == 0) tid = id;
	else tid = 10000 - tid;
	if (tid < 0 || tid >= nlines)
		return;

	int old = atomic_inc(&rconn2[tid * max_conn]);
	if (old >= max_conn - 2) {
		rconn2[tid * max_conn] = max_conn - 2;
		return;
	}
	rconn2[tid * max_conn + old + 1] = id; // including me.
}

__kernel void line_merge_calc(global const int* rconn, global int* len, global float* lines) {
	int me = get_global_id(0);
	int nlines = len[0];
	int n = rconn[me * max_conn];
	if (me >= nlines || n <= 1) return;

	float x = 0, xx = 0, y = 0, yy = 0, xy = 0, minx = 999999, miny = 999999, maxx = 0, maxy = 0;
	float w = 0;
	for (int i = 0; i < n; ++i) {
		int p = rconn[me * max_conn + 1 + i];
		float mw = lines[p * 12 + 7];
		w += 2 * mw;
		float tx, ty;
		tx = lines[p * 12 + 8];
		ty = lines[p * 12 + 9];
		x += tx * mw; y += ty * mw; xx += tx * tx * mw; yy += ty * ty * mw; xy += tx * ty * mw;
		tx = lines[p * 12 + 10];
		ty = lines[p * 12 + 11];
		x += tx * mw; y += ty * mw; xx += tx * tx * mw; yy += ty * ty * mw; xy += tx * ty * mw;

		float minx2 = min(lines[p * 12 + 8], lines[p * 12 + 10]), miny2 = min(lines[p * 12 + 11], lines[p * 12 + 9]),
			maxx2 = max(lines[p * 12 + 8], lines[p * 12 + 10]), maxy2 = max(lines[p * 12 + 11], lines[p * 12 + 9]);
		if (minx2 < minx)  minx = minx2;
		if (maxx2 > maxx)  maxx = maxx2;
		if (miny2 < miny)  miny = miny2;
		if (maxy2 > maxy)  maxy = maxy2;
	}

	x /= w; y /= w; xx /= w; yy /= w; xy /= w;
	float a = xx - x * x, b = xy - x * y, c = yy - y * y;
	float sqt = sqrt((a - c) * (a - c) + 4 * b * b);
	float l1 = a + c + sqt, l2 = a + c - sqt;
	float dx, dy;
	if (fabs(a - l1 / 2) > fabs(c - l1 / 2)) {
		dy = l1 / 2 - a; dx = b;
	}
	else
	{
		dx = l1 / 2 - c; dy = b;
	}
	float norm = sqrt(dx * dx + dy * dy);
	dx /= norm; dy /= norm;
	float A = dy, B = -dx, C = dx * y - dy * x;
	lines[me * 12] = A;
	lines[me * 12 + 1] = B;
	lines[me * 12 + 2] = C;
	lines[me * 12 + 3] = x;
	lines[me * 12 + 4] = y;
	if (fabs(minx * A + miny * B + C) > fabs(minx * A + maxy * B + C)) {
		float t = miny;
		miny = maxy;
		maxy = t;
	}
	lines[me * 12 + 7] = (maxx - minx) * (maxx - minx) + (maxy - miny) * (maxy - miny);
	lines[me * 12 + 8] = (B * B * minx - A * B * miny - A * C) / (A * A + B * B);
	lines[me * 12 + 9] = (A * A * miny - A * B * minx - B * C) / (A * A + B * B);
	lines[me * 12 + 10] = (B * B * maxx - A * B * maxy - A * C) / (A * A + B * B);
	lines[me * 12 + 11] = (A * A * maxy - A * B * maxx - B * C) / (A * A + B * B);
}


__kernel void line_merge_reduce(global char* conn, global float* lines, global float* rlines, global int* len, global int* rconn) {
	const int me = get_local_id(0);

	local int sz_lines;
	if (me == 0) {
		sz_lines = 0;
	}

	// filter out valid points:
	for (int i = 0; i < len[0] / reduce_sz + (len[0] % reduce_sz > 0); ++i) {
		int x0 = me + (i * reduce_sz);
		bool is_point = (x0 < len[0] && rconn[x0 * max_conn] >= 1);
		int p_sum = sub_group_scan_inclusive_add(is_point); // simd 32
		int pos = sz_lines + p_sum - 1;
		if (is_point && pos < max_lines) {
			rlines[pos * 12] = lines[x0 * 12];
			rlines[pos * 12 + 1] = lines[x0 * 12 + 1];
			rlines[pos * 12 + 2] = lines[x0 * 12 + 2];
			rlines[pos * 12 + 3] = lines[x0 * 12 + 3];
			rlines[pos * 12 + 4] = lines[x0 * 12 + 4];
			rlines[pos * 12 + 5] = lines[x0 * 12 + 5];
			rlines[pos * 12 + 6] = lines[x0 * 12 + 6];
			rlines[pos * 12 + 7] = lines[x0 * 12 + 7];
			rlines[pos * 12 + 8] = lines[x0 * 12 + 8];
			rlines[pos * 12 + 9] = lines[x0 * 12 + 9];
			rlines[pos * 12 + 10] = lines[x0 * 12 + 10];
			rlines[pos * 12 + 11] = lines[x0 * 12 + 11];
			// add to spatial index.
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (me == reduce_sz - 1)
			sz_lines += p_sum;
	}

	if (me == reduce_sz - 1) {
		len[3] = len[0]; // segments.
		len[0] = min(sz_lines, max_lines); // lines
		len[1] = 0; //quads
		len[2] = 0; //tags
	}
}

#define quad_thresh 1/2.0f;
#define conn_b 16
__kernel void quadrilateral_prepare(global char* conn, global float* lines, global int* len, global int* reduced_conns) {

	int p0 = get_global_id(0);
	int nlines = len[0];
	int np = (nlines) * (nlines - 1) / 2;
	for (int i = p0; i < np; i += prepare_threads) {
		int first = floor((sqrt(1.0 + 8 * i) - 1) / 2);
		int second = i - first * (first + 1) / 2;

		float A1 = lines[first * 12], B1 = lines[first * 12 + 1], C1 = lines[first * 12 + 2];
		float A2 = lines[second * 12], B2 = lines[second * 12 + 1], C2 = lines[second * 12 + 2];

		float gx1 = lines[first * 12 + 5], gy1 = lines[first * 12 + 6];
		float gx2 = lines[second * 12 + 5], gy2 = lines[second * 12 + 6];

		float mx1 = lines[first * 12 + 3], my1 = lines[first * 12 + 4];
		float mx2 = lines[second * 12 + 3], my2 = lines[second * 12 + 4];

		float ptx = (C2 * B1 - C1 * B2) / (A1 * B2 - A2 * B1);
		float pty = (A1 * C2 - A2 * C1) / (B1 * A2 - A1 * B2);

		float vec1x = mx1 - ptx, vec1y = my1 - pty;
		float vec2x = mx2 - ptx, vec2y = my2 - pty;

		float c1 = (vec1x * gx2 + vec1y * gy2) / sqrt(vec1x * vec1x + vec1y * vec1y); // should >0
		float c2 = (vec2x * gx1 + vec2y * gy1) / sqrt(vec2x * vec2x + vec2y * vec2y);

		float l1 = lines[first * 12 + 7];
		float l2 = lines[second * 12 + 7];
		float l = min(l1, l2) * quad_thresh;

		float minx1 = lines[first * 12 + 8], miny1 = lines[first * 12 + 9],
			maxx1 = lines[first * 12 + 10], maxy1 = lines[first * 12 + 11];
		float minx2 = lines[second * 12 + 8], miny2 = lines[second * 12 + 9],
			maxx2 = lines[second * 12 + 10], maxy2 = lines[second * 12 + 11];

		float d1a = (ptx - minx1) * (ptx - minx1) + (pty - miny1) * (pty - miny1);
		float d1b = (ptx - maxx1) * (ptx - maxx1) + (pty - maxy1) * (pty - maxy1);
		float d1 = min(d1a, d1b);

		float d2a = (ptx - minx2) * (ptx - minx2) + (pty - miny2) * (pty - miny2);
		float d2b = (ptx - maxx2) * (ptx - maxx2) + (pty - maxy2) * (pty - maxy2);
		float d2 = min(d2a, d2b);

		// wrapping a black region and near intersection.
		bool ok = conn[first * nlines + second] = conn[second * nlines + first] = c1 > 0.3 && c2 > 0.3 && d1 < l&& d2 < l;
		if (ok) {
			if (reduced_conns[first * conn_b] >= conn_b || reduced_conns[second * conn_b] >= conn_b)
				return;
			int old1 = atomic_inc(&reduced_conns[first * conn_b]);
			if (old1 >= conn_b - 2) {
				reduced_conns[first * conn_b] = conn_b - 2;
				return;
			}
			int old2 = atomic_inc(&reduced_conns[second * conn_b]);
			if (old2 >= conn_b - 2) {
				reduced_conns[second * conn_b] = conn_b - 2;
				return;
			}

			reduced_conns[first * conn_b + old1 + 1] = second;
			reduced_conns[second * conn_b + old2 + 1] = first;
		}
	}

}

__kernel void quadrilateral_find(global char* conn, global float* lines, global int* len, global int* rconns, global float* points, global float* H) {

	int nlines = len[0];
	int first = get_global_id(0);
	if (first >= nlines) return;
	//todo: use dfs.
	for (int i = 1; i <= rconns[first * conn_b]; ++i) {
		int second = rconns[first * conn_b + i];
		if (second < first + 1) continue;
		for (int j = 1; j <= rconns[second * conn_b]; ++j) {
			int third = rconns[second * conn_b + j];
			if (third < first + 1) continue;
			for (int k = 1; k <= rconns[third * conn_b]; ++k) {
				int fourth = rconns[third * conn_b + k];
				if (fourth < second + 1) continue;
				if (conn[fourth * nlines + first]) {
					float A1 = lines[first * 12], B1 = lines[first * 12 + 1], C1 = lines[first * 12 + 2];
					float A2 = lines[second * 12], B2 = lines[second * 12 + 1], C2 = lines[second * 12 + 2];
					float A3 = lines[third * 12], B3 = lines[third * 12 + 1], C3 = lines[third * 12 + 2];
					float A4 = lines[fourth * 12], B4 = lines[fourth * 12 + 1], C4 = lines[fourth * 12 + 2];

					float x1, y1, x2, y2, x3, y3, x4, y4;

					x1 = (C2 * B1 - C1 * B2) / (A1 * B2 - A2 * B1);
					y1 = (A1 * C2 - A2 * C1) / (B1 * A2 - A1 * B2);
					x2 = (C3 * B2 - C2 * B3) / (A2 * B3 - A3 * B2);
					y2 = (A2 * C3 - A3 * C2) / (B2 * A3 - A2 * B3);
					x3 = (C4 * B3 - C3 * B4) / (A3 * B4 - A4 * B3);
					y3 = (A3 * C4 - A4 * C3) / (B3 * A4 - A3 * B4);
					x4 = (C4 * B1 - C1 * B4) / (A1 * B4 - A4 * B1);
					y4 = (A1 * C4 - A4 * C1) / (B1 * A4 - A1 * B4);

					// swap 2/4 if clockwise:
					if ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1) <= 0) {
						float tmp;
						tmp = x2;
						x2 = x4;
						x4 = tmp;
						tmp = y2;
						y2 = y4;
						y4 = tmp;
					}

					// filter illegal quads.

					// homography computation:

					float A[] = {
						-1, -1, 1, 0, 0, 0, x1, x1, x1,
						0, 0, 0, -1, -1, 1, y1, y1, y1,
						-1, 1, 1, 0, 0, 0, x2, -x2, x2,
						0, 0, 0, -1, 1, 1, y2, -y2, y2,
						1, 1, 1, 0, 0, 0, -x3, -x3, x3,
						0, 0, 0, 1, 1, 1, -y3, -y3, y3,
						1, -1, 1, 0, 0, 0, -x4, x4, x4,
						0, 0, 0, 1,-1, 1, -y4, y4, y4,
					};


					float epsilon = 0.000001;
					bool singular = false;

					// Fucking Slow Gaussian Elimination Equation Solver
					for (int col = 0; col < 8; col++) {
						// Find best row to swap with.
						float max_val = 0;
						int max_val_idx = -1;
						for (int row = col; row < 8; row++) {
							float val = A[row * 9 + col];
							if (val < 0) val = -val;
							if (val > max_val) {
								max_val = val;
								max_val_idx = row;
							}
						}

						if (max_val < epsilon) {
							// singular matrix?
							singular = true;
							break;
						}

						// Swap to get best row.
#pragma unroll
						for (int i = col; i < 9; i++) {
							float tmp = A[col * 9 + i];
							A[col * 9 + i] = A[max_val_idx * 9 + i];
							A[max_val_idx * 9 + i] = tmp;
						}

						//// Do eliminate.
						for (int i = col + 1; i < 8; i++) {
							float f = A[i * 9 + col] / A[col * 9 + col];
							A[i * 9 + col] = 0;
							for (int j = col + 1; j < 9; j++) {
								A[i * 9 + j] -= f * A[col * 9 + j];
							}
						}
					}

					if (singular)
						continue;

					// OK:

					int ptr = atomic_inc(&len[1]);
					if (ptr >= max_quads) {
						len[1] = max_quads;
						return;
					}

					points[ptr * 8 + 0] = x1;
					points[ptr * 8 + 1] = y1;

					points[ptr * 8 + 2] = x2;
					points[ptr * 8 + 3] = y2;

					points[ptr * 8 + 4] = x3;
					points[ptr * 8 + 5] = y3;

					points[ptr * 8 + 6] = x4;
					points[ptr * 8 + 7] = y4;

					// Back solve. 
					for (int col = 7; col >= 0; col--) {
						float sum = 0;
						for (int i = col + 1; i < 8; i++) {
							sum += A[col * 9 + i] * A[i * 9 + 8];
						}
						A[col * 9 + 8] = (A[col * 9 + 8] - sum) / A[col * 9 + col];
					}
					H[ptr * 8 + 0] = A[8];
					H[ptr * 8 + 1] = A[17];
					H[ptr * 8 + 2] = A[26];
					H[ptr * 8 + 3] = A[35];
					H[ptr * 8 + 4] = A[44];
					H[ptr * 8 + 5] = A[53];
					H[ptr * 8 + 6] = A[62];
					H[ptr * 8 + 7] = A[71];
				}

			}
		}
	}
}


#define rec_sz 1600
#define rec_l 40

__kernel void homography_transform(global const unsigned char* pix, global float* H, global int* len, global unsigned char* rectified, int w, int h) {
	int id = get_group_id(0);
	if (id >= len[1]) return;
	int lid = get_local_id(0);
	for (int i = 0; i < rec_sz / reduce_sz; ++i) {
		int me = lid + i * reduce_sz;
		int ix = me % rec_l, iy = me / rec_l;
		float x = ix / 20.0f - 1;
		float y = iy / 20.0f - 1;
		//homography compute:

		float xx = H[id * 8 + 0] * x + H[id * 8 + 1] * y + H[id * 8 + 2];
		float yy = H[id * 8 + 3] * x + H[id * 8 + 4] * y + H[id * 8 + 5];
		float zz = H[id * 8 + 6] * x + H[id * 8 + 7] * y + 1;

		float px = xx / zz;
		float py = yy / zz;

		int x1 = floor(px);
		int x2 = x1 + 1;
		float dx = px - x1;
		int y1 = floor(py);
		int y2 = y1 + 1;
		float dy = py - y1;
		if (x1 < 0) x1 = 0; if (x2 < 0) x2 = 0; if (y1 < 0) y1 = 0; if (y2 < 0) y2 = 0;
		if (x1 >= w) x1 = w - 1; if (x2 >= w) x2 = w - 1; if (y1 >= h) y1 = h - 1; if (y2 >= h) y2 = h - 1;

		if (me < 1600)
			rectified[id * rec_sz + me] = (pix[y1 * w + x1] * (1 - dx) * (1 - dy) +
				pix[y1 * w + x2] * dx * (1 - dy) +
				pix[y2 * w + x1] * (1 - dx) * dy +
				pix[y2 * w + x2] * dx * dy);
	}
}


#define code_n 12512
// actually 12512 used.

void SWAP(int* a, int id1, int id2) {
	if (a[id1] > a[id2]) {
		int tmp = a[id2];
		a[id2] = a[id1];
		a[id1] = tmp;
	}
}


__kernel void rect_decoding(global const unsigned char* rectified, global int* len, int iw, global int* result, global int* ordering) {
	int id = get_group_id(0);
	if (id >= len[1]) return;
	int lid = get_local_id(0);
	local int cval[64];
	cval[lid] = 0;
	cval[lid + 32] = 0;

	//calc value.
	for (int i = 0; i < 256 / reduce_sz; ++i) {
		int me = lid + i * reduce_sz;
		int ix = me % 16, iy = me / 16;
		int x = ix % 2 + (ix / 2) * 5 + 2;
		int y = iy % 2 + (iy / 2) * 5 + 2;
		atomic_add(&cval[(iy / 2) * 8 + ix / 2], rectified[id * rec_sz + y * 40 + x]);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// neighbor value mapping.
	local int cmax[36], cmin[36];
	local char c[36];
	for (int i = 0; i < 36 / reduce_sz + (36 % reduce_sz > 0); ++i) {
		int me = lid + i * reduce_sz;
		if (me < 36) {
			int mx = me % 6 + 1;
			int my = me / 6 + 1;
			int mid = my * 8 + mx;
			int a[8] = { cval[mid - 9], cval[mid - 8],cval[mid - 7],cval[mid - 1],cval[mid + 1],cval[mid + 7],cval[mid + 8],cval[mid + 9] };
			SWAP(a, 0, 1);
			SWAP(a, 2, 3);
			SWAP(a, 0, 2);
			SWAP(a, 1, 3);
			SWAP(a, 1, 2);
			SWAP(a, 4, 5);
			SWAP(a, 6, 7);
			SWAP(a, 4, 6);
			SWAP(a, 5, 7);
			SWAP(a, 5, 6);
			SWAP(a, 0, 4);
			SWAP(a, 1, 5);
			SWAP(a, 1, 4);
			SWAP(a, 2, 6);
			SWAP(a, 3, 7);
			SWAP(a, 3, 6);
			SWAP(a, 2, 4);
			SWAP(a, 3, 5);
			SWAP(a, 3, 4);
			int lv = a[1] - a[0], now;
			now = a[2] - a[1]; if (now > lv) lv = now;
			now = a[3] - a[2]; if (now > lv) lv = now;
			now = a[4] - a[3]; if (now > lv) lv = now;
			now = a[5] - a[4]; if (now > lv) lv = now;
			now = a[6] - a[5]; if (now > lv) lv = now;
			now = a[7] - a[6]; if (now > lv) lv = now;
			if (lv > (a[7] - a[0]) * 0.3f) { // bw distinguishable.
				if (abs_diff(a[7], cval[mid]) < abs_diff(a[0], cval[mid]))
					c[me] = 0;
				else
					c[me] = 1;
			}
			else
				c[me] = -1; //same to others.
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// propagate:
	for (int n = 0; n < 3; ++n) {
		for (int i = 0; i < 36 / reduce_sz + (36 % reduce_sz > 0); ++i) {
			int me = lid + i * reduce_sz;
			if (me >= 36) continue;
			if (c[me] != -1) {
				if (c[me - 7] == -1) c[me - 7] = c[me];
				if (c[me - 6] == -1) c[me - 6] = c[me];
				if (c[me - 5] == -1) c[me - 5] = c[me];
				if (c[me - 1] == -1) c[me - 1] = c[me];
				if (c[me + 1] == -1) c[me + 1] = c[me];
				if (c[me + 7] == -1) c[me + 7] = c[me];
				if (c[me + 6] == -1) c[me + 6] = c[me];
				if (c[me + 5] == -1) c[me + 5] = c[me];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	 
	local int c1, c2, c3, c4, e1, e2, e3, e4;
	if (lid == 0) {
		c1 = c2 = c3 = c4 = e1 = e2 = e3 = e4 = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < 36 / reduce_sz + (36 % reduce_sz > 0); ++i) {
		int me = lid + i * reduce_sz;

		if (me < 36) {
			int val = c[me];

			if (ordering[me * 4 + 0] >= 0) // is payload bit.
				atomic_add(&c1, val << ordering[me * 4 + 0]);
			else if (!val) // is check bit, should be black.
				atomic_inc(&e1);

			if (ordering[me * 4 + 1] >= 0)
				atomic_add(&c2, val << ordering[me * 4 + 1]);
			else if (!val)
				atomic_inc(&e2);

			if (ordering[me * 4 + 2] >= 0)
				atomic_add(&c3, val << ordering[me * 4 + 2]);
			else if (!val)
				atomic_inc(&e3);

			if (ordering[me * 4 + 3] >= 0)
				atomic_add(&c4, val << ordering[me * 4 + 3]);
			else if (!val)
				atomic_inc(&e4);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	 
	if (lid == 0) { // debug output
		result[id * 8 + 0] = c1;
		result[id * 8 + 1] = e1;
		result[id * 8 + 2] = c2;
		result[id * 8 + 3] = e2;
		result[id * 8 + 4] = c3;
		result[id * 8 + 5] = e3;
		result[id * 8 + 6] = c4;
		result[id * 8 + 7] = e4;
	}
}

__kernel void error_correcting(global int* result, global int* len, global int* result_ok, global int* codes) {
	int rn = len[1] * 4;
	if (get_global_id(0) >= code_n) return;
	int code = codes[get_global_id(0)];
	for (int i = 0; i < rn; ++i) {
		int c = result[i * 2 + 0];
		int e = result[i * 2 + 1];
		if (e == 0 && popcount(code ^ c) <= 2) {
			int tid = atomic_inc(&len[2]);
			if (tid >= max_quads) {
				len[2] = max_quads;
				return;
			}
			result_ok[tid * 3 + 0] = get_global_id(0) + 1; // id start from 1.
			result_ok[tid * 3 + 1] = i % 4;
			result_ok[tid * 3 + 2] = i / 4;
		}
	}
}
