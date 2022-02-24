int toHash(float x, float y, int p)
{
    int xx = ((int)x) / p;
    int yy = ((int)y) / p;

    return (1103515245 * xx + 12345 + yy) * 1140671485 + 12820163 + yy;
}

int reHash(int hash) {
    return hash * 1140671485 + 12820163;
}

// line:
#define BLEN1 ${blen1} //1024
#define BLEN2 ${blen2} //512

struct next {
    unsigned int next_hash; //同桶但不同哈希的一个点
    unsigned int next_point; //同哈希的下一个点
};

__kernel void computeHash(global const float2* ref_arr, int len,
    global int* hashs1, int rect1,
    global int* hashs2, int rect2) {

    const int tid = get_global_id(0);

    if (tid >= len) return;

    hashs1[tid] = toHash(ref_arr[tid].x, ref_arr[tid].y, rect1);
    hashs2[tid] = toHash(ref_arr[tid].x, ref_arr[tid].y, rect2);
}

void putbucket(global const int* hashs, int blen, global int* bucket_arr, int tid, global struct next* next) {
    int myH = hashs[tid];
    int line1 = myH % blen;
    if (atomic_cmpxchg(&bucket_arr[line1], -1, tid) != -1) { // failed to occupy bucket.
        int prev = bucket_arr[line1];
        while (true) {
            if (hashs[prev] == myH) {
                // 同哈希列
                next[tid].next_point = atomic_xchg(&next[prev].next_point, tid);
                break;
            }
            if ((prev = atomic_cmpxchg(&next[prev].next_hash, -1, tid)) == -1)
                break; // success to occupy next_hash.
        }
    }
}
__kernel void init(int len,
    global const int* hashs1,
    global const int* hashs2,
    global int* bucket_arr1, int rect1, global struct next* next1,
    global int* bucket_arr2, int rect2, global struct next* next2)
{
    const int tid = get_global_id(0);

    if (tid >= len) return;

    putbucket(hashs1, BLEN1, bucket_arr1, tid, next1);
    putbucket(hashs2, BLEN2, bucket_arr2, tid, next2);
}



float ComputeWeight(float td)
{
    float CWXs[] = { 0,10,20,30,40,80,200,400,800 };
    float CWYs[] = { 1, 0.99f, 0.95f, 0.8f, 0.55f, 0.3f, 0.15f, 0.05f, 0 };
    int st = 0, ed = 8;
    while (ed - st > 1)
    {
        int mid = (ed + st) / 2;
        if (CWXs[mid] < td)
            st = mid;
        else ed = mid;
    }

    int d = CWXs[ed] - CWXs[st];
    return (td - CWXs[st]) / d * CWYs[st] + (CWXs[ed] - td) / d * CWYs[ed];
}

__kernel void batchnn(float fcos, float fsin, float fmdx, float fmdy, global const float2* observed, int len,
    global const float2* ref_arr,
    global const int* bucket_arr1, int rect1, global struct next* next1, global const int* hashes1,
    global const int* bucket_arr2, int rect2, global struct next* next2, global const int* hashes2,
    global float2* targets, global float* ws)
{
    int tid = get_global_id(0);
    if (tid >= len) return;

    float tx = observed[tid].x * fcos - observed[tid].y * fsin + fmdx;
    float ty = observed[tid].x * fsin + observed[tid].y * fcos + fmdy;
    int tries = 0;
    float d1 = 999999, d2 = 999999;
    short best1 = -1, best2 = -1;

    int xs[] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    int ys[] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

    for (int j = 0; j < 9; ++j) {
        int h1 = toHash(tx + xs[j], ty + ys[j], rect1);
        int line1 = h1 % BLEN1;
        int head = bucket_arr1[line1];
        if (head == -1) continue;
        while (hashes1[head] != h1 && head != -1)
            head = next1[head].next_hash;
        int i = 0;
        while (i++ < 8 && head != -1) {
            float2 p = ref_arr[head];
            float myd = (p.x - tx) * (p.x - tx) + (p.y - ty) * (p.y - ty);
            if (myd < d1)
            {
                d2 = d1;
                best2 = best1;
                d1 = myd;
                best1 = head;
            }
            else if (myd < d2 && myd > d1)
            {
                d2 = myd;
                best2 = head;
            }
            head = next1[head].next_point;
        }
    }

    if (best1 != -1)
    {
        float2 p1 = ref_arr[best1];
        if (best2 != -1)
        {
            float2 p2 = ref_arr[best1];
            float lxB = p1.x - p2.x, lyB = p1.y - p2.y, dABdB = lxB * lxB + lyB * lyB;
            float u2C = ((tx - p1.x) * lxB + (ty - p1.y) * lyB) / dABdB;
            float pCx = p1.x + u2C * lxB; // predicted point perpendicular
            float pCy = p1.y + u2C * lyB;

            targets[tid].x = pCx;
            targets[tid].y = pCy;
            ws[tid] = ComputeWeight(sqrt((pCx - tx) * (pCx - tx) + (pCy - ty) * (pCy - ty)));
        }
        else {
            targets[tid].x = p1.x;
            targets[tid].y = p1.y;
            ws[tid]= ComputeWeight(sqrt((p1.x - tx) * (p1.x - tx) + (p1.y - ty) * (p1.y - ty)));
        }
        return;
    }

    for (int j = 0; j < 9; ++j) {
        int h2 = toHash(tx + xs[j], ty + ys[j], rect2);
        int line2 = h2 % BLEN2;
        int head = bucket_arr2[line2];
        if (head == -1) continue;
        while (hashes2[head] != h2 && head != -1)
            head = next2[head].next_hash;
        int i = 0;
        while (i++ < 8 && head != -1) {
            float2 p = ref_arr[head];
            float myd = (p.x - tx) * (p.x - tx) + (p.y - ty) * (p.y - ty);
            if (myd < d1)
            {
                d1 = myd;
                best1 = head;
            }
            head = next2[head].next_point;
        }
    }

    if (best1 != -1) {
        float2 p1 = ref_arr[best1];
        targets[tid].x = p1.x;
        targets[tid].y = p1.y;
        ws[tid] = ComputeWeight(sqrt((p1.x - tx) * (p1.x - tx) + (p1.y - ty) * (p1.y - ty)));
    }
    else
        ws[tid] = 0;
}

////////////////////////////////////////////////////////