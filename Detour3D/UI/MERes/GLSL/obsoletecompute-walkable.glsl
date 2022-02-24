#version 430

layout(std430, binding = 0) buffer outputSSBO {
    float result[];
};

layout(std430, binding = 1) buffer inputFrameHeader {
    float header[][8];
//    float frameLen;
//    float offset;
//    float centerX;
//    float centerY;
//    float blX;
//    float blY;
//    float urX;
//    float urY;
};

layout(std430, binding = 2) buffer inputFrameData {
    float data[][5];
};

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

uniform mat4 projMat;
uniform vec3 cameraPosition;

uniform int computeWidth;
uniform int computeHeight;

uniform int frameNum;

uint idx;
float targetX;
float targetY;

bool terminate = false;

float compareRadius = radians(5);

vec3 ApplyProjection(vec3 ndc)
{
    float xx = ndc.x, yy = ndc.y, zz = ndc.z;

//    float d = 1.0 / (projMat[0][3] * xx + projMat[1][3] * yy + projMat[2][3] * zz + projMat[3][3]);
//
//    float x = (projMat[0][0] * xx + projMat[1][0] * yy + projMat[2][0] * zz + projMat[3][0]) * d;
//    float y = (projMat[0][1] * xx + projMat[1][1] * yy + projMat[2][1] * zz + projMat[3][1]) * d;
//    float z = (projMat[0][2] * xx + projMat[1][2] * yy + projMat[2][2] * zz + projMat[3][2]) * d;

    float d = 1.0 / (projMat[3][0] * xx + projMat[3][1] * yy + projMat[3][2] * zz + projMat[3][3]);

    float x = (projMat[0][0] * xx + projMat[0][1] * yy + projMat[0][2] * zz + projMat[0][3]) * d;
    float y = (projMat[1][0] * xx + projMat[1][1] * yy + projMat[1][2] * zz + projMat[1][3]) * d;
    float z = (projMat[2][0] * xx + projMat[2][1] * yy + projMat[2][2] * zz + projMat[2][3]) * d;

    return vec3(x, y, z);
}

// mode true: left, false: right. 
int PointBinarySearch(int low, int high, float targetTheta, bool mode)
{
    while (low <= high)
    {
        int mid = low + (high - low) / 2;
        if (mode)
        {
            if (targetTheta <= data[mid][2]) high = mid - 1;
            else low = mid + 1;
        }
        else
        {
            if (targetTheta >= data[mid][2]) low = mid + 1;
            else high = mid - 1;
        }
    }

    if (mode) return low;
    return high;
}

void ConductProcedure(int frameId)
{  
    int frameLen = int(header[frameId][0]);
    int frameOffset = int(header[frameId][1]);
    float centerX = header[frameId][2];
    float centerY = header[frameId][3];

    vec3 tmp = vec3(targetX - centerX, targetY - centerY, 0);
    float dist = length(tmp);
    float theta = acos(dot(tmp, vec3(1, 0, 0)) / dist);
    vec3 cro = cross(tmp, vec3(1, 0, 0));
    if (cro.z > 0) theta = radians(360) - theta;
    
    // special cases
    if (theta < data[frameOffset][2] || theta > data[frameOffset + frameLen - 1][2])
    {
        
        return;
    }
    if (theta - compareRadius < data[frameOffset][2]) return;
    if (theta + compareRadius > data[frameOffset + frameLen - 1][2]) return;

    int low = PointBinarySearch(frameOffset, frameOffset + frameLen, theta - compareRadius, true);
    int high = PointBinarySearch(frameOffset, frameOffset + frameLen, theta + compareRadius, false);
    
    float maxDist = -1;
    float totalDist = 0;
    for (int i = low; i <= high; ++i)
    {
        if (data[i][4] < 1) continue;
        maxDist = max(maxDist, data[i][3]);
        totalDist += data[i][3];
    }
    float avgDist = totalDist / (high - low + 1);
    for (int i = low; i <= high; ++i)
    {
        if (data[i][4] < 1) continue;
        if (dist < data[i][3])
        {
            float threshold = min(1, data[i][3] / 3);
            float incFactor = dist < threshold ? 1.0 : 1.0 / pow(max(1, dist), 2);
            result[idx] += 0.3 * incFactor;
        }
    }
    if (dist > avgDist) result[idx] -= 0.1;
}

void FrameSearch()
{
    for (int i = 0; i < frameNum; ++i)
    {
        float blX = header[i][4];
        float blY = header[i][5];
        float urX = header[i][6];
        float urY = header[i][7];

        // check bounding box
        if (targetX < blX || targetX > urX || targetY < blY || targetY > urY) continue;

        ConductProcedure(i);

        if (terminate) return;
    }
}

void main()
{
    // Unproject to world
    vec3 ndc = vec3(
        float(gl_GlobalInvocationID.x) / float(computeWidth) * 2.0 - 1.0,
        -float(gl_GlobalInvocationID.y) / float(computeHeight) * 2.0 + 1.0,
        1
    );

    vec3 proj = ApplyProjection(ndc);

    float dz = cameraPosition.z - proj.z;

    targetX = cameraPosition.x + cameraPosition.z / dz * (proj.x - cameraPosition.x);
    targetY = cameraPosition.y + cameraPosition.z / dz * (proj.y - cameraPosition.y);

    // see if walkable
    idx = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * computeWidth;
    result[idx] = 0;

    FrameSearch();
}
