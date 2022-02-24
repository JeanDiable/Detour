//#version 430
//
//#define MAX_BOX_FRAME_COUNT 2

layout(std430, binding = 0) buffer outputSSBO {
    float result[];
};

layout(std430, binding = 1) buffer inputBoxesData {
    int boxes[][MAX_BOX_FRAME_COUNT];
};

layout(std430, binding = 2) buffer inputFrameHeader {
    float header[][2];
};

layout(std430, binding = 3) buffer inputFrameData {
    float data[][360];
};

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

uniform mat4 projMat;
uniform vec3 cameraPosition;

uniform int computeWidth;
uniform int computeHeight;
//uniform float walkableFactor;

uniform int boxUnit;
uniform int xBoxesCount;
//uniform int yBoxesCount;
uniform int xStartBox;
uniform int yStartBox;

uniform int execFlag;

uint pixelId;
float targetX;
float targetY;

vec3 ApplyProjection(vec3 ndc)
{
    float xx = ndc.x, yy = ndc.y, zz = ndc.z;

    float d = 1.0 / (projMat[3][0] * xx + projMat[3][1] * yy + projMat[3][2] * zz + projMat[3][3]);

    float x = (projMat[0][0] * xx + projMat[0][1] * yy + projMat[0][2] * zz + projMat[0][3]) * d;
    float y = (projMat[1][0] * xx + projMat[1][1] * yy + projMat[1][2] * zz + projMat[1][3]) * d;
    float z = (projMat[2][0] * xx + projMat[2][1] * yy + projMat[2][2] * zz + projMat[2][3]) * d;

    return vec3(x, y, z);
}

void main()
{
    if (execFlag <= 0) return;    

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
    pixelId = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * computeWidth;
    //result[pixelId] = 0;
    if (result[pixelId] >= 1)
    {
        result[pixelId] = 1;
        return;
    }

    int xBoxOffset = (int(floor(targetX / boxUnit) * boxUnit + boxUnit / 2) - xStartBox) / boxUnit;
    int yBoxOffset = (int(floor(targetY / boxUnit) * boxUnit + boxUnit / 2) - yStartBox) / boxUnit;
    int boxId = xBoxOffset + yBoxOffset * xBoxesCount;

    for (int i = 0; i < MAX_BOX_FRAME_COUNT; ++i)
    {
        int frameId = int(boxes[boxId][i]);
        if (frameId < 0) break;

        float frameCenterX = header[frameId][0];
        float frameCenterY = header[frameId][1];

        vec3 tmp = vec3(targetX - frameCenterX, targetY - frameCenterY, 0);
        float dist = length(tmp);
        float theta = acos(dot(tmp, vec3(1, 0, 0)) / dist);
        vec3 cro = cross(tmp, vec3(1, 0, 0));
        if (cro.z > 0) theta = radians(360) - theta;
        theta = degrees(theta);

        int lowerTheta = int(floor(theta));
        lowerTheta = (lowerTheta + 360) % 360;
        int upperTheta = int(ceil(theta));
        upperTheta = (upperTheta + 360) % 360;

//        if (abs(data[frameId][lowerTheta] - data[frameId][upperTheta]) > 0.2 * data[frameId][lowerTheta]) continue;
//        else if (abs(data[frameId][lowerTheta] - data[frameId][upperTheta]) > 0.2 * data[frameId][upperTheta]) continue;

        float avgDist = (data[frameId][lowerTheta] + data[frameId][upperTheta]) / 2;

        if (dist <= avgDist)
        {
            result[pixelId] += 1 / (dist + 0.0001);
            //break;
        }
        if (result[pixelId] >= 1)
        {
            result[pixelId] = 1;
            return;
        }
    }
}
