#version 430 core

layout(std430, binding = 0) buffer outputSSBO {
    float result[];
};

layout(std430, binding = 1) buffer inputCounter {
    float pixelCount[];
};

uniform int searchWidth;
uniform int searchHeight;

in float color;
out vec4 FragColor;

void main()
{
    uint pixelIdx = uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * searchWidth;
    int pixelCnt = int(pixelCount[pixelIdx]);

    if (pixelCnt < 16)
    {
        uint idx = uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * searchWidth + pixelCnt * searchHeight * searchWidth;
        result[idx] = color.x;//color.x;

        pixelCount[pixelIdx]++;
    }

    FragColor = vec4(0.3);// vec4(color);
}