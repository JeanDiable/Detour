#version 430 core

layout (location = 4) in vec3 aPosition;
layout (location = 5) in vec4 aColor;

out VS_OUT {
    float aColor;
} vs_out;

void main()
{
    gl_Position = vec4(aPosition, 1.0);
    vs_out.aColor = aColor.x;
}