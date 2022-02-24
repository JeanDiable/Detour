#version 330 core
layout (location = 10) in vec3 aPos;
layout (location = 11) in float aColor;

out VS_OUT {
    float maxAlpha;
} vs_out;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    vs_out.maxAlpha = aColor;
}