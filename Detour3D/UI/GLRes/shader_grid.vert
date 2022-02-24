#version 330 core
layout (location = 2) in vec3 aPos;
layout (location = 3) in float aMaxAlpha;

out VS_OUT {
    float maxAlpha;
} vs_out;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    vs_out.maxAlpha = aMaxAlpha;
}