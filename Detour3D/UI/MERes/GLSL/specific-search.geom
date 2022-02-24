#version 430 core

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

in VS_OUT {
    float aColor;
} gs_in[];

out float color;

void main()
{
    mat4 transf = modelMatrix * viewMatrix * projectionMatrix;

    for (int i = 0; i < 3; ++i)
    {
        color = gs_in[i].aColor;
        gl_Position = gl_in[i].gl_Position * transf;
        EmitVertex();
    }

    EndPrimitive();
}