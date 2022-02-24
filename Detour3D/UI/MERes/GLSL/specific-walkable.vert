#version 330 core
layout (location = 12) in float alpha;

out float Alpha;

uniform mat4 projectionMatrix;

uniform int computeWidth;
uniform int computeHeight;
uniform float walkableFactor;

void main()
{
    float xIdx = gl_VertexID % computeWidth;// * walkableFactor;
    float yIdx = gl_VertexID / computeWidth;// * walkableFactor;

    gl_Position = vec4(xIdx, computeHeight - yIdx, 0, 1) * projectionMatrix;
    gl_Position.z = 1;
    gl_PointSize = walkableFactor-0.2;
    Alpha = alpha;// > 0.5 ? 1 : 0;
}