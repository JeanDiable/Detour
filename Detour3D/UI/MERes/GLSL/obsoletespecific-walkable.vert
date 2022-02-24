#version 330 core
layout (location = 12) in float alpha;

out float Alpha;

uniform mat4 projectionMatrix;

uniform int computeWidth;
uniform int computeHeight;

void main()
{
    int xIdx = gl_VertexID % computeWidth;
    int yIdx = gl_VertexID / computeWidth;

    gl_Position = vec4(xIdx, computeHeight - yIdx, 0, 1) * projectionMatrix;
    gl_Position.z = 1;
    Alpha = alpha > 0.5 ? 1 : (alpha < 0 ? 0 : alpha / 0.5);
    gl_PointSize = 10;
}