#version 330 core
layout (location = 5) in vec3 aPos;

uniform mat4 m_model;
uniform mat4 m_view;
uniform mat4 m_projection;

void main()
{
    gl_Position = vec4(aPos, 1.0) * m_model * m_view * m_projection;
}