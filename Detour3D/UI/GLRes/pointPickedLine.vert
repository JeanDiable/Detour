#version 330 core
layout (location = 4) in vec3 aPos;

uniform mat4 m_model;
uniform mat4 m_view;
uniform mat4 m_projection;

out float alpha;

void main()
{
    gl_Position = vec4(aPos, 1.0) * m_model * m_view * m_projection;
    alpha = aPos.y > 0 ? 0.8 : 0.2;
}