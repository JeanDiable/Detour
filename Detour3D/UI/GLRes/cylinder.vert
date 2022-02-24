#version 330 core
layout (location = 8) in vec3 aPos;
layout (location = 9) in vec3 aNorm;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 m_model;
uniform mat4 m_view;
uniform mat4 m_projection;

void main()
{
    gl_Position = vec4(aPos, 1.0) * m_model * m_view * m_projection;

    FragPos = vec3(vec4(aPos, 1.0) * m_model);
    Normal = mat3(transpose(inverse(m_model))) * aNorm;
}