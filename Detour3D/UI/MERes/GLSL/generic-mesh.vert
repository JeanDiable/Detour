#version 330 core
layout (location = 6) in vec3 aPos;
layout (location = 7) in vec3 aNorm;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

void main()
{
    gl_Position = vec4(aPos, 1.0) * modelMatrix * viewMatrix * projectionMatrix;

    FragPos = vec3(vec4(aPos, 1.0) * modelMatrix);
    Normal = mat3(transpose(inverse(modelMatrix))) * aNorm;
}