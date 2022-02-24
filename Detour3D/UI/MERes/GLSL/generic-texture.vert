//#version 330 core
//layout (location = 6) in vec3 aPosition;
//layout (location = 7) in vec3 aTexCoord;
//
//out vec3 texCoord;
//
//uniform mat4 modelMatrix;
//uniform mat4 viewMatrix;
//uniform mat4 projectionMatrix;
//
//void main()
//{
//    gl_Position = vec4(aPosition, 1.0) * modelMatrix * viewMatrix * projectionMatrix;
//    texCoord = aTexCoord;
//}

#version 330 core
layout (location = 6) in vec3 aPosition;
layout (location = 7) in vec2 aTexCoord;

out vec2 texCoord;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

void main()
{
    gl_Position = vec4(aPosition, 1.0) * modelMatrix * viewMatrix * projectionMatrix;
    texCoord = aTexCoord;
}