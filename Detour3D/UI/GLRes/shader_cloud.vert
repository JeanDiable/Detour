#version 330 core
layout (location = 0) in vec3 aPosition;   // the position variable has attribute position 0
layout (location = 1) in vec3 aColor; // the color variable has attribute position 1
  
out vec3 ourColor; // output a color to the fragment shader

uniform mat4 m_model;
uniform mat4 m_view;
uniform mat4 m_projection;

void main()
{
    gl_Position = vec4(aPosition, 1.0) * m_model * m_view * m_projection;
    ourColor = aColor; // set ourColor to the input color we got from the vertex data
}