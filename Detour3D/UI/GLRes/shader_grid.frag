#version 330 core

in float fAlpha;
out vec4 FragColor;
  
void main()
{
    FragColor = vec4(1.0, 1.0, 1.0, fAlpha);
}