#version 330 core

out vec4 FragColor;  
in float Alpha;
  
void main()
{
    FragColor = vec4(0.096, 0.504, 0.096, Alpha / 2);
}