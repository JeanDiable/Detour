#version 330 core

out vec4 FragColor;  
in float Alpha;
  
void main()
{
    FragColor = vec4(0.196, 0.804, 0.196, Alpha / 2);
}