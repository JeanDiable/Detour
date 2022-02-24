#version 330 core

in float fAlpha;
out vec4 FragColor;
  
void main()
{
//0x8a/256.0f,0x2b/256.0f,0xe2/256.0f
    FragColor = vec4(0x8a/256.0,0x2b/256.0,0xe2/256.0, fAlpha);
}