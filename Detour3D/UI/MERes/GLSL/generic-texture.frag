//#version 330 core
//
//out vec4 FragColor;  
//in vec3 texCoord;
//
//uniform sampler2DArray texture0;
//  
//void main()
//{
//	FragColor = texture(texture0, texCoord.xyz);
//}

//#version 330 core
//
//out vec4 FragColor;  
//in vec2 texCoord;
//
//uniform sampler2D texture0;
//  
//void main()
//{
//	FragColor = texture(texture0, texCoord);
//}

#version 330 core

out vec4 FragColor;
in vec2 texCoord;

uniform sampler2D texture0;
  
void main()
{
	FragColor = texture(texture0, texCoord);
}