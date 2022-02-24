#version 330 core
out vec4 FragColor;  
in vec4 ourColor;
  
void main()
{
//    if (ourColor.x > 0.5) FragColor = vec4(1, 0, 0, 1);
//    else FragColor = vec4(0, 1, 1, 1.0);
	FragColor = ourColor;
}