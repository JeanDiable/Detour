#version 330 core
layout (location = 8) in vec3 aPosition;   // the position variable has attribute position 0
layout (location = 9) in float aColor; // the color variable has attribute position 1

out vec4 ourColor; // output a color to the fragment shader

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

uniform float useIntensityColor;
uniform float pointSize;
uniform vec4 assignColor;

void main()
{
    const int step1 = 16;
    const int step2 = 60;
    const int step3 = 60;
    const int step4 = 120;
    gl_Position = vec4(aPosition, 1.0) * modelMatrix * viewMatrix * projectionMatrix;
    gl_PointSize = pointSize;
    if (assignColor.w > 0) ourColor = assignColor;
    else if (useIntensityColor > 0) {
        if (aColor < step1) ourColor = vec4(1, aColor / step1, 0, 1); 
        else if (aColor < step1 + step2) ourColor = vec4((step2 - aColor) / step2, 1, 0, 1);
        else if (aColor < step1 + step2 + step3) ourColor = vec4(0, 1, aColor / step3, 1);
        else ourColor = vec4(0, (step4 - aColor) / step4, 1, 1);
    }
    else ourColor = vec4(1, 1, 1, 1); // set ourColor to the input color we got from the vertex data
}