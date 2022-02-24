#version 330 core
layout (lines) in;
layout (line_strip, max_vertices = 4) out;

in VS_OUT {
    float maxAlpha;
} gs_in[];

uniform mat4 m_model;
uniform mat4 m_view;
uniform mat4 m_projection;

uniform vec3 center;
uniform float innerRadius;
uniform float outerRadius;

out float fAlpha;

float LerpFloat(float lValue, float rValue, float ll, float rr, float pos)
{
    float percent = (abs(pos) - ll) / (rr - ll);
    return mix(lValue, rValue, percent);
}

void main()
{
    float maxAlpha = gs_in[0].maxAlpha;

    vec4 pos0 = gl_in[0].gl_Position; // on positive axis
    vec4 pos1 = gl_in[1].gl_Position; // on negative axis
    vec4 posMiddle = (pos0 + pos1) / 2;

    mat4 transf = m_model * m_view * m_projection;

    // parallel to x-axis
    if (pos0.z == pos1.z)
    {
        float zOffset = abs(posMiddle.z - center.z);
        float theta = acos(zOffset / innerRadius);

        if (zOffset <= innerRadius)
        {
            vec4 subPos0 = vec4(zOffset * tan(theta) + center.x, 0, posMiddle.z, 1.0);
            vec4 subPos1 = vec4(-zOffset * tan(theta) + center.x, 0, posMiddle.z, 1.0);
            if (posMiddle.z == center.z)
            {
                subPos0 = vec4(innerRadius + center.x, 0, center.z, 1.0);
                subPos1 = vec4(-innerRadius + center.x, 0, center.z, 1.0);
            }

            fAlpha = 0;
            gl_Position = pos0 * transf;
            EmitVertex();
            fAlpha = maxAlpha;
            gl_Position = subPos0 * transf;
            EmitVertex();
            //fAlpha = maxAlpha;
            gl_Position = subPos1 * transf;
            EmitVertex();
            fAlpha = 0;
            gl_Position = pos1 * transf;
            EmitVertex();

            EndPrimitive();
        }
        else
        {
            vec4 subPos = vec4(center.x, 0, posMiddle.z, 1.0);

            fAlpha = 0;
            gl_Position = pos0 * transf;
            EmitVertex();
            fAlpha = LerpFloat(maxAlpha, 0, innerRadius, outerRadius, abs(posMiddle.z - center.z));
            gl_Position = subPos * transf;
            EmitVertex();
            fAlpha = 0;
            gl_Position = pos1 * transf;
            EmitVertex();

            EndPrimitive();
        }
    }
    // parallel to z-axis
    else
    {
        float xOffset = abs(posMiddle.x - center.x);
        float theta = acos(xOffset / innerRadius);

        if (xOffset <= innerRadius)
        {
            vec4 subPos0 = vec4(posMiddle.x, 0, xOffset * tan(theta) + center.z, 1.0);
            vec4 subPos1 = vec4(posMiddle.x, 0, -xOffset * tan(theta) + center.z, 1.0);
            if (posMiddle.x == center.x)
            {
                subPos0 = vec4(center.x, 0, innerRadius + center.z, 1.0);
                subPos1 = vec4(center.x, 0, -innerRadius + center.z, 1.0);
            }

            fAlpha = 0;
            gl_Position = pos0 * transf;
            EmitVertex();
            fAlpha = maxAlpha;
            gl_Position = subPos0 * transf;
            EmitVertex();
            fAlpha = maxAlpha;
            gl_Position = subPos1 * transf;
            EmitVertex();
            fAlpha = 0;
            gl_Position = pos1 * transf;
            EmitVertex();

            EndPrimitive();
        }
        else
        {
            vec4 subPos = vec4(posMiddle.x, 0, center.z, 1.0);

            fAlpha = 0;
            gl_Position = pos0 * transf;
            EmitVertex();
            fAlpha = LerpFloat(maxAlpha, 0, innerRadius, outerRadius, abs(posMiddle.x - center.x));
            gl_Position = subPos * transf;
            EmitVertex();
            fAlpha = 0;
            gl_Position = pos1 * transf;
            EmitVertex();

            EndPrimitive();
        }
    }
}