namespace THREE
{
    using System;
    using System.Collections.Generic;

    using ThreeCs.Math;
    using ThreeCs.Renderers.Shaders;
    using ThreeCs.Renderers.WebGL;

    public class DepthSobelShader : WebGlShader
    {
        /// <summary>
        /// Constructor
        /// </summary>
        public DepthSobelShader()
        {
            #region construct uniform variables

            this.Uniforms =
                UniformsUtils.Merge(new List<Uniforms>
                {
                    new Uniforms { { "tDiffuse", new Uniform() { {"type", "t"},  {"value", null } } }},
                    new Uniforms { { "amount",   new Uniform() { {"type", "f"},  {"value", 0.0005f } } }},
                    new Uniforms { { "angle",    new Uniform() { {"type", "f"},  {"value", 1.0f } }}}
                });
            #endregion

            #region construct VertexShader
            var vs = new List<string>();

            vs.Add("varying vec2 vUv;");

            vs.Add("void main() {");

            vs.Add("vUv = uv;");
            vs.Add("gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );");

            vs.Add("}");

            // join
            this.VertexShader = String.Join("\n", vs).Trim();
            #endregion

            #region construct FragmentShader
            var fs = new List<string>();

            fs.Add("uniform sampler2D tDiffuse;");
            fs.Add("uniform float amount;");
            fs.Add("uniform float angle;");

            fs.Add("varying vec2 vUv;");
            fs.Add(@"

const float UnpackDownscale = 255. / 256.; // 0..1 -> fraction (excluding 1)
const vec3 PackFactors = vec3( 256. * 256. * 256., 256. * 256., 256. );
const vec4 UnpackFactors = UnpackDownscale / vec4( PackFactors, 1. );
                float unpackDepth( const in vec4 v ) {
	                return dot( v, UnpackFactors );
                }
            ");

            fs.Add("void main() {");

            fs.Add("vec2 offset1 = amount * vec2( 0.5, 0);");
            fs.Add("vec2 offset2 = amount * vec2( 0, 0.5);");
            fs.Add("vec4 c = texture2DLod(tDiffuse, vUv, 0);");
            fs.Add("vec4 c1 = texture2DLod(tDiffuse, vUv + offset1,0);");
            fs.Add("vec4 c2 = texture2DLod(tDiffuse, vUv + offset2,0);");
            fs.Add("vec4 c3 = texture2DLod(tDiffuse, vUv - offset1,0);");
            fs.Add("vec4 c4 = texture2DLod(tDiffuse, vUv - offset2,0);");
            //
            fs.Add("float fd = unpackDepth( c );");
            fs.Add("float fdx1 = unpackDepth( c1 ); if (fdx1==0) fdx1=fd;");
            fs.Add("float fdy1 = unpackDepth( c2 ); if (fdy1==0) fdy1=fd;");
            fs.Add("float fdx2 = unpackDepth( c3 ); if (fdx2==0) fdx2=fd;");
            fs.Add("float fdy2 = unpackDepth( c4 ); if (fdy2==0) fdy2=fd;");
            //
            fs.Add("float d1 = max(abs(fdx1-fd), abs(fd-fdx2))*30000;");
            fs.Add("float d2 = max(abs(fdy1-fd), abs(fd-fdy2))*30000;");
            fs.Add("float d = sqrt(d1*d1+d2*d2);");

            // fs.Add("gl_FragColor = c;");

            // fs.Add("gl_FragColor = vec4(fd*50000,0,0,1);");

            fs.Add("gl_FragColor = vec4(0,0,0,clamp(d,0,1));");

            fs.Add("}");

            // join
            this.FragmentShader = String.Join("\n", fs).Trim();
            #endregion

        }
    }
}
