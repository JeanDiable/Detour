namespace THREE
{
    using System;
    using System.Collections.Generic;

    using ThreeCs.Math;
    using ThreeCs.Renderers.Shaders;
    using ThreeCs.Renderers.WebGL;

    public class BlurShader : WebGlShader
    {
        /// <summary>
        /// Constructor
        /// </summary>
        public BlurShader()
        {
            #region construct uniform variables

            this.Uniforms =
                UniformsUtils.Merge(new List<Uniforms>
                {
                    new Uniforms { { "tDiffuse", new Uniform() { {"type", "t"},  {"value", null } } }},
                    new Uniforms { { "amount",   new Uniform() { {"type", "f"},  {"value", 0.005f } } }},
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

            float unpackDepth( const in vec4 rgba_depth ) {

                const vec4 bit_shift = vec4(0, 1.0 / (256.0 * 256.0), 1.0 / 256.0, 1.0);
                float depth = dot(rgba_depth, bit_shift);
                return depth;

            }
            ");

            fs.Add("void main() {");

            fs.Add("vec2 offset1 = amount * vec2( 0.5, 0);");
            fs.Add("vec2 offset2 = amount * vec2( 0, 0.5);");
            fs.Add("vec4 c = texture2D(tDiffuse, vUv);");
            fs.Add("vec4 c1 = texture2D(tDiffuse, vUv + offset1);");
            fs.Add("vec4 c2 = texture2D(tDiffuse, vUv + offset2);");
            fs.Add("vec4 c3 = texture2D(tDiffuse, vUv - offset1);");
            fs.Add("vec4 c4 = texture2D(tDiffuse, vUv - offset2);");
            //
            // fs.Add("c = c + c1.a*(1-c1.a)*c1*0.25;");
            // fs.Add("c = c + c1.a*(1-c1.a)*c2*0.25;");
            // fs.Add("c = c + c1.a*(1-c1.a)*c3*0.25;");
            // fs.Add("c = c + c1.a*(1-c1.a)*c4*0.25;");
            fs.Add("c = 0.6*c + 0.4*0.25*(c1+c2+c3+c4);");
            // fs.Add("gl_FragColor = vec4(d*20000,0,0,1);");
            fs.Add("gl_FragColor = c;");

            fs.Add("}");

            // join
            this.FragmentShader = String.Join("\n", fs).Trim();
            #endregion

        }
    }
}
