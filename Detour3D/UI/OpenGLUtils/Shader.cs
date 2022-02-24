using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using OpenTK;
using OpenTK.Graphics.OpenGL;

namespace Fake.UI.OpenGLUtils
{
    // A simple class meant to help create shaders.
    public class Shader
    {
        string GetShader(string name)
        {
            return new StreamReader(Assembly.GetExecutingAssembly()
                    .GetManifestResourceStream($@"Fake.UI.GLRes.{name}"))
                .ReadToEnd();
        }
        public readonly int Handle;

        private readonly Dictionary<string, int> _uniformLocations;

        public Shader(string vertName, string fragName, string geomName = "")
        {
            //var assembly = Assembly.GetExecutingAssembly();

            // vertex shader
            var vertShaderSource = GetShader(vertName); //File.ReadAllText("res/" + vertName);
            var vertexShader = GL.CreateShader(ShaderType.VertexShader);
            GL.ShaderSource(vertexShader, vertShaderSource);
            CompileShader(vertexShader);
            //var vertResourcePath = assembly.GetManifestResourceNames()
            //    .Single(str => str.EndsWith(vertName));
            //var vertResourceStream = assembly.GetManifestResourceStream(vertResourcePath);
            //var vertexShader = GL.CreateShader(ShaderType.VertexShader);
            //using (StreamReader reader = new StreamReader(vertResourceStream))
            //{
            //    string shaderSource = reader.ReadToEnd();
            //    GL.ShaderSource(vertexShader, shaderSource);
            //    CompileShader(vertexShader);
            //}
              
            // geometry shader
            var geometryShader = 0;
            if (geomName != "")
            {
                var geomShaderSource = GetShader(geomName);//File.ReadAllText("res/" + geomName));
                geometryShader = GL.CreateShader(ShaderType.GeometryShader);
                GL.ShaderSource(geometryShader, geomShaderSource);
                CompileShader(geometryShader);
            }

            // fragment shader
            var fragShaderSource = GetShader(fragName);//File.ReadAllText("res/" + fragName));
            var fragmentShader = GL.CreateShader(ShaderType.FragmentShader);
            GL.ShaderSource(fragmentShader, fragShaderSource);
            CompileShader(fragmentShader);
            ////var fragResourcePath = "LidarController.OpenGLUtils.res." + fragName;
            //var fragResourcePath = assembly.GetManifestResourceNames()
            //    .Single(str => str.EndsWith(fragName));
            //var fragResourceStream = assembly.GetManifestResourceStream(fragResourcePath);
            //var fragmentShader = GL.CreateShader(ShaderType.FragmentShader);
            //using (StreamReader reader = new StreamReader(fragResourceStream))
            //{
            //    string shaderSource = reader.ReadToEnd();
            //    GL.ShaderSource(fragmentShader, shaderSource);
            //    CompileShader(fragmentShader);
            //}

            Handle = GL.CreateProgram();

            GL.AttachShader(Handle, vertexShader);
            if (geomName != "") GL.AttachShader(Handle, geometryShader);
            GL.AttachShader(Handle, fragmentShader);

            LinkProgram(Handle);

            GL.DetachShader(Handle, vertexShader);
            if (geomName != "") GL.DetachShader(Handle, geometryShader);
            GL.DetachShader(Handle, fragmentShader);

            GL.DeleteShader(fragmentShader);
            if (geomName != "") GL.DeleteShader(geometryShader);
            GL.DeleteShader(vertexShader);

            GL.GetProgram(Handle, GetProgramParameterName.ActiveUniforms, out var numberOfUniforms);

            _uniformLocations = new Dictionary<string, int>();

            for (var i = 0; i < numberOfUniforms; i++)
            {
                var key = GL.GetActiveUniform(Handle, i, out _, out _);
                var location = GL.GetUniformLocation(Handle, key);
                _uniformLocations.Add(key, location);
            }
        }

        private static void CompileShader(int shader)
        {
            GL.CompileShader(shader);
            GL.GetShader(shader, ShaderParameter.CompileStatus, out var code);
            if (code != (int)All.True)
            {
                var infoLog = GL.GetShaderInfoLog(shader);
                throw new Exception($"Error occurred whilst compiling Shader({shader}).\n\n{infoLog}");
            }
        }

        private static void LinkProgram(int program)
        {
            GL.LinkProgram(program);
            GL.GetProgram(program, GetProgramParameterName.LinkStatus, out var code);
            if (code != (int)All.True)
            {
                throw new Exception($"Error occurred whilst linking Program({program})");
            }
        }

        public void Use()
        {
            GL.UseProgram(Handle);
        }

        public int GetAttribLocation(string attribName)
        {
            return GL.GetAttribLocation(Handle, attribName);
        }

        public void SetInt(string name, int data)
        {
            GL.UseProgram(Handle);
            GL.Uniform1(_uniformLocations[name], data);
        }

        public void SetFloat(string name, float data)
        {
            GL.UseProgram(Handle);
            GL.Uniform1(_uniformLocations[name], data);
        }

        public void SetMatrix4(string name, Matrix4 data)
        {
            GL.UseProgram(Handle);
            GL.UniformMatrix4(_uniformLocations[name], true, ref data);
        }

        public void SetVector3(string name, Vector3 data)
        {
            GL.UseProgram(Handle);
            GL.Uniform3(_uniformLocations[name], data);
        }
    }
}
