using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using OpenTK;
using OpenTK.Graphics.OpenGL;

namespace Detour3D.UI.MessyEngine.MEShaders
{
    // A simple class meant to help create shaders.
    public class MEShader
    {
        string GetShader(string name)
        {
            return new StreamReader(Assembly.GetExecutingAssembly()
                    .GetManifestResourceStream($@"Fake.UI.MERes.GLSL.{name}"))
                .ReadToEnd();
        }
        public readonly int Handle;

        private readonly Dictionary<string, int> _uniformLocations;

        public MEShader(MEShaderType shaderType)
        {
            var vertName = shaderType.name + ".vert";
            var geomName = shaderType.name + ".geom";
            var fragName = shaderType.name + ".frag";

            // vertex shader
            var vertShaderSource = GetShader(vertName);
            var vertexShader = GL.CreateShader(ShaderType.VertexShader);
            GL.ShaderSource(vertexShader, vertShaderSource);
            CompileShader(vertexShader);

            // geometry shader
            var geometryShader = 0;
            if (shaderType.isUseGeom)
            {
                var geomShaderSource = GetShader(geomName);
                geometryShader = GL.CreateShader(ShaderType.GeometryShader);
                GL.ShaderSource(geometryShader, geomShaderSource);
                CompileShader(geometryShader);
            }

            // fragment shader
            var fragShaderSource = GetShader(fragName);
            var fragmentShader = GL.CreateShader(ShaderType.FragmentShader);
            GL.ShaderSource(fragmentShader, fragShaderSource);
            CompileShader(fragmentShader);

            Handle = GL.CreateProgram();

            GL.AttachShader(Handle, vertexShader);
            if (shaderType.isUseGeom) GL.AttachShader(Handle, geometryShader);
            GL.AttachShader(Handle, fragmentShader);

            LinkProgram(Handle);

            GL.DetachShader(Handle, vertexShader);
            if (shaderType.isUseGeom) GL.DetachShader(Handle, geometryShader);
            GL.DetachShader(Handle, fragmentShader);

            GL.DeleteShader(fragmentShader);
            if (shaderType.isUseGeom) GL.DeleteShader(geometryShader);
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

        public MEShader(string vertName, string fragName, string geomName = "")
        {
            // vertex shader
            var vertShaderSource = GetShader(vertName); //File.ReadAllText("res/" + vertName);
            var vertexShader = GL.CreateShader(ShaderType.VertexShader);
            GL.ShaderSource(vertexShader, vertShaderSource);
            CompileShader(vertexShader);

            // geometry shader
            var geometryShader = 0;
            if (geomName != "")
            {
                var geomShaderSource = GetShader(geomName);
                geometryShader = GL.CreateShader(ShaderType.GeometryShader);
                GL.ShaderSource(geometryShader, geomShaderSource);
                CompileShader(geometryShader);
            }

            // fragment shader
            var fragShaderSource = GetShader(fragName);
            var fragmentShader = GL.CreateShader(ShaderType.FragmentShader);
            GL.ShaderSource(fragmentShader, fragShaderSource);
            CompileShader(fragmentShader);

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

        private int GetAttribLocation(string attribName)
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

        public void SetVector4(string name, Vector4 data)
        {
            GL.UseProgram(Handle);
            GL.Uniform4(_uniformLocations[name], data);
        }

        public void SetUniforms(Dictionary<string, dynamic> parameters)
        {
            foreach (var parameter in parameters)
            {
                if (parameter.Value is int) SetInt(parameter.Key, parameter.Value);
                else if (parameter.Value is float) SetFloat(parameter.Key, parameter.Value);
                else if (parameter.Value is Matrix4) SetMatrix4(parameter.Key, parameter.Value);
                else if (parameter.Value is Vector3) SetVector3(parameter.Key, parameter.Value);
            }
        }
    }
}
