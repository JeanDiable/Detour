using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Fake.Components;
using OpenTK;
using OpenTK.Graphics.OpenGL;

namespace Fake.UI.MessyEngine.MEShaders
{
    public class ObsoleteWalkableComputer
    {
        string GetShader(string name)
        {
            return new StreamReader(Assembly.GetExecutingAssembly()
                    .GetManifestResourceStream($@"Fake.UI.MERes.GLSL.{name}"))
                .ReadToEnd();
        }

        private int _handle;

        private Dictionary<string, int> _uniformLocations;

        private int _inputHeaderSSBO;

        private int _inputDataSSBO;

        private int _outputSSBO;

        private int _computeShader;

        private float[] _output;

        private int _computeWidth;

        private int _computeHeight;

        private int _stride;

        private int _totalSize;

        private int _frameNum;

        private List<ObsoleteMapHelper.FrameHeader> _inputFrameHeaders;

        private List<Float5> _inputFrameData;

        public ObsoleteWalkableComputer(Size computeSize, int stride)
        {
            UpdateSize(computeSize, stride);

            GenerateComputeShader();

            _outputSSBO = GL.GenBuffer();
            //GL.BindBuffer(BufferTarget.ShaderStorageBuffer, _outputSSBO);

            _inputHeaderSSBO = GL.GenBuffer();
            //GL.BindBuffer(BufferTarget.ShaderStorageBuffer, _outputSSBO);

            _inputDataSSBO = GL.GenBuffer();
        }

        public void UpdateFramesData(List<ObsoleteMapHelper.FrameHeader> header, List<Float5> data, int frameNum)
        {
            _inputFrameHeaders = header;
            _inputFrameData = data;
            _frameNum = frameNum;
        }

        public unsafe float[] Compute()
        {   
            GL.UseProgram(_handle);

            GL.BindBuffer(BufferTarget.ShaderStorageBuffer, _outputSSBO);
            GL.BufferData(BufferTarget.ShaderStorageBuffer, _totalSize * sizeof(float), _output, BufferUsageHint.DynamicDraw);
            GL.BindBufferBase(BufferRangeTarget.ShaderStorageBuffer, 0, _outputSSBO);
            //GL.BindBuffer(BufferTarget.ShaderStorageBuffer, 0); // unbind

            GL.BindBuffer(BufferTarget.ShaderStorageBuffer, _inputHeaderSSBO);
            GL.BufferData(BufferTarget.ShaderStorageBuffer, _frameNum * sizeof(ObsoleteMapHelper.FrameHeader), _inputFrameHeaders.ToArray(), BufferUsageHint.DynamicDraw);
            GL.BindBufferBase(BufferRangeTarget.ShaderStorageBuffer, 1, _inputHeaderSSBO);

            GL.BindBuffer(BufferTarget.ShaderStorageBuffer, _inputDataSSBO);
            GL.BufferData(BufferTarget.ShaderStorageBuffer, _inputFrameData.Count * sizeof(Float5), _inputFrameData.ToArray(), BufferUsageHint.DynamicDraw);
            GL.BindBufferBase(BufferRangeTarget.ShaderStorageBuffer, 2, _inputDataSSBO);

            _output = new float[_totalSize];
            GL.DispatchCompute(_computeWidth, _computeHeight, 1);
            GL.MemoryBarrier(MemoryBarrierFlags.ShaderStorageBarrierBit);

            GL.BindBufferBase(BufferRangeTarget.ShaderStorageBuffer, 0, _outputSSBO);
            var intPtr = GL.MapBuffer(BufferTarget.ShaderStorageBuffer, BufferAccess.ReadOnly);
            Marshal.Copy(intPtr, _output, 0, _totalSize);
            //Console.WriteLine($"{_computeWidth}, {_computeHeight}, {_totalSize}, {_output.Length}");

            return _output;
        }

        public void GenerateComputeShader(bool del = false)
        {
            var shaderSource = GetShader("compute-walkable.glsl");
//            shaderSource = $@"#version 430

//#define WIDTH {_computeWidth}
//#define HEIGHT {_computeHeight}
//#define FRAME_NUM {_frameNum}" + shaderSource;
            var lengths = new List<int>();
            _computeShader = GL.CreateShader(ShaderType.ComputeShader);
            GL.ShaderSource(_computeShader, shaderSource);

            // compile shader
            GL.CompileShader(_computeShader);
            GL.GetShader(_computeShader, ShaderParameter.CompileStatus, out var c);
            if (c != (int) All.True)
            {
                var infoLog = GL.GetShaderInfoLog(_computeShader);
                throw new Exception($"Error occurred whilst compiling Shader({_computeShader}).\n\n{infoLog}");
            }

            // create program
            if (del) GL.DeleteProgram(_handle);
            _handle = GL.CreateProgram();
            GL.AttachShader(_handle, _computeShader);

            // link program
            GL.LinkProgram(_handle);
            GL.GetProgram(_handle, GetProgramParameterName.LinkStatus, out var cc);
            if (cc != (int)All.True)
            {
                throw new Exception($"Error occurred whilst linking Program({_handle})");
            }

            // release shader
            GL.DetachShader(_handle, _computeShader);
            GL.DeleteShader(_computeShader);

            // get uniforms
            GL.GetProgram(_handle, GetProgramParameterName.ActiveUniforms, out var numberOfUniforms);
            _uniformLocations = new Dictionary<string, int>();
            for (var i = 0; i < numberOfUniforms; i++)
            {
                var key = GL.GetActiveUniform(_handle, i, out _, out _);
                var location = GL.GetUniformLocation(_handle, key);
                _uniformLocations.Add(key, location);
            }
        }

        public void UpdateSize(Size computeSize, int stride)
        {
            _computeWidth = computeSize.Width;
            _computeHeight = computeSize.Height;
            _stride = stride;
            _totalSize = _computeHeight * _computeWidth;
        }

        private void SetInt(string name, int data)
        {
            GL.UseProgram(_handle);
            GL.Uniform1(_uniformLocations[name], data);
        }

        private void SetFloat(string name, float data)
        {
            GL.UseProgram(_handle);
            GL.Uniform1(_uniformLocations[name], data);
        }

        private void SetMatrix4(string name, Matrix4 data)
        {
            GL.UseProgram(_handle);
            GL.UniformMatrix4(_uniformLocations[name], true, ref data);
        }

        private void SetVector3(string name, Vector3 data)
        {
            GL.UseProgram(_handle);
            GL.Uniform3(_uniformLocations[name], data);
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
