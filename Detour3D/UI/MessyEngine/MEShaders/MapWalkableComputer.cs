using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using Detour3D.UI.ImGuiI;
using OpenTK;
using OpenTK.Graphics.OpenGL;

namespace Fake.UI.MessyEngine.MEShaders
{
    class MapWalkableComputer
    {
        private int _handle;

        private Dictionary<string, int> _uniformLocations;

        private int _boxesDataSSBO;

        private int _frameHeaderSSBO;

        private int _frameDataSSBO;

        private int _outputSSBO;

        private int _computeShader;

        private float[] _output;

        private int _computeWidth;

        private int _computeHeight;

        private int _totalSize;

        //private List<MapHelper.FrameHeader> _inputFrameHeaders;

        // private int[] _inputBoxesData;
        //
        // private List<float> _inputFrameHeader;
        //
        // private List<float> _inputFrameData;

        public bool disabled = false;
        public MapWalkableComputer(Size computeSize)
        {
            UpdateSize(computeSize, 5.0f);

            if (!GenerateComputeShader(1, true))
                return;

            _boxesDataSSBO = GL.GenBuffer();
            _frameHeaderSSBO = GL.GenBuffer();
            _frameDataSSBO = GL.GenBuffer();
            _outputSSBO = GL.GenBuffer();
        }

        //public void UpdateFramesData(List<MapHelper.FrameHeader> header, List<float> data)
        //{
        //    _inputFrameHeaders = header;
        //    _inputFrameData = data;
        //}

        string GetShader(string name)
        {
            return new StreamReader(Assembly.GetExecutingAssembly()
                    .GetManifestResourceStream($@"Detour3D.UI.MERes.GLSL.{name}"))
                .ReadToEnd();
        }

        public void UpdateSize(Size computeSize, float factor)
        {
            _computeWidth = (int)Math.Ceiling(computeSize.Width / factor);
            _computeHeight = (int)Math.Ceiling(computeSize.Height / factor);

            _totalSize = _computeHeight * _computeWidth;
            _output = new float[_totalSize];
        }

        public void UpdateBoxesFrames(int[] inputBoxesData, List<float> inputFrameHeader, List<float> inputFrameData)
        {
            GL.BindBuffer(BufferTarget.ShaderStorageBuffer, _outputSSBO);
            GL.BufferData(BufferTarget.ShaderStorageBuffer, _totalSize * sizeof(float), _output, BufferUsageHint.DynamicDraw);

            GL.BindBuffer(BufferTarget.ShaderStorageBuffer, _boxesDataSSBO);
            GL.BufferData(BufferTarget.ShaderStorageBuffer, inputBoxesData.Length * sizeof(int), inputBoxesData, BufferUsageHint.DynamicDraw);

            GL.BindBuffer(BufferTarget.ShaderStorageBuffer, _frameHeaderSSBO);
            GL.BufferData(BufferTarget.ShaderStorageBuffer, inputFrameHeader.Count * sizeof(float), inputFrameHeader.ToArray(), BufferUsageHint.DynamicDraw);

            GL.BindBuffer(BufferTarget.ShaderStorageBuffer, _frameDataSSBO);
            GL.BufferData(BufferTarget.ShaderStorageBuffer, inputFrameData.Count * sizeof(float), inputFrameData.ToArray(), BufferUsageHint.DynamicDraw);

            GL.BindBufferBase(BufferRangeTarget.ShaderStorageBuffer, 0, _outputSSBO);
            GL.BindBufferBase(BufferRangeTarget.ShaderStorageBuffer, 1, _boxesDataSSBO);
            GL.BindBufferBase(BufferRangeTarget.ShaderStorageBuffer, 2, _frameHeaderSSBO);
            GL.BindBufferBase(BufferRangeTarget.ShaderStorageBuffer, 3, _frameDataSSBO);
        }

        public float[] Compute()
        {
            GL.UseProgram(_handle);

            _output = new float[_totalSize];
            GL.BindBuffer(BufferTarget.ShaderStorageBuffer, _outputSSBO);
            GL.BufferData(BufferTarget.ShaderStorageBuffer, _totalSize * sizeof(float), _output, BufferUsageHint.DynamicDraw);
            GL.DispatchCompute(_computeWidth, _computeHeight, 1);
            GL.MemoryBarrier(MemoryBarrierFlags.ShaderStorageBarrierBit);
            Util.CheckGLError("check compute");
            GL.BindBufferBase(BufferRangeTarget.ShaderStorageBuffer, 0, _outputSSBO);
            var intPtr = GL.MapBuffer(BufferTarget.ShaderStorageBuffer, BufferAccess.ReadOnly);
            Marshal.Copy(intPtr, _output, 0, _totalSize);

            return _output;
        }
        
        public bool GenerateComputeShader(int maxBoxFrameCount, bool init)
        {
            var shaderSource = GetShader("compute-walkable.glsl");
            shaderSource = $@"#version 430

#define MAX_BOX_FRAME_COUNT {maxBoxFrameCount}" + shaderSource;
            var lengths = new List<int>();
            if (init) _computeShader = GL.CreateShader(ShaderType.ComputeShader);
            GL.ShaderSource(_computeShader, shaderSource);

            // compile shader
            GL.CompileShader(_computeShader);
            GL.GetShader(_computeShader, ShaderParameter.CompileStatus, out var c);
            if (c != (int)All.True)
            {
                var infoLog = GL.GetShaderInfoLog(_computeShader);
                Console.WriteLine($"Error occurred whilst compiling Shader({_computeShader}).\n\n{infoLog}");
                disabled = true;
                return false;
            }

            // create program
            // if (del) GL.DeleteProgram(_handle);
            if (init) _handle = GL.CreateProgram();
            if (init) GL.AttachShader(_handle, _computeShader);

            // link program
            GL.LinkProgram(_handle);
            GL.GetProgram(_handle, GetProgramParameterName.LinkStatus, out var cc);
            if (cc != (int)All.True)
            {
                throw new Exception($"Error occurred whilst linking Program({_handle})");
            }

            // release shader
            // GL.DetachShader(_handle, _computeShader);
            // GL.DeleteShader(_computeShader);

            // get uniforms
            GL.GetProgram(_handle, GetProgramParameterName.ActiveUniforms, out var numberOfUniforms);
            _uniformLocations = new Dictionary<string, int>();
            for (var i = 0; i < numberOfUniforms; i++)
            {
                var key = GL.GetActiveUniform(_handle, i, out _, out _);
                var location = GL.GetUniformLocation(_handle, key);
                _uniformLocations.Add(key, location);
            }

            return true;
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
