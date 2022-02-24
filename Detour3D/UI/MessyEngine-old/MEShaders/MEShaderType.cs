using System.Collections.Generic;
using Detour3D.UI.MessyEngine.MEMeshes;
using OpenTK.Graphics.OpenGL;

namespace Detour3D.UI.MessyEngine.MEShaders
{
    public class MEShaderType
    {
        public string name;
        public List<int> layoutIndices;
        public List<int> layoutSizes;
        public List<VertexAttribPointerType> layoutPointerTypes;
        public List<bool> layoutNormalized;
        public List<int> layoutOffsets;

        public bool isUseGeom;

        //public static readonly MEShaderType GenericCircle = new MEShaderType(
        //    "generic-circle",
        //    new List<int>() { 3 }, 
        //    new List<int>() { 3 },
        //    new List<VertexAttribPointerType>() { VertexAttribPointerType.Float },
        //    new List<bool>() { false }, 
        //    new List<int>() { 0 }, 
        //    false);

        //public static readonly MEShaderType GenericLine = new MEShaderType(
        //    "generic-line",
        //    new List<int>() { 4, 5 }, 
        //    new List<int>() { 3, 3 },
        //    new List<VertexAttribPointerType>() { VertexAttribPointerType.Float, VertexAttribPointerType.Float },
        //    new List<bool>() { false, false },
        //    new List<int>() { 0, 6 * sizeof(float) },
        //    false);
        
        //public static readonly MEShaderType GenericMesh = new MEShaderType(
        //    "generic-mesh",
        //    new List<int>() { 6, 7 }, 
        //    new List<int>() { 3, 3 },
        //    new List<VertexAttribPointerType>() { VertexAttribPointerType.Float, VertexAttribPointerType.Float },
        //    new List<bool>() { false, false },
        //    new List<int>() { 0, 3 * sizeof(float) },
        //    false);
        
        public static readonly MEShaderType GenericPoint = new MEShaderType(
            "generic-point",
            new List<int>() { 8, 9 }, 
            new List<int>() { VertexSize.PositionSize, VertexSize.ColorSize },
            new List<VertexAttribPointerType>() { VertexAttribPointerType.Float, VertexAttribPointerType.Float },
            new List<bool>() { false, false },
            new List<int>() { VertexOffset.PositionOffset, VertexOffset.ColorOffset },
            false);

        public static readonly MEShaderType SpecificGrid = new MEShaderType(
            "specific-grid",
            new List<int>() { 10, 11 }, 
            new List<int>() { VertexSize.PositionSize, VertexSize.ColorSize }, 
            new List<VertexAttribPointerType>() { VertexAttribPointerType.Float, VertexAttribPointerType.Float },
            new List<bool>() { false, false },
            new List<int>() { VertexOffset.PositionOffset, VertexOffset.ColorOffset },
            true);

        public static readonly MEShaderType GenericTriangle = new MEShaderType(
            "generic-point",
            new List<int>() { 8, 9 },
            new List<int>() { VertexSize.PositionSize, VertexSize.ColorSize },
            new List<VertexAttribPointerType>() { VertexAttribPointerType.Float, VertexAttribPointerType.Float },
            new List<bool>() { false, false },
            new List<int>() { VertexOffset.PositionOffset, VertexOffset.ColorOffset },
            false);

        public static readonly MEShaderType SpecificSearch = new MEShaderType(
            "specific-search",
            new List<int>() { 4, 5 },
            new List<int>() { VertexSize.PositionSize, VertexSize.ColorSize },
            new List<VertexAttribPointerType>() { VertexAttribPointerType.Float, VertexAttribPointerType.Float },
            new List<bool>() { false, false },
            new List<int>() { VertexOffset.PositionOffset, VertexOffset.ColorOffset },
            true);

        public static readonly MEShaderType GenericTexture = new MEShaderType(
            "generic-texture",
            new List<int>() { 6, 7 },
            new List<int>() { VertexSize.PositionSize},//, VertexSize.TexCoordsSize },
            new List<VertexAttribPointerType>() { VertexAttribPointerType.Float, VertexAttribPointerType.Float },
            new List<bool>() { false, false },
            new List<int>() { VertexOffset.PositionOffset},// VertexOffset.TexCoordsOffset },
            false);

        public MEShaderType(string name, List<int> layoutIndices, List<int> layoutSizes, 
            List<VertexAttribPointerType> layoutPointerTypes, List<bool> layoutNormalized, 
            List<int> layoutOffsets, bool isUseGeom)
        {
            this.name = name;
            this.layoutIndices = layoutIndices;
            this.layoutSizes = layoutSizes;
            this.layoutPointerTypes = layoutPointerTypes;
            this.layoutNormalized = layoutNormalized;
            this.layoutOffsets = layoutOffsets;

            this.isUseGeom = isUseGeom;
        }
    }
}
