using System.Collections.Generic;
using Detour3D.UI.MessyEngine.MEMeshes;

namespace Detour3D.UI.MessyEngine.MEObjects
{
    public interface IMEObjectInterface
    {
        // data and display
        void UpdateMeshData(List<Vertex> verticesList = null, List<uint> indicesList = null);
        void UpdateUniforms(Dictionary<string, dynamic> dict = null);
        void Draw();

        // motion and physics
    }
}
