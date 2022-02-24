using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Fake.UI.MessyEngine.MEMeshes;
using OpenTK;

namespace Fake.UI.MessyEngine.MEObjects
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
