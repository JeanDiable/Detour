using ThreeCs.Core;

namespace ThreeCs.Renderers.Shaders
{
    using System.Collections.Generic;

    public class Attributes : Dictionary<string, Attribute>
    {
        private BufferAttribute<uint> _index;
        private BufferAttribute<float> _position;
        private BufferAttribute<float> _normal;
        private BufferAttribute<float> _uv;

        public BufferAttribute<uint> Index
        {
            get => ContainsKey("index") ? this["index"] as BufferAttribute<uint> : null;
            set => this["index"] = value;
        }

        public BufferAttribute<float> Position
        {
            get => ContainsKey("position") ? this["position"] as BufferAttribute<float> : null;
            set => this["position"] = value;
        }

        public BufferAttribute<float> Normal
        {
            get => ContainsKey("normal") ? this["normal"] as BufferAttribute<float> : null;
            set => this["normal"] = value;
        }

        public BufferAttribute<float> UV
        {
            get => ContainsKey("uv") ? this["uv"] as BufferAttribute<float> : null;
            set => this["uv"] = value;
        }
    }
}
