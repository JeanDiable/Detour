using ThreeCs.Cameras;
using ThreeCs.Core;
using ThreeCs.Renderers;

namespace ThreeCs.Lights
{
    using System.Drawing;

    using ThreeCs.Math;
    using ThreeCs.Textures;

    public interface ILightShadow
    {
        Object3D target { get; set; }

        bool onlyShadow { get; set; }

        float shadowCameraNear { get; set; }

        float shadowCameraFar { get; set; }

        float shadowCameraFov { get; set; }

        bool shadowCameraVisible { get; set; }

        float shadowBias { get; set; }

        float shadowDarkness { get; set; }

        float shadowMapWidth { get; set; }

        float shadowMapHeight { get; set; }

        WebGLRenderTarget shadowMap { get; set; }

        Size shadowMapSize { get; set; }

        Camera shadowCamera { get; set; }

        Matrix4 shadowMatrix { get; set; }
    }
}
