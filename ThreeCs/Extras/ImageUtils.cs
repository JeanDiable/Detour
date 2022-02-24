using System.IO;

namespace ThreeCs.Extras
{
    using System.Drawing;

    using ThreeCs.Loaders;
    using ThreeCs.Textures;

    public class ImageUtils
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="url"></param>
        /// <param name="mapping"></param>
        /// <returns></returns>
        public static Texture LoadTexture(string url, TextureMapping mapping = null)
        {
            var image = (Bitmap)Image.FromFile(url, true);

            image.RotateFlip(RotateFlipType.Rotate180FlipX);

            return new Texture(image, mapping) { NeedsUpdate = true, SourceFile = url, Format = ImageLoader.PixelFormatToThree(image.PixelFormat) };
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="url"></param>
        /// <param name="mapping"></param>
        /// <returns></returns>
        public static Texture LoadTextureFromAssets(string name, Stream stream, TextureMapping mapping = null)
        {
            var image = (Bitmap)Image.FromStream(stream, true);

            image.RotateFlip(RotateFlipType.Rotate180FlipX);

            return new Texture(image, mapping) { NeedsUpdate = true, SourceFile = name, Format = ImageLoader.PixelFormatToThree(image.PixelFormat) };
        }
        
    }
}

