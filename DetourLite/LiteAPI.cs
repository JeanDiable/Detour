using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DetourCore;
using DetourCore.CartDefinition;
using EmbedIO;
using EmbedIO.Routing;
using EmbedIO.WebApi;

namespace DetourLite
{
    public class LiteAPI:WebApiController
    {
        public static byte[] sensorCache = new Byte[1024 * 512]; //512K max

        [Route(HttpVerb.Get, "/getView")]
        public object getView(IHttpContext context)
        {
            var len = 0;
            var pos = CartLocation.latest;
            using (var ms = new MemoryStream(sensorCache))
            using (var bw = new BinaryWriter(ms))
            {
                foreach (var painter in Program.Painters.ToArray())
                    foreach (var bytes in painter.Value.actions.ToArray())
                        bw.Write(bytes);
                len = (int)ms.Position;
            }

            using (var stream = HttpContext.OpenResponseStream())
                stream.Write(sensorCache, 0, len);
            return null;
        }
    }
}
