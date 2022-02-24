using System;
using System.Diagnostics;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using DetourCore.Debug;

namespace DetourCore.ExternalComm
{
    class TCPInterface
    {
        public static void init()
        {
            try
            {
                var socket = new Socket(AddressFamily.InterNetwork,
                    SocketType.Stream, ProtocolType.Tcp);
                socket.Bind(new IPEndPoint(IPAddress.Any, 4333));
                socket.Listen(2);
                Stopwatch sw = new Stopwatch();
                sw.Start();
                new Thread(() =>
                {
                    D.Log($"[Position TCP] listen on 4333");

                    while (true)
                    {
                        var handler = socket.Accept();
                        D.Log($"[Position TCP] incoming connection");

                        new Thread(() =>
                        {
                            try
                            {
                                while (true)
                                {
                                    lock (CartLocation.notify)
                                        Monitor.Wait(CartLocation.notify);
                                    var sending = Encoding.ASCII.GetBytes(CartLocation.GetPosString());
                                    handler.Send(sending);
                                }
                            }
                            catch (Exception ex)
                            {
                                D.Log($"[Position TCP] connection lost");
                            }
                        }).Start();
                    }
                }).Start();
            }
            catch (Exception ex)
            {
                D.Log($"[Position TCP] not started due to {ex.Message}");
            }
        }
    }
}
