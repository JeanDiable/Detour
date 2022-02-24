using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Text;

namespace DetourCore.CartDefinition.InternalTypes
{
    [LayoutDefinition.ComponentType(typename = "Direct link to Hokuyo Lidar")]
    public class HokuyoURGLidar:Lidar.Lidar2D
    {
        public string ConnectString = "192.168.0.10:10940";

        /// <summary>
        /// Read to "\n\n" from NetworkStream
        /// </summary>
        /// <returns>receive data</returns>
        static string read_line(NetworkStream stream)
        {
            if (stream.CanRead)
            {
                StringBuilder sb = new StringBuilder();
                bool is_NL2 = false;
                bool is_NL = false;
                do
                {
                    char buf = (char)stream.ReadByte();
                    if (buf == '\n')
                    {
                        if (is_NL)
                        {
                            is_NL2 = true;
                        }
                        else
                        {
                            is_NL = true;
                        }
                    }
                    else
                    {
                        is_NL = false;
                    }
                    sb.Append(buf);
                } while (!is_NL2);

                return sb.ToString();
            }
            else
            {
                return null;
            }
        }

        /// <summary>
        /// write data
        /// </summary>
        static bool write(NetworkStream stream, string data)
        {
            if (stream.CanWrite)
            {
                byte[] buffer = Encoding.ASCII.GetBytes(data);
                stream.Write(buffer, 0, buffer.Length);
                return true;
            }
            else
            {
                return false;
            }
        }

        public class SCIP_Writer
        {
            /// <summary>
            /// Create MD command
            /// </summary>
            /// <param name="start">measurement start step</param>
            /// <param name="end">measurement end step</param>
            /// <param name="grouping">grouping step number</param>
            /// <param name="skips">skip scan number</param>
            /// <param name="scans">get scan numbar</param>
            /// <returns>created command</returns>
            public static string MD(int start, int end, int grouping = 1, int skips = 0, int scans = 0)
            {
                return "MD" + start.ToString("D4") + end.ToString("D4") + grouping.ToString("D2") + skips.ToString("D1") + scans.ToString("D2") + "\n";
            }

            public static string VV()
            {
                return "VV\n";
            }

            public static string II()
            {
                return "II\n";
            }

            public static string PP()
            {
                return "PP\n";
            }

            public static string SCIP2()
            {
                return "SCIP2.0" + "\n";
            }

            public static string QT()
            {
                return "QT\n";
            }
        }

        public class SCIP_Reader
        {
            /// <summary>
            /// read MD command
            /// </summary>
            /// <param name="get_command">received command</param>
            /// <param name="time_stamp">timestamp data</param>
            /// <param name="distances">distance data</param>
            /// <returns>is successful</returns>
            public static bool MD(string get_command, ref long time_stamp, ref List<long> distances)
            {
                distances.Clear();
                string[] split_command = get_command.Split(new char[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);

                if (!split_command[0].StartsWith("MD"))
                {
                    return false;
                }

                if (split_command[1].StartsWith("00"))
                {
                    return true;
                }
                else if (split_command[1].StartsWith("99"))
                {
                    time_stamp = SCIP_Reader.decode(split_command[2], 4);
                    distance_data(split_command, 3, ref distances);
                    return true;
                }
                else
                {
                    return false;
                }
            }

            /// <summary>
            /// read distance data
            /// </summary>
            /// <param name="lines"></param>
            /// <param name="start_line"></param>
            /// <returns></returns>
            public static bool distance_data(string[] lines, int start_line, ref List<long> distances)
            {
                StringBuilder sb = new StringBuilder();
                for (int i = start_line; i < lines.Length; ++i)
                {
                    sb.Append(lines[i].Substring(0, lines[i].Length - 1));
                }
                return SCIP_Reader.decode_array(sb.ToString(), 3, ref distances);
            }

            /// <summary>
            /// decode part of string 
            /// </summary>
            /// <param name="data">encoded string</param>
            /// <param name="size">encode size</param>
            /// <param name="offset">decode start position</param>
            /// <returns>decode result</returns>
            public static long decode(string data, int size, int offset = 0)
            {
                long value = 0;

                for (int i = 0; i < size; ++i)
                {
                    value <<= 6;
                    value |= (long)data[offset + i] - 0x30;
                }

                return value;
            }

            /// <summary>
            /// decode multiple data
            /// </summary>
            /// <param name="data">encoded string</param>
            /// <param name="size">encode size</param>
            /// <returns>decode result</returns>
            public static bool decode_array(string data, int size, ref List<long> decoded_data)
            {
                for (int pos = 0; pos <= data.Length - size; pos += size)
                {
                    decoded_data.Add(decode(data, size, pos));
                }
                return true;
            }
        }

        public int start_step = 0;
        public int end_step = 760;

        private NetworkStream stream;
        private int resolution, frontAxis;
        private int frame;

        public override void InitReadLidar()
        {
            Console.WriteLine("Init Hokuyo Lidar reading...");
            var nls=ConnectString.Split(':');
            TcpClient urg = new TcpClient();
            urg.Connect(nls[0], Convert.ToInt32(nls[1]));
            stream = urg.GetStream();

            write(stream, SCIP_Writer.PP());
            var measurementParams=read_line(stream);
            var lines=measurementParams.Split('\n');

            var lidarName = lines.First(l => l.StartsWith("MODL")).Split(';')[0].Replace("MODL:", "");
            resolution = Convert.ToInt32(lines.First(l => l.StartsWith("ARES")).Split(';')[0].Replace("ARES:", ""));
            frontAxis = Convert.ToInt32(lines.First(l => l.StartsWith("AFRT")).Split(';')[0].Replace("AFRT:", ""));
            Console.WriteLine($"Model:{lidarName}, Resolution:{resolution}, frontAxis:{frontAxis}");

            write(stream, SCIP_Writer.SCIP2());
            read_line(stream); // ignore echo back
            write(stream, SCIP_Writer.MD(start_step, end_step));
            read_line(stream);  // ignore echo back

            write(stream, SCIP_Writer.QT());    // stop measurement mode
            read_line(stream); // ignore echo back
        }

        public override LidarOutput ReadLidar()
        {
            start:
            string receive_data = read_line(stream);
            List<long> distances = new List<long>();
            long time_stamp = 0;
            if (!SCIP_Reader.MD(receive_data, ref time_stamp, ref distances))
            {
                Console.WriteLine("Hokuyo read error");
                Console.WriteLine(receive_data);
                InitReadLidar();
                goto start;
            }
            if (distances.Count == 0)
            {
                Console.WriteLine("Hokuyo read error");
                Console.WriteLine(receive_data);
                goto start;
            }

            return new LidarOutput()
            {
                points = distances.Select((d, i) => new RawLidar()
                    {d = d, intensity = 0, th = ((float) i - frontAxis) / resolution * 360}).ToArray(),
                tick = frame++
            };
        }
    }
}
