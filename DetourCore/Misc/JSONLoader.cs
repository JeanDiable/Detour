using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using Newtonsoft.Json;

namespace DetourCore.Misc
{
    public class JSONLoader<T> : JsonConverter
    {
        private Tuple<string, Type>[] Types;
        
        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            writer.WriteStartObject();
            writer.WritePropertyName("type");
            var tls = new List<string>();
            var type = value.GetType();
            while (type != typeof(object))
            {
                tls.Add(type.Name.ToLower());
                type = type.BaseType;
            }

            writer.WriteValue(string.Join(",", tls));
            writer.WritePropertyName("options");
            writer.WriteStartObject();
            foreach (var f in value.GetType().GetFields())
            {
                writer.WritePropertyName(f.Name);
                writer.WriteRawValue(JsonConvert.SerializeObject(f.GetValue(value), serializer.Formatting));
            }
            writer.WriteEndObject();
            writer.WriteEndObject();
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            if (reader.TokenType == JsonToken.Null)
                return null;

            if (Types == null)
                Types = AppDomain.CurrentDomain.GetAssemblies().SelectMany(asm =>
                {
                    try
                    {
                        return asm.GetTypes()
                            .Where(t => typeof(T).IsAssignableFrom(t))
                            .Select(t => Tuple.Create(t.Name.ToLower(), t));
                    }
                    catch (ReflectionTypeLoadException le)
                    {
                        Console.WriteLine($"Failed to fully load {asm.FullName} asm");
                        Console.WriteLine(ExceptionFormatter.FormatEx(le));
                        return le.Types
                            .Where(t => typeof(T).IsAssignableFrom(t))
                            .Select(t => Tuple.Create(t.Name.ToLower(), t));
                    }
                }).ToArray();

            reader.Read();
            if ((string)reader.Value != "type") throw new InvalidOperationException();
            reader.Read();
            var tname = (string)reader.Value;
            var tls = tname.Split(',');
            Tuple<string, Type> type = null;
            for (int i = 0; i < tls.Length && type == null; ++i)
            {
                type = Types.FirstOrDefault(t => t.Item1 == tls[i]);
                if (type == null && i == 0) D.Log($"* json read type {tname} failed, fallback");
                if (type != null && i > 0) D.Log($"* json read type {tname} fallback to {tls[i]}");
            }

            if (type==null)
                throw new Exception($"{tname} is not recognized");
            var value = type.Item2.GetConstructor(new Type[0]).Invoke(new object[] { });
            reader.Read(); // options
            reader.Read(); // inner.
            serializer.Populate(reader, value);
            reader.Read();
            return value;
        }

        public override bool CanConvert(Type objectType)
        {
            return typeof(LayoutDefinition.Component).IsAssignableFrom(objectType);
        }
    }
}