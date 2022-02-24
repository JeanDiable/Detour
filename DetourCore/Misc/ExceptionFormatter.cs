using System;
using System.Linq;

namespace DetourCore.Misc
{
    public class ExceptionFormatter
    {
        public static string FormatEx(Exception ex)
        {
            if (ex == null) return "";
            string exStr = $"* Exception({ex.GetType().Name}):{ex.Message}, stack:{ex.StackTrace}\r\n";
            foreach (var iEx in ex.GetType().GetFields().Where(p => typeof(Exception).IsAssignableFrom(p.FieldType)))
            {
                var ie = iEx.GetValue(ex);
                if (ie == null) continue;
                exStr += $"  *{iEx.Name} " + FormatEx((Exception) ie);
            }

            foreach (var iEx in ex.GetType().GetProperties()
                .Where(p => typeof(Exception).IsAssignableFrom(p.PropertyType)))
            {
                var ie = iEx.GetValue(ex);
                if (ie == null) continue;
                exStr += $"  *{iEx.Name} " + FormatEx((Exception)ie);
            }

            foreach (var iEx in ex.GetType().GetProperties()
                .Where(p => typeof(Exception[]) == p.PropertyType))
            {
                var ies = (Exception[])iEx.GetValue(ex);
                if (ies == null) continue;
                foreach (var ie in ies)
                {
                    exStr += $"  *{iEx.Name} " + FormatEx((Exception)ie);
                }
            }
            return exStr;
        }
    }
}
