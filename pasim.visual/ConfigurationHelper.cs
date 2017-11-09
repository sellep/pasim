using ManagedCuda.VectorTypes;
using Newtonsoft.Json;
using pasim.core;
using pasim.visual;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.visual
{

    public static class ConfigurationHelper
    {

        public static void Parse(string line, out string kernel, out dim3 gridDim, out dim3 blockDim)
        {
            string[] words = line.Split(' ');

            kernel = words[0];

            StringBuilder sb = new StringBuilder();
            int i;
            bool isDim = false;
            for (i = 0; i < line.Length; i++)
            {
                if (isDim)
                {
                    sb.Append(line[i]);

                    if (line[i] == ')')
                        break;
                }
                else if (line[i] == '(')
                {
                    sb.Append(line[i]);
                    isDim = true;
                }
            }

            gridDim = dim3ext.Parse(sb.ToString());

            sb.Clear();
            isDim = false;
            for (i = i + 1; i < line.Length; i++)
            {
                if (isDim)
                {
                    sb.Append(line[i]);

                    if (line[i] == ')')
                        break;
                }
                else if (line[i] == '(')
                {
                    sb.Append(line[i]);
                    isDim = true;
                }
            }

            blockDim = dim3ext.Parse(sb.ToString());
        }

        public static PasimSetup Load()
        {
            string cfgPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "pasim.cfg");
            if (!File.Exists(cfgPath))
                return null;

            string cfg = File.ReadAllText(cfgPath);
            if (string.IsNullOrEmpty(cfg))
                return null;

            PasimSetup setup = JsonConvert.DeserializeObject<PasimSetup>(cfg);
            return setup;
        }

        public static void Save(PasimSetup setup, string path = null)
        {
            string cfgPath = path??Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "pasim.cfg");
            string cfg = JsonConvert.SerializeObject(setup, Formatting.Indented);

            using (Stream s = File.OpenWrite(cfgPath))
            {
                s.SetLength(0);

                using (StreamWriter sw = new StreamWriter(s))
                {
                    sw.WriteLine(cfg);
                }
            }
        }
    }
}
