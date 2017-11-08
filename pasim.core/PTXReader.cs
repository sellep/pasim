using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.core
{

    public static class PTXReader
    {

        public static string ReadKernelName(string file)
        {
            /**
                looking for something like '\t// .globl\t_Z20delta_momentum_naiveP6float3PK6float4jf'
             */

            string[] lines = File.ReadAllLines(file);
            string[] words;

            for (int i = 0; i < lines.Length; i++)
            {
                if (!lines[i].StartsWith("\t// .globl"))
                    continue;

                words = lines[i].Split('\t');
                return words[words.Length - 1];
            }

            throw new Exception("Entry point not found");
        }
    }
}
