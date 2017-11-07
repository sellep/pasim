using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.core
{

    public static class KernelDescriptor
    {

        public static bool IsValidDimensionFor(string module, dim3 gridDim, dim3 blockDim)
        {
            uint? blockDimx = InternalBlockDimension(GetDescriptions(module));
            if (blockDimx == null)
                return true;

            return blockDimx.Value == blockDim.x;
        }

        public static bool UsesDynamicSharedMemory(string module)
        {
            string[] descriptions = GetDescriptions(module);
            return descriptions.Contains("shmem") && InternalBlockDimension(descriptions) == null;
        }

        public static bool UsesStaticDynamicSharedMemory(string module)
        {
            string[] descriptions = GetDescriptions(module);
            return descriptions.Contains("shmem") && InternalBlockDimension(descriptions) != null;
        }

        public static uint? BlockDimension(string module) => InternalBlockDimension(GetDescriptions(module));

        private static uint? InternalBlockDimension(string[] descriptions)
        {
            uint log;

            for (int i = 0; i < descriptions.Length; i++)
            {
                if (descriptions[i][0] != 'b')
                    continue;

                if (!uint.TryParse(descriptions[i].Substring(1), out log))
                    continue;

                return (uint)2 << (int)log;
            }

            return null;
        }

        private static string[] GetDescriptions(string module) => Path.GetFileNameWithoutExtension(module).Split('_');
    }
}
