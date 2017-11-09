using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.core
{

    public static class dim3ext
    {

        public static dim3 Parse(string str)
        {
            string sub = str.Substring(1, str.Length - 2);
            string[] dims = sub.Split(';');

            dim3 result = new dim3(
                uint.Parse(dims[0].Trim()),
                uint.Parse(dims[1].Trim()),
                uint.Parse(dims[2].Trim()));

            return result;
        }
    }
}
