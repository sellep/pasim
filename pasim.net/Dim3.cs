using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace pasim.net
{

    [StructLayout(LayoutKind.Sequential)]
    public struct Dim3
    {
        public uint x;
        public uint y;
        public uint z;

        public Dim3(uint x, uint y, uint z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }
    }
}
