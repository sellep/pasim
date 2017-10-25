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

        public static bool operator ==(Dim3 d1, Dim3 d2)
        {
            return d1.x == d2.x && d1.y == d2.y && d1.z == d2.z;
        }

        public static bool operator !=(Dim3 d1, Dim3 d2)
        {
            return !(d1 == d2);
        }
}
}
