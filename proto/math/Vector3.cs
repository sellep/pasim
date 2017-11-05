using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace pasim.math
{
    [StructLayout(LayoutKind.Sequential)]
    public struct Vector3
    {
        public float x;
        public float y;
        public float z;

        public Vector3(float positionMax)
        {
            x = Rand.NextSingle() * positionMax * 2 - positionMax;
            y = Rand.NextSingle() * positionMax * 2 - positionMax;
            z = Rand.NextSingle() * positionMax * 2 - positionMax;
        }

        public Vector3(float x, float y, float z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public static Vector3 operator +(Vector3 a, Vector3 b) => new Vector3(a.x + b.x, a.y + b.y, a.z + b.z);

        public static Vector3 operator -(Vector3 a, Vector3 b) => new Vector3(a.x - b.x, a.y - b.y, a.z - b.z);

        public static Vector3 operator *(Vector3 a, float b) => new Vector3(a.x * b, a.y * b, a.z * b);

        public static Vector3 operator /(Vector3 a, float b) => new Vector3(a.x / b, a.y / b, a.z / b);

        public override string ToString() => $"({x}, {y}, {z})";

        public static float distance(Vector3 a, Vector3 b)
        {
            Vector3 diff = a - b;
            return (float)Math.Sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        }

        public static void zero(ref Vector3 v)
        {
            v.x = v.y = v.z = 0;
        }

        public static float dot_product(ref Vector3 a, ref Vector3 b)
        {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        public static float norm(ref Vector3 a)
        {
            return (float)Math.Sqrt(dot_product(ref a, ref a));
        }
    }
}
