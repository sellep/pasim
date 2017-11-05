using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.math
{

    public struct Vector2
    {

        public float x;
        public float y;

        public Vector2(float positionMax)
        {
            x = Rand.NextSingle() * positionMax * 2 - positionMax;
            y = Rand.NextSingle() * positionMax * 2 - positionMax;
        }

        public Vector2(float x, float y)
        {
            this.x = x;
            this.y = y;
        }

        public static Vector2 operator +(Vector2 a, Vector2 b) => new Vector2(a.x + b.x, a.y + b.y);

        public static Vector2 operator -(Vector2 a, Vector2 b) => new Vector2(a.x - b.x, a.y - b.y);

        public static Vector2 operator *(Vector2 a, float b) => new Vector2(a.x * b, a.y * b);

        public static Vector2 operator /(Vector2 a, float b) => new Vector2(a.x / b, a.y / b);

        public static float dot_product(ref Vector2 a, ref Vector2 b)
        {
            return a.x * b.x + a.y * b.y;
        }

        public static float norm(ref Vector2 a)
        {
            return (float)Math.Sqrt(dot_product(ref a, ref a));
        }
    }
}
