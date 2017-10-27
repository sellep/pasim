using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace meshes
{

    public static class Rand
    {

        private static Random _Rand = new Random((int)DateTime.Now.Ticks);

        public static float NextSingle() => (float)_Rand.NextDouble();

        public static uint Next(uint max) => (uint)_Rand.Next((int)max);
    }
}
