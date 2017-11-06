using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.core
{

    public static class Rand
    {
        private static Random _Rand = new Random((int)DateTime.Now.Ticks);

        public static float Nextf(float max) => (float)_Rand.NextDouble() * max;

        public static float Nextf(float min, float max) => (float)_Rand.NextDouble() * (max - min) + min;
    }
}
