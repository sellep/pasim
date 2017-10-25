using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.net
{

    public class ParticleSystem
    {
        internal IntPtr? Handle = null;
        internal IntPtr PositionsHandle;

        public ParticleSystem(uint particles, float massMin, float massMax, float positionMax, float momentumMax)
        {
            Particles = particles;

            Masses = new float[particles];
            Positions = new Vector3[particles];
            Momentums = new Vector3[particles];

            Random rand = new Random();

            for (int i = 0; i < particles; i++)
            {
                Masses[i] = (float) rand.NextDouble() * (massMax - massMin) + massMin;
                Positions[i] = new Vector3((float)rand.NextDouble() * positionMax, (float)rand.NextDouble() * positionMax, (float)rand.NextDouble() * positionMax);
                Momentums[i] = new Vector3((float)rand.NextDouble() * momentumMax, (float)rand.NextDouble() * momentumMax, (float)rand.NextDouble() * momentumMax);
            }
        }

        public uint Particles { get; }

        public float[] Masses { get; }

        public Vector3[] Positions { get; }

        public Vector3[] Momentums { get; }
    }
}
