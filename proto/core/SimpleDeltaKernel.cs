using pasim.cusim;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using pasim.math;

namespace pasim.core
{

    public class SimpleDeltaKernel : CudaKernel
    {
        public dim3 blockDim { get; set; }

        public dim3 gridDim { get; set; }

        public uint count;
        public Vector2[] positions;
        public float[] masses;
        public Vector2[] delta;
        public float gravitational_force;

        public void Global(CudaContext context, object data)
        {
            int idx = context.blockX;
            float dt = (float)data;

            Vector2 dist, force = new Vector2(0, 0);
            float norm;

            for (uint i = 0; i < count; i++)
            {
                if (i == idx)
                    continue;

                dist = positions[i] - positions[idx];
                norm = Vector2.norm(ref dist);

                force += dist / (norm * norm * norm) * gravitational_force * masses[i] * masses[idx];
            }

            delta[idx] = force * dt;
        }
    }
}
