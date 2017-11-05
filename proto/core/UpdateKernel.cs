using pasim.cusim;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using pasim.math;

namespace pasim.core
{

    public class UpdateKernel : CudaKernel
    {
        public dim3 blockDim { get; set; }

        public dim3 gridDim { get; set; }

        public Vector2[] positions;
        public Vector2[] momentums;
        public float[] masses;
        public Vector2[] delta;

        public void Global(CudaContext context, object data)
        {
            int idx = context.blockX;
            float dt = (float) data;

            momentums[idx] += delta[idx];
            positions[idx] += momentums[idx] * dt / masses[idx];
        }
    }
}
