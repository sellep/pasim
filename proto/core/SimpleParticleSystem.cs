using pasim.cusim;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using pasim.math;

namespace pasim.core
{

    public class SimpleParticleSystem
    {
        public static float POSITION_MAX = 100;
        public static float MOMENTUM_MAX = 0.1f;
        public static float GRAVITATIONAL_FORCE = 1f;

        public uint count;
        public Vector2[] positions;
        public Vector2[] momentums;
        public Vector2[] delta;
        public float[] masses;

        public SimpleDeltaKernel delta_kernel;
        public UpdateKernel update_kernel;
        public CudaEngine delta_engine;
        public CudaEngine update_engine;

        public SimpleParticleSystem(uint count)
        {
            this.count = count;

            positions = new Vector2[count];
            momentums = new Vector2[count];
            delta = new Vector2[count];
            masses = new float[count];

            for (uint i = 0; i < count; i++)
            {
                positions[i] = new Vector2(POSITION_MAX);
                momentums[i] = new Vector2(MOMENTUM_MAX);
                masses[i] = Rand.NextSingle() * 5 + 0.5f;
            }

            delta_kernel = new SimpleDeltaKernel();
            delta_kernel.count = count;
            delta_kernel.delta = delta;
            delta_kernel.positions = positions;
            delta_kernel.masses = masses;
            delta_kernel.blockDim = new dim3(count, 1, 1);
            delta_kernel.gridDim = new dim3(1, 1, 1);
            delta_kernel.gravitational_force = GRAVITATIONAL_FORCE;

            update_kernel = new UpdateKernel();
            update_kernel.positions = positions;
            update_kernel.momentums = momentums;
            update_kernel.masses = masses;
            update_kernel.delta = delta;
            update_kernel.blockDim = new dim3(count, 1, 1);
            update_kernel.gridDim = new dim3(1, 1, 1);

            delta_engine = new CudaEngine(delta_kernel);
            update_engine = new CudaEngine(update_kernel);
        }

        public void Tick(float dt)
        {
            delta_engine.Launch(dt);
            update_engine.Launch(dt);
        }
    }
}
