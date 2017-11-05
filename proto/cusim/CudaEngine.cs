using pasim.math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace pasim.cusim
{

    public class CudaEngine
    {
        private readonly object _Sync = new object();
        private CudaKernel _Kernel = null;
        private List<CudaContext> _Contexts = new List<CudaContext>();

        public CudaEngine(CudaKernel kernel)
        {
            _Kernel = kernel;

            for (int grid_z = 0; grid_z < kernel.gridDim.z; grid_z++)
            {
                for (int grid_y = 0; grid_y < kernel.gridDim.y; grid_y++)
                {
                    for (int grid_x = 0; grid_x < kernel.gridDim.x; grid_x++)
                    {
                        //blocks
                        for (int block_z = 0; block_z < kernel.blockDim.z; block_z++)
                        {
                            for (int block_y = 0; block_y < kernel.blockDim.y; block_y++)
                            {
                                for (int block_x = 0; block_x < kernel.blockDim.x; block_x++)
                                {
                                    _Contexts.Add(new CudaContext(grid_z, grid_y, grid_x, block_z, block_y, block_x));
                                }
                            }
                        }
                    }
                }
            }
        }

        public void Launch(object data = null)
        {
            Thread[] cores = new Thread[Environment.ProcessorCount];

            for (int i = 0; i < cores.Length; i++)
            {
                cores[i] = new Thread(CoreJob);
                cores[i].Start(data);
            }

            for (int i = 0; i < cores.Length; i++)
            {
                cores[i].Join();
            }
        }

        private void CoreJob(object data)
        {
            while (true)
            {
                CudaContext context;

                lock (_Sync)
                {
                    if (_Contexts.Count == 0)
                        return;

                    context = _Contexts.First();
                    _Contexts.RemoveAt(0);
                }

                _Kernel.Global(context, data);
            }
        }
    }
}
