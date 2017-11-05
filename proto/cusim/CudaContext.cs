using pasim.math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.cusim
{

    public struct CudaContext
    {
        public int gridZ;
        public int gridY;
        public int gridX;

        public int blockZ;
        public int blockY;
        public int blockX;

        internal CudaContext(int gridZ, int gridY, int gridX, int blockZ, int blockY, int blockX)
        {
            this.gridZ = gridZ;
            this.gridY = gridY;
            this.gridX = gridX;

            this.blockZ = blockZ;
            this.blockX = blockX;
            this.blockY = blockY;
        }
    }
}
