using pasim.math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.cusim
{

    public interface CudaKernel
    {

        dim3 blockDim { get; }

        dim3 gridDim { get; }

        void Global(CudaContext context, object data);
    }
}
