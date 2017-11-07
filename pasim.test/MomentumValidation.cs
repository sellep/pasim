using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.test
{

    public class MomentumValidation
    {

        public MomentumValidation(string module, float4[] bodies)
        {
            Module = module;
            Bodies = bodies;
        }

        public string Module { get; }

        public float4[] Bodies { get; }
    }
}
