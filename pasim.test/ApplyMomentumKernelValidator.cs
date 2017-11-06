using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using pasim.core;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.test
{

    public class ApplyMomentumKernelValidator : ApplyMomentumBase
    {
        private ParticleSystem _System;
        

        public ApplyMomentumKernelValidator(string kernelDirectory, CudaContext ctx, ParticleSystem system)
            : base(kernelDirectory, ctx)
        {
            _System = system;
        }

        public void Validate(dim3 gridDim, dim3 blockDim)
        {
            List<KernelValidation> validations = new List<KernelValidation>();
            uint i;

            foreach (string file in _Modules.Keys)
            {
                _System.SyncDevice();

                CudaKernel kernel = CreateCudaKernel(file, _Modules[file], gridDim, blockDim);
                kernel.Run(_System.DevBodies, _System.DevMomentums, _System.N, 0.01f);

                validations.Add(new KernelValidation(file, _System.GetDeviceBodies()));
            }

            for (i = 0; i < _System.N; i++)
            {
                StringBuilder sb = new StringBuilder();
                bool error = false;

                float4 root = validations.First().Bodies[i];

                foreach (KernelValidation validation in validations)
                {
                    if (validation.Bodies[i] != root)
                    {
                        error = true;
                    }

                    sb.AppendLine($"{Path.GetFileName(validation.Module)}: {validation.Bodies[i]}");
                }

                if (error)
                    throw new Exception($"Data missmatch at {i}:{Environment.NewLine}{sb.ToString()}");
            }
        }
    }
}
