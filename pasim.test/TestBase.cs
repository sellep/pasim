using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.test
{

    public abstract class TestBase
    {
        protected CudaContext _Context;
        protected Dictionary<string, CUmodule> _Modules = new Dictionary<string, CUmodule>();

        protected static dim3[] _GridDims = new[]
        {
            new dim3(512, 1, 1),
            new dim3(256, 1, 1),
            new dim3(128, 1, 1),
            new dim3(64, 1, 1),
            new dim3(32, 1, 1),
            new dim3(16, 1, 1),
            new dim3(8, 1, 1),
            new dim3(4, 1, 1),
            new dim3(2, 1, 1)
        };

        protected static dim3[] _BlockDims = new[]
        {
            new dim3(1024, 1, 1),
            new dim3(512, 1, 1),
            new dim3(256, 1, 1),
            new dim3(128, 1, 1),
            new dim3(64, 1, 1),
            new dim3(32, 1, 1),
        };

        protected abstract string _KernelPattern { get; }

        public TestBase(string kernelDirectory, CudaContext ctx)
        {
            foreach (string file in Directory.GetFiles(kernelDirectory, _KernelPattern))
            {
                Console.WriteLine($"found module {Path.GetFileName(file)}");
                _Modules.Add(file, ctx.LoadModule(file));
            }

            if (_Modules.Count == 0)
                throw new Exception("No kernels found");

            _Context = ctx;
        }
    }
}
