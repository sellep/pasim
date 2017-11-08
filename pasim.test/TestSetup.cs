using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.test
{

    public class TestSetup
    {

        public string KernelDirectory { get; set; }

        public bool IgnoreNaive { get; set; }

        public string KernelType { get; set; }

        public uint N { get; set; }

        public uint OuterIterations { get; set; } = 2;

        public uint InnerIterations { get; set; } = 4;

        public bool NoLogFile { get; set; }

        public void Print()
        {
            Console.WriteLine("Test setup");
            Console.WriteLine($"Kernel directory: {KernelDirectory}");
            Console.WriteLine($"Ignore naive: {IgnoreNaive}");
            Console.WriteLine($"Kernel type: {KernelType}");
            Console.WriteLine($"Outer iterations: {OuterIterations}");
            Console.WriteLine($"Inner iterations: {InnerIterations}");
            Console.WriteLine($"N: {N}");
            Console.WriteLine($"No log file: {NoLogFile}");
        }
    }
}
