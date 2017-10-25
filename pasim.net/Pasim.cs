using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace pasim.net
{

    public static class Pasim
    {

        [DllImport("pasim.core.dll")]
        private static extern int pasim_init(ref IntPtr handle, uint particles, IntPtr dimensions, IntPtr masses, IntPtr positions, IntPtr momentums);

        [DllImport("pasim.core.dll")]
        private static extern int pasim_deinit(IntPtr handle);

        [DllImport("pasim.core.dll")]
        private static extern int pasim_tick(IntPtr handle, float dt);

        [DllImport("pasim.core.dll")]
        private static extern void pasim_error_string(StringBuilder buffer, int capacity, int status);

        [DllImport("pasim.core.dll")]
        private static extern int pasim_sync_host(IntPtr handle);

        [DllImport("pasim.core.dll")]
        private static extern int pasim_dev_props(IntPtr handle);

        public static CudaStatus Init(ParticleSystem system)
        {
            Assert.NotNull(system);
            Assert.Null(system.Handle);

            IntPtr dimensions = MallocAndCopy(new[] { system.BlockDim, system.GridDim });
            IntPtr masses = MallocAndCopy(system.Masses);
            IntPtr momentums = MallocAndCopy(system.Momentums);
            system.PositionsHandle = MallocAndCopy(system.Positions);

            IntPtr handle = IntPtr.Zero;
            CudaStatus status = (CudaStatus) pasim_init(ref handle, system.Particles, dimensions, masses, system.PositionsHandle, momentums);

            Marshal.FreeHGlobal(dimensions);
            Marshal.FreeHGlobal(masses);
            Marshal.FreeHGlobal(momentums);

            if (status == CudaStatus.cudaSuccess)
            {
                system.Handle = handle;
            }

            return status;
        }

        public static string GetErrorString(CudaStatus status)
        {
            StringBuilder sb = new StringBuilder(200);
            pasim_error_string(sb, sb.Capacity, (int)status);
            return sb.ToString();
        }

        public static CudaStatus Tick(ParticleSystem system, float dt)
        {
            Assert.NotNull(system);
            Assert.NotNull(system.Handle);

            return (CudaStatus) pasim_tick(system.Handle.Value, dt);
        }

        public static void Deinit(ParticleSystem system)
        {
            Assert.NotNull(system);
            Assert.NotNull(system.Handle);

            int status = pasim_deinit(system.Handle.Value);

            Marshal.FreeHGlobal(system.PositionsHandle);
            system.Handle = default(IntPtr);
        }

        public static CudaStatus Update(ParticleSystem system)
        {
            Assert.NotNull(system);
            Assert.NotNull(system.Handle);

            CudaStatus status = (CudaStatus)pasim_sync_host(system.Handle.Value);

            IntPtr handle = system.PositionsHandle;
            if (status == CudaStatus.cudaSuccess)
            {
                int size = Marshal.SizeOf<Vector3>();
                IntPtr j = system.PositionsHandle;
                for (int i = 0; i < system.Particles; i++, handle += size)
                {
                    system.Positions[i] = Marshal.PtrToStructure<Vector3>(handle);
                }
            }

            return status;
        }

        public static CudaDeviceProp GetDeviceProperties()
        {
            IntPtr handle = Malloc<CudaDeviceProp>(out int size);

            CudaStatus status = (CudaStatus)pasim_dev_props(handle);
            CudaDeviceProp props = Marshal.PtrToStructure<CudaDeviceProp>(handle);
            Marshal.FreeHGlobal(handle);

            if (status != CudaStatus.cudaSuccess)
                throw new Exception(status.ToString());

            return props;
        }

        public static void QueryDimensions(uint requiredThreads, out Dim3 block, out Dim3 grid, int? maxThreadsPerBlock = null)
        {
            //block size must be multiple of 32

            block = new Dim3(0, 0, 1);
            grid = new Dim3(0, 0, 1);

            uint diff = uint.MaxValue;
            Dim3 tmp = new Dim3(0, 0, 0);

            CudaDeviceProp props = GetDeviceProperties();

            block.x = (uint)Math.Sqrt(maxThreadsPerBlock.GetValueOrDefault(props.maxThreadsPerBlock));
            block.y = block.x;

            for (tmp.x = (uint)Math.Ceiling(Math.Sqrt((double)requiredThreads / (block.x * block.y))); tmp.x > 0; tmp.x--)
            {
                while (block.x * block.y * tmp.x * tmp.y < requiredThreads)
                {
                    tmp.y++;
                }

                if ((block.x * block.y * tmp.x * tmp.y) - requiredThreads < diff)
                {
                    diff = (block.x * block.y * tmp.x * tmp.y) - requiredThreads;
                    grid.x = tmp.x;
                    grid.y = tmp.y;
                }
            }
        }

        private static IntPtr MallocAndCopy(float[] arr)
        {
            IntPtr handle = Malloc<float>(out int size, arr.Length);

            Marshal.Copy(arr, 0, handle, arr.Length);
            return handle;
        }

        private static IntPtr MallocAndCopy<T>(T[] arr)
        {
            IntPtr handle = Malloc<T>(out int size, arr.Length);

            IntPtr j = handle;
            for (int i = 0; i < arr.Length; i++, j += size)
            {
                Marshal.StructureToPtr(arr[i], j, false);
            }

            return handle;
        }

        private static IntPtr Malloc<T>(out int size, int length = 1)
        {
            size = Marshal.SizeOf<T>();
            return Marshal.AllocHGlobal(size * length);
        }
    }
}
