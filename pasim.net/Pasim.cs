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
        private static extern int pasim_init(ref IntPtr handle, uint particles, IntPtr masses, IntPtr positions, IntPtr momentums);

        [DllImport("pasim.core.dll")]
        private static extern int pasim_deinit(IntPtr handle);

        [DllImport("pasim.core.dll")]
        private static extern void pasim_tick(IntPtr handle, float dt);

        [DllImport("pasim.core.dll")]
        private static extern void pasim_error_string(StringBuilder buffer, int capacity, int status);

        public static CudaStatus Init(ParticleSystem system)
        {
            Assert.NotNull(system);
            Assert.Null(system.Handle);

            IntPtr masses = MallocAndCopy(system.Masses);
            IntPtr positions = MallocAndCopy(system.Positions);
            IntPtr momentums = MallocAndCopy(system.Momentums);

            IntPtr handle = IntPtr.Zero;
            CudaStatus status = (CudaStatus) pasim_init(ref handle, system.Particles, masses, positions, momentums);

            Marshal.FreeHGlobal(masses);
            Marshal.FreeHGlobal(positions);
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

        public static void Tick(ParticleSystem system, float dt)
        {
            Assert.NotNull(system);
            Assert.NotNull(system.Handle);

            pasim_tick(system.Handle.Value, dt);
        }

        public static void Deinit(ParticleSystem system)
        {
            Assert.NotNull(system);
            Assert.NotNull(system.Handle);

            int status = pasim_deinit(system.Handle.Value);

            system.Handle = default(IntPtr);
        }

        private static IntPtr MallocAndCopy(float[] arr)
        {
            int size = Marshal.SizeOf<float>() * arr.Length;
            IntPtr handle = Marshal.AllocHGlobal(size);
            Marshal.Copy(arr, 0, handle, arr.Length);
            return handle;
        }

        private static IntPtr MallocAndCopy(Vector3[] arr)
        {
            int size = Marshal.SizeOf<Vector3>();
            IntPtr handle = Marshal.AllocHGlobal(size * arr.Length);

            IntPtr j = handle;
            for (int i = 0; i < arr.Length; i++, handle += size)
            {
                Marshal.StructureToPtr(arr[i], handle, false);
            }

            return handle;
        }
    }
}
