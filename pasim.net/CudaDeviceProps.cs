using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace pasim.net
{
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaDeviceProp
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
        public string name;                  /**< ASCII string identifying device */
        public ulong totalGlobalMem;             /**< Global memory available on device in bytes */
        public ulong sharedMemPerBlock;          /**< Shared memory available per block in bytes */
        public int regsPerBlock;               /**< 32-bit registers available per block */
        public int warpSize;                   /**< Warp size in threads */
        public uint memPitch;                   /**< Maximum pitch in bytes allowed by memory copies */
        public int maxThreadsPerBlock;         /**< Maximum number of threads per block */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxThreadsDim;           /**< Maximum size of each dimension of a block */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxGridSize;             /**< Maximum size of each dimension of a grid */
        public int clockRate;                  /**< Clock frequency in kilohertz */
        public ulong totalConstMem;              /**< Constant memory available on device in bytes */
        public int major;                      /**< Major compute capability */
        public int minor;                      /**< Minor compute capability */
        public ulong textureAlignment;           /**< Alignment requirement for textures */
        public ulong texturePitchAlignment;      /**< Pitch alignment requirement for texture references bound to pitched memory */
        public int deviceOverlap;              /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
        public int multiProcessorCount;        /**< Number of multiprocessors on device */
        public int kernelExecTimeoutEnabled;   /**< Specified whether there is a run time limit on kernels */
        public int integrated;                 /**< Device is integrated as opposed to discrete */
        public int canMapHostMemory;           /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
        public int computeMode;                /**< Compute mode (See ::cudaComputeMode) */
        public int maxTexture1D;               /**< Maximum 1D texture size */
        public int maxTexture1DMipmap;         /**< Maximum 1D mipmapped texture size */
        public int maxTexture1DLinear;         /**< Maximum size for 1D textures bound to linear memory */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture2D;            /**< Maximum 2D texture dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture2DMipmap;      /**< Maximum 2D mipmapped texture dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxTexture2DLinear;      /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture2DGather;      /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxTexture3D;            /**< Maximum 3D texture dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxTexture3DAlt;         /**< Maximum alternate 3D texture dimensions */
        public int maxTextureCubemap;          /**< Maximum Cubemap texture dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture1DLayered;     /**< Maximum 1D layered texture dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxTexture2DLayered;     /**< Maximum 2D layered texture dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTextureCubemapLayered;/**< Maximum Cubemap layered texture dimensions */
        public int maxSurface1D;               /**< Maximum 1D surface size */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxSurface2D;            /**< Maximum 2D surface dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxSurface3D;            /**< Maximum 3D surface dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxSurface1DLayered;     /**< Maximum 1D layered surface dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxSurface2DLayered;     /**< Maximum 2D layered surface dimensions */
        public int maxSurfaceCubemap;          /**< Maximum Cubemap surface dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxSurfaceCubemapLayered;/**< Maximum Cubemap layered surface dimensions */
        public ulong surfaceAlignment;           /**< Alignment requirements for surfaces */
        public int concurrentKernels;          /**< Device can possibly execute multiple kernels concurrently */
        public int ECCEnabled;                 /**< Device has ECC support enabled */
        public int pciBusID;                   /**< PCI bus ID of the device */
        public int pciDeviceID;                /**< PCI device ID of the device */
        public int pciDomainID;                /**< PCI domain ID of the device */
        public int tccDriver;                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
        public int asyncEngineCount;           /**< Number of asynchronous engines */
        public int unifiedAddressing;          /**< Device shares a unified address space with the host */
        public int memoryClockRate;            /**< Peak memory clock frequency in kilohertz */
        public int memoryBusWidth;             /**< Global memory bus width in bits */
        public int l2CacheSize;                /**< Size of L2 cache in bytes */
        public int maxThreadsPerMultiProcessor;/**< Maximum resident threads per multiprocessor */
        public int streamPrioritiesSupported;  /**< Device supports stream priorities */
        public int globalL1CacheSupported;     /**< Device supports caching globals in L1 */
        public int localL1CacheSupported;      /**< Device supports caching locals in L1 */
        public ulong sharedMemPerMultiprocessor; /**< Shared memory available per multiprocessor in bytes */
        public int regsPerMultiprocessor;      /**< 32-bit registers available per multiprocessor */
        public int managedMemory;              /**< Device supports allocating managed memory on this system */
        public int isMultiGpuBoard;            /**< Device is on a multi-GPU board */
        public int multiGpuBoardGroupID;       /**< Unique identifier for a group of devices on the same multi-GPU board */
        public int hostNativeAtomicSupported;  /**< Link between the device and the host supports native atomic operations */
        public int singleToDoublePrecisionPerfRatio; /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
        public int pageableMemoryAccess;       /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
        public int concurrentManagedAccess;    /**< Device can coherently access managed memory concurrently with the CPU */
        public int computePreemptionSupported; /**< Device supports Compute Preemption */
        public int canUseHostPointerForRegisteredMem; /**< Device can access host registered memory at the same virtual address as the CPU */
        public int cooperativeLaunch;          /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel */
        public int cooperativeMultiDeviceLaunch; /**< Device can participate in cooperative kernels launched via ::cudaLaunchCooperativeKernelMultiDevice */
        public ulong sharedMemPerBlockOptin;     /**< Per device maximum shared memory per block usable by special opt in */
    }
}
