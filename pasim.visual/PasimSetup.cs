using ManagedCuda.VectorTypes;
using Newtonsoft.Json;
using pasim.core;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.visual
{

    public class PasimSetup
    {
        private const string n_arg = "n=";
        private const string multiplier_arg = "nm=";
        private const string momentum_kernel_arg = "mom_k=";
        private const string momentum_grid_arg = "mom_g=";
        private const string momentum_block_arg = "mom_b=";
        private const string position_kernel_arg = "pos_k=";
        private const string position_grid_arg = "pos_g=";
        private const string position_block_arg = "pos_b=";
        private const string g_arg = "g=";
        private const string d_arg = "d=";
        private const string pos_max_arg = "pos_max=";
        private const string mass_min_arg = "mass_min=";
        private const string mass_max_arg = "mass_max=";
        private const string mom_max_arg = "mom_max=";
        private const string dt_arg = "dt=";

        public string KernelDirectory { get; set; }

        public string MomentumKernel { get; set; }

        public dim3 MomentumGrid { get; set; }

        public dim3 MomentumBlock { get; set; }

        public string PositionKernel { get; set; }

        public dim3 PositionGrid { get; set; }

        public dim3 PositionBlock { get; set; }

        public uint N { get; set; }

        public uint NMultiplier { get; set; }

        public float G { get; set; }

        public float InitPositionMax { get; set; }

        public float InitMassMin { get; set; }

        public float InitMassMax { get; set; }

        public float InitMomentumMax { get; set; }

        public float DT { get; set; }

        [JsonIgnore]
        public string MomentumKernelPath => Path.Combine(KernelDirectory, MomentumKernel + ".ptx");

        [JsonIgnore]
        public string PositionKernelPath => Path.Combine(KernelDirectory, PositionKernel + ".ptx");

        public static PasimSetup Parse(string[] args)
        {
            PasimSetup setup = new PasimSetup();

            foreach (string arg in args)
            {
                if (arg.StartsWith(d_arg))
                {
                    setup.KernelDirectory = arg.Substring(d_arg.Length);
                }
                else if (arg.StartsWith(momentum_kernel_arg))
                {
                    setup.MomentumKernel = arg.Substring(momentum_kernel_arg.Length);
                }
                else if (arg.StartsWith(momentum_grid_arg))
                {
                    setup.MomentumGrid = dim3ext.Parse(arg.Substring(momentum_grid_arg.Length));
                }
                else if (arg.StartsWith(momentum_block_arg))
                {
                    setup.MomentumBlock = dim3ext.Parse(arg.Substring(momentum_block_arg.Length));
                }
                else if (arg.StartsWith(position_kernel_arg))
                {
                    setup.PositionKernel = arg.Substring(position_kernel_arg.Length);
                }
                else if (arg.StartsWith(position_grid_arg))
                {
                    setup.PositionGrid = dim3ext.Parse(arg.Substring(position_grid_arg.Length));
                }
                else if (arg.StartsWith(position_block_arg))
                {
                    setup.PositionBlock = dim3ext.Parse(arg.Substring(position_block_arg.Length));
                }
                else if (arg.StartsWith(n_arg))
                {
                    setup.N = uint.Parse(arg.Substring(n_arg.Length));
                }
                else if (arg.StartsWith(multiplier_arg))
                {
                    setup.NMultiplier = uint.Parse(arg.Substring(multiplier_arg.Length));
                }
                else if (arg.StartsWith(g_arg))
                {
                    setup.G = float.Parse(arg.Substring(g_arg.Length));
                }
                else if (arg.StartsWith(pos_max_arg))
                {
                    setup.InitPositionMax = float.Parse(arg.Substring(pos_max_arg.Length));
                }
                else if (arg.StartsWith(mass_min_arg))
                {
                    setup.InitMassMin = float.Parse(arg.Substring(mass_min_arg.Length));
                }
                else if (arg.StartsWith(mass_max_arg))
                {
                    setup.InitMassMax = float.Parse(arg.Substring(mass_max_arg.Length));
                }
                else if (arg.StartsWith(mom_max_arg))
                {
                    setup.InitMomentumMax = float.Parse(arg.Substring(mom_max_arg.Length));
                }
                else if (arg.StartsWith(dt_arg))
                {
                    setup.DT = float.Parse(arg.Substring(dt_arg.Length));
                }
            }

            return setup;
        }

        public string CreateCLArgs()
        {
            return
                $"{d_arg}{KernelDirectory} " +
                $"{n_arg}{N} " +
                $"{multiplier_arg}{NMultiplier} " +
                $"{g_arg}{G} " +
                $"{pos_max_arg}{InitPositionMax} " +
                $"{mass_min_arg}{InitMassMin} " +
                $"{mass_max_arg}{InitMassMax} " +
                $"{mom_max_arg}{InitMomentumMax} " +
                $"{dt_arg}{DT} " +
                $"\"{momentum_kernel_arg}{MomentumKernel}\" " +
                $"\"{momentum_grid_arg}{MomentumGrid}\" " +
                $"\"{momentum_block_arg}{MomentumBlock}\" " +
                $"\"{position_kernel_arg}{PositionKernel}\" " +
                $"\"{position_grid_arg}{PositionGrid}\" " +
                $"\"{position_block_arg}{PositionBlock}\"";
        }
    }
}
