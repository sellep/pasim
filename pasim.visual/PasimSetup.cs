using ManagedCuda.VectorTypes;
using pasim.core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.visual
{

    public class PasimSetup
    {

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

        public static PasimSetup Parse(string[] args)
        {
            const string n_arg = "n=";
            const string multiplier_arg = "nm=";
            const string momentum_kernel_arg = "mom_k=";
            const string momentum_grid_arg = "mom_g=";
            const string momentum_block_arg = "mom_b=";
            const string position_kernel_arg = "pos_k=";
            const string position_grid_arg = "pos_g=";
            const string position_block_arg = "pos_b=";
            const string g_arg = "g=";
            const string d_arg = "d=";
            const string pos_max_arg = "pos_max=";
            const string mass_min_arg = "mass_min=";
            const string mass_max_arg = "mass_max=";
            const string mom_max_arg = "mom_max=";
            const string dt_arg = "dt=";

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
    }
}
