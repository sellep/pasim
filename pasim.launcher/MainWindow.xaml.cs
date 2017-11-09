using ManagedCuda.VectorTypes;
using pasim.core;
using pasim.visual;
using System.Diagnostics;
using System.Linq;
using System.Windows;
using System.Windows.Controls;

namespace pasim.launcher
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {

        private PasimSetup _Setup = null;

        public MainWindow()
        {
            InitializeComponent();

            if (TryLoadSetup())
            {
                UpdateUI();
            }
            else
            {
                SetDefaults();
            }
        }

        private bool TryLoadSetup()
        {
            _Setup = ConfigurationHelper.Load();
            if (_Setup != null)
                return true;

            _Setup = new PasimSetup();
            return false;
        }

        private void SetDefaults()
        {
            _MomentumKernels.ItemsSource = KernelHelper.GetModulePaths(_KernelDirectory.Text, "momentum", true).Select(k => System.IO.Path.GetFileNameWithoutExtension(k));
            _MomentumKernels.SelectedIndex = 0;

            _MomentumGridDims.ItemsSource = KernelHelper.GridDims;
            _MomentumGridDims.SelectedIndex = 0;

            _MomentumBlockDims.ItemsSource = KernelHelper.BlockDims;
            _MomentumBlockDims.SelectedIndex = 0;

            _PositionKernels.ItemsSource = KernelHelper.GetModulePaths(_KernelDirectory.Text, "position", false).Select(k => System.IO.Path.GetFileNameWithoutExtension(k));
            _PositionKernels.SelectedIndex = 0;

            _PositionGridDims.ItemsSource = KernelHelper.GridDims;
            _PositionGridDims.SelectedIndex = 0;

            _PositionBlockDims.ItemsSource = KernelHelper.BlockDims;
            _PositionBlockDims.SelectedIndex = 0;
        }

        private void _MomentumDetect_Click(object sender, RoutedEventArgs e)
        {
            UpdateSetup();

            string args = $"d={_KernelDirectory.Text} nonaive=true t=momentum n={_Setup.N * _Setup.NMultiplier} nolog=true";

            DetectionWindow win = new DetectionWindow(args);
            win.ShowDialog();

            if (win.ExitCode != 0)
                return;

            ConfigurationHelper.Parse(win.LastLine, out string kernel, out dim3 gridDim, out dim3 blockDim);

            _Setup.MomentumKernel = kernel;
            _Setup.MomentumGrid = gridDim;
            _Setup.MomentumBlock = blockDim;

            UpdateUI();

            ConfigurationHelper.Save(_Setup);
        }

        private void _PositionDetect_Click(object sender, RoutedEventArgs e)
        {
            UpdateSetup();

            string args = $"d={_KernelDirectory.Text} t=position n={_Setup.N * _Setup.NMultiplier} nolog=true";

            DetectionWindow win = new DetectionWindow(args);
            win.ShowDialog();

            if (win.ExitCode != 0)
                return;

            ConfigurationHelper.Parse(win.LastLine, out string kernel, out dim3 gridDim, out dim3 blockDim);

            _Setup.PositionKernel = kernel;
            _Setup.PositionGrid = gridDim;
            _Setup.PositionBlock = blockDim;

            UpdateUI();

            ConfigurationHelper.Save(_Setup);
        }

        private void _Launch_Click(object sender, RoutedEventArgs e)
        {
            UpdateSetup();

            ConfigurationHelper.Save(_Setup);

            string args = _Setup.CreateCLArgs();
            Process.Start("pasim.visual.exe", args);

            Close();
        }

        private void UpdateUI()
        {
            _KernelDirectory.Text = _Setup.KernelDirectory;
            _SystemParticles.Text = _Setup.N.ToString();
            _ParticleMultiplier.SelectedItem = _Setup.NMultiplier.ToString();
            _SystemGravitationalConstant.Text = _Setup.G.ToString();
            _SystemInitPositionMax.Text = _Setup.InitPositionMax.ToString();
            _SystemInitMassMin.Text = _Setup.InitMassMin.ToString();
            _SystemInitMassMax.Text = _Setup.InitMassMax.ToString();
            _SystemInitMomentumMax.Text = _Setup.InitMomentumMax.ToString();
            _SystemDeltaTime.Text = _Setup.DT.ToString();
            _MomentumKernels.ItemsSource = KernelHelper.GetModulePaths(_Setup.KernelDirectory, "momentum", true).Select(k => System.IO.Path.GetFileNameWithoutExtension(k));
            _MomentumKernels.SelectedItem = _Setup.MomentumKernel;
            _MomentumGridDims.ItemsSource = KernelHelper.GridDims;
            _MomentumGridDims.SelectedItem = _Setup.MomentumGrid;
            _MomentumBlockDims.ItemsSource = KernelHelper.BlockDims;
            _MomentumBlockDims.SelectedItem = _Setup.MomentumBlock;
            _PositionKernels.ItemsSource = KernelHelper.GetModulePaths(_Setup.KernelDirectory, "position", false).Select(k => System.IO.Path.GetFileNameWithoutExtension(k));
            _PositionKernels.SelectedItem = _Setup.PositionKernel;
            _PositionGridDims.ItemsSource = KernelHelper.GridDims;
            _PositionGridDims.SelectedItem = _Setup.PositionGrid;
            _PositionBlockDims.ItemsSource = KernelHelper.BlockDims;
            _PositionBlockDims.SelectedItem = _Setup.PositionBlock;
        }

        private void UpdateSetup()
        {
            _Setup.KernelDirectory = _KernelDirectory.Text;
            _Setup.N = uint.Parse(_SystemParticles.Text);
            _Setup.NMultiplier = uint.Parse((_ParticleMultiplier.SelectedItem as ComboBoxItem).Content.ToString());
            _Setup.G = float.Parse(_SystemGravitationalConstant.Text);
            _Setup.InitPositionMax = float.Parse(_SystemInitPositionMax.Text);
            _Setup.InitMassMin = float.Parse(_SystemInitMassMin.Text);
            _Setup.InitMassMax = float.Parse(_SystemInitMassMax.Text);
            _Setup.InitMomentumMax = float.Parse(_SystemInitMomentumMax.Text);
            _Setup.DT = float.Parse(_SystemDeltaTime.Text);
            _Setup.MomentumKernel = _MomentumKernels.SelectedItem.ToString();
            _Setup.MomentumGrid = (dim3)_MomentumGridDims.SelectedItem;
            _Setup.MomentumBlock = (dim3)_MomentumBlockDims.SelectedItem;
            _Setup.PositionKernel = _PositionKernels.SelectedItem.ToString();
            _Setup.PositionGrid = (dim3)_PositionGridDims.SelectedItem;
            _Setup.PositionBlock = (dim3)_PositionBlockDims.SelectedItem;
        }
    }
}
