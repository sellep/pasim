using pasim.core;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace pasim.launcher
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            LoadModules();
        }

        private void LoadPreviousSettings()
        {

        }

        private void LoadModules()
        {
            _MomentumKernels.ItemsSource = KernelHelper.GetModulePaths(_KernelDirectory.Text, "momentum", true).Select(k => System.IO.Path.GetFileNameWithoutExtension(k));
            _MomentumKernels.SelectedIndex = 0;

            _MomentumGridDims.ItemsSource = KernelHelper.GridDims;
            _MomentumGridDims.SelectedIndex = 0;

            _MomentumBlockDims.ItemsSource = KernelHelper.BlockDims;
            _MomentumBlockDims.SelectedIndex = 0;

            _PositionKernels.ItemsSource = KernelHelper.GetModulePaths(_KernelDirectory.Text, "position", false).Select(k => System.IO.Path.GetFileNameWithoutExtension(k));
            _PositionKernels.SelectedIndex = 0;
        }

        private void _Detect_Click(object sender, RoutedEventArgs e)
        {
            string args = $"d={_KernelDirectory.Text} nonaive=true t=momentum n={1024 * 1} nolog=true";

            DetectionWindow win = new DetectionWindow(args);
            win.ShowDialog();
        }
    }
}
