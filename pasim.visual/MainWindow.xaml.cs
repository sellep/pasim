using pasim.net;
using System;
using System.Collections.Generic;
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

namespace pasim.visual
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {

        private static void AssertStatus(Func<CudaStatus> a)
        {
            CudaStatus status = a();
            if (status != CudaStatus.cudaSuccess)
                throw new Exception($"{status.ToString()}: {Pasim.GetErrorString(status)}");
        }

        public MainWindow()
        {
            InitializeComponent();

            ParticleSystem sys = new ParticleSystem(100000, 1, 1, 100, 0.5f);

            AssertStatus(() => Pasim.Init(sys));

            AssertStatus(() => Pasim.Tick(sys, 12));

            AssertStatus(() => Pasim.Update(sys));

            Pasim.Deinit(sys);
        }
    }
}
