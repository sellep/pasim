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
        public MainWindow()
        {
            //https://stackoverflow.com/questions/14968529/drawing-an-opengl-scene-to-c-sharp-bitmap-off-screen-gets-clipped
            //


            InitializeComponent();

            ParticleSystem sys = new ParticleSystem(1000, 1, 1, 100, 0.5f);

            Pasim.Init(sys);

            Pasim.Tick(sys, 12);

            Pasim.Deinit(sys);
        }
    }
}
