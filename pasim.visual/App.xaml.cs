using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;

namespace pasim.visual
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {


        protected override void OnStartup(StartupEventArgs e)
        {
            if (e.Args.Length == 0)
            {
                MessageBox.Show("Use pasim.launcher.exe to setup simulation conditions", "pasim", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            PasimSetup setup = PasimSetup.Parse(e.Args);

            MainWindow win = new MainWindow(setup);
            win.Show();
        }
    }
}
