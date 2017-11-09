using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.IO;
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

            for (int i = 0; i < e.Args.Length; i++)
            {
                File.AppendAllText("pasim.visual.log", e.Args[i] + Environment.NewLine);
            }

            PasimSetup setup = PasimSetup.Parse(e.Args);

            ConfigurationHelper.Save(setup, Path.GetTempFileName());

            MainWindow win = new MainWindow(setup);
            win.Show();
        }
    }
}
