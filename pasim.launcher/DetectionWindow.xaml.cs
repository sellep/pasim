using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace pasim.launcher
{
    /// <summary>
    /// Interaction logic for DetectionWindow.xaml
    /// </summary>
    public partial class DetectionWindow : Window
    {
        private string _Args;
        private bool _AllowClose = false;

        public string LastLine { get; private set; }

        public int ExitCode { get; private set; }

        public DetectionWindow(string args)
        {
            InitializeComponent();

            Closing += (s, e) =>
            {
                if (!_AllowClose)
                {
                    e.Cancel = true;
                }
            };

            Title += " " + args;

            _Args = args;

            Thread t = new Thread(Run);
            t.Start();
        }

        private void Run()
        {
            ProcessStartInfo psi = new ProcessStartInfo("pasim.test.exe", _Args);
            psi.UseShellExecute = false;
            psi.RedirectStandardOutput = true;
            psi.CreateNoWindow = true;

            Process p = new Process();
            p.StartInfo = psi;

            p.OutputDataReceived += (s, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                {
                    LastLine = e.Data;
                }

                Dispatcher.BeginInvoke(new Action(() =>
                {
                    _Output.Text += $"{e.Data}{Environment.NewLine}";
                    _Viewer.ScrollToBottom();
                }));
            };

            p.Start();
            p.BeginOutputReadLine();

            p.WaitForExit();

            ExitCode = p.ExitCode;
            _AllowClose = true;
            Dispatcher.BeginInvoke(new Action(() => Close()));
        }
    }
}
