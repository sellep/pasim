using pasim.net;
using SharpGL;
using SharpGL.Enumerations;
using SharpGL.SceneGraph;
using SharpGL.WPF;
using System;
using System.Collections.Generic;
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
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.ComponentModel;

namespace pasim.visual
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private const float _POSITION_MAX = 100;

        private ParticleSystem _System = new ParticleSystem(10000, 1, 1, _POSITION_MAX, 0.5f);
        private Thread _PhysicsWorker = null;
        private volatile bool _UpdateAvailable = false;
        private volatile bool _Terminate = false;
        private volatile uint _PhysicFrames = 0;

        private static void AssertStatus(Func<CudaStatus> a)
        {
            CudaStatus status = a();
            if (status != CudaStatus.cudaSuccess)
                throw new Exception($"{status.ToString()}: {Pasim.GetErrorString(status)}");
        }

        public MainWindow()
        {
            InitializeComponent();

            OpenGLControl control = new OpenGLControl();

            control.OpenGLInitialized += Control_OpenGLInitialized;
            control.OpenGLDraw += Control_OpenGLDraw;
            control.Resized += Control_Resized;
            //control.MouseLeftButtonDown += (s, e) => _DoPhysics = !_DoPhysics;

            _PhysicsWorker = new Thread(PhysicsThread);
            _PhysicsWorker.Start();

            _RenderTarget.Content = control;
        }

        private void Control_OpenGLDraw(object sender, OpenGLEventArgs args)
        {
            if (!_UpdateAvailable)
                return;

            IEnumerable<Vector3> positions = _System.Positions.Select(r => r).ToArray();

            _Info.Text = $"Physic frames: {_PhysicFrames}";

            OpenGL gl = args.OpenGL;

            gl.Clear(OpenGL.GL_COLOR_BUFFER_BIT | OpenGL.GL_DEPTH_BUFFER_BIT);
            gl.LoadIdentity();

            gl.Translate(-_POSITION_MAX/2, -_POSITION_MAX/2, -_POSITION_MAX);

            gl.Begin(BeginMode.Points);

            gl.Color(0f, 1f, 0f);

            foreach (Vector3 r in positions)
            {
                gl.Vertex(r.x, r.y, r.z);
            }

            gl.End();
            gl.Flush();
        }

        private void PhysicsThread()
        {
            AssertStatus(() => Pasim.Init(_System));

            while (!_Terminate)
            {
                AssertStatus(() => Pasim.Tick(_System, 0.1f));
                AssertStatus(() => Pasim.Update(_System));
                _PhysicFrames++;
                _UpdateAvailable = true;
            }

            Pasim.Deinit(_System);
        }

        private void Control_Resized(object sender, OpenGLEventArgs args)
        {
            OpenGL gl = args.OpenGL;

            gl.MatrixMode(OpenGL.GL_PROJECTION);
            gl.LoadIdentity();

            gl.Perspective(45.0f, (float)gl.RenderContextProvider.Width / gl.RenderContextProvider.Height, 0.1f, 100.0f);

            gl.MatrixMode(OpenGL.GL_MODELVIEW);
        }

        private void Control_OpenGLInitialized(object sender, OpenGLEventArgs args)
        {
            args.OpenGL.Enable(OpenGL.GL_DEPTH_TEST);
        }

        protected override void OnClosing(CancelEventArgs e)
        {
            base.OnClosing(e);

            _Terminate = true;
        }
    }
}
