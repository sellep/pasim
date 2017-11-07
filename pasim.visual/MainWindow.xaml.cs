using ManagedCuda.VectorTypes;
using pasim.core;
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

    public partial class MainWindow : Window
    {
        public const float VIEW_MAX = 500;
        public const float POSITION_MAX = 100;
        public const float DT = 0.5f;

        private float _RotateY = 0;

        private float4[] _Bodies = null;

        private readonly object _Sync = new object();
        private volatile bool _Terminate = false;
        private Thread _PhysicsThread = null;

        public MainWindow()
        {
            InitializeComponent();

            OpenGLControl control = new OpenGLControl();

            control.OpenGLInitialized += Control_OpenGLInitialized;
            control.OpenGLDraw += Control_OpenGLDraw;
            control.Resized += Control_Resized;

            _RenderTarget.Content = control;

            _Bodies = ParticleSystem.InitializeBodies(1024 * 4, POSITION_MAX, 0.5f, 1f);

            _PhysicsThread = new Thread(() =>
            {
                ParticleSystem system = new ParticleSystem(_Bodies, ParticleSystem.InitializeMomentums(1024 * 4, 0.1f));

                system.SetMomentumKernel(@"C:\git\pasim\pasim.gpu\x64\Debug\kernel_momentum_shmem_b7_u8.ptx", new dim3(16, 1, 1), new dim3(256, 1, 1));
                system.SetPositionKernel(@"C:\git\pasim\pasim.gpu\x64\Debug\kernel_position_naive.ptx", new dim3(16, 1, 1), new dim3(256, 1, 1));

                while (!_Terminate)
                {
                    system.Tick(DT);

                    lock (_Sync)
                    {
                        system.Synchronize(_Bodies);
                    }
                }

                system.Dispose();
            });

            _PhysicsThread.Start();
        }

        private void Control_OpenGLDraw(object sender, OpenGLEventArgs args)
        {
            OpenGL gl = args.OpenGL;

            gl.Clear(OpenGL.GL_COLOR_BUFFER_BIT | OpenGL.GL_DEPTH_BUFFER_BIT);
            gl.LoadIdentity();

            gl.PushMatrix();

            gl.Translate(0, 0, -300);
            gl.Rotate(0f, _RotateY, 0f);

            //draw particles
            lock (_Sync)
            {
                gl.Color(0f, 0.65f, 1f);
                gl.Begin(BeginMode.Points);

                for (uint i = 0; i < _Bodies.Length; i++)
                {
                    gl.Vertex(_Bodies[i].x, _Bodies[i].y, _Bodies[i].z);
                }

                gl.End();
            }

            //draw boundary
            //gl.Color(0f, 1f, 0f);

            //gl.Begin(BeginMode.LineLoop);
            //gl.Vertex(-POSITION_MAX, -POSITION_MAX, POSITION_MAX);
            //gl.Vertex(POSITION_MAX, -POSITION_MAX, POSITION_MAX);
            //gl.Vertex(POSITION_MAX, POSITION_MAX, POSITION_MAX);
            //gl.Vertex(-POSITION_MAX, POSITION_MAX, POSITION_MAX);
            //gl.End();

            //gl.Begin(BeginMode.LineLoop);
            //gl.Vertex(-POSITION_MAX, -POSITION_MAX, -POSITION_MAX);
            //gl.Vertex(POSITION_MAX, -POSITION_MAX, -POSITION_MAX);
            //gl.Vertex(POSITION_MAX, POSITION_MAX, -POSITION_MAX);
            //gl.Vertex(-POSITION_MAX, POSITION_MAX, -POSITION_MAX);
            //gl.End();

            //gl.Begin(BeginMode.Lines);
            //gl.Vertex(-POSITION_MAX, -POSITION_MAX, -POSITION_MAX);
            //gl.Vertex(-POSITION_MAX, -POSITION_MAX, POSITION_MAX);
            //gl.Vertex(POSITION_MAX, -POSITION_MAX, -POSITION_MAX);
            //gl.Vertex(POSITION_MAX, -POSITION_MAX, POSITION_MAX);
            //gl.Vertex(-POSITION_MAX, POSITION_MAX, -POSITION_MAX);
            //gl.Vertex(-POSITION_MAX, POSITION_MAX, POSITION_MAX);
            //gl.Vertex(POSITION_MAX, POSITION_MAX, -POSITION_MAX);
            //gl.Vertex(POSITION_MAX, POSITION_MAX, POSITION_MAX);
            //gl.End();

            gl.PopMatrix();

            gl.Flush();

            _RotateY += 0.01f;
        }

        private void Control_Resized(object sender, OpenGLEventArgs args)
        {
            OpenGL gl = args.OpenGL;

            gl.Viewport(0, 0, gl.RenderContextProvider.Width, gl.RenderContextProvider.Height);
            gl.MatrixMode(OpenGL.GL_PROJECTION);
            gl.LoadIdentity();

            gl.Perspective(90.0f, (float)gl.RenderContextProvider.Width / gl.RenderContextProvider.Height, 0.1, VIEW_MAX);
            //gl.Ortho(-VIEW_MAX, VIEW_MAX, -VIEW_MAX, VIEW_MAX, 1, -1);

            gl.MatrixMode(OpenGL.GL_MODELVIEW);
        }

        private void Control_OpenGLInitialized(object sender, OpenGLEventArgs args)
        {
            OpenGL gl = args.OpenGL;
            gl.Enable(OpenGL.GL_DEPTH_TEST);
            gl.Enable(OpenGL.GL_LINE_SMOOTH);
            gl.Enable(OpenGL.GL_POINT_SMOOTH);
            gl.Enable(OpenGL.GL_PROGRAM_POINT_SIZE);
            gl.Enable(OpenGL.GL_BLEND);
            gl.Hint(OpenGL.GL_LINE_SMOOTH_HINT, OpenGL.GL_NICEST);
            gl.Hint(OpenGL.GL_POINT_SMOOTH_HINT, OpenGL.GL_NICEST);

            gl.BlendFunc(OpenGL.GL_SRC_ALPHA, OpenGL.GL_ONE_MINUS_SRC_ALPHA);
        }

        /*private void Control_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            OpenGL gl = (sender as OpenGLControl).OpenGL;

            double winx, winy;
            int[] viewport = new int[4];

            gl.GetInteger(OpenGL.GL_VIEWPORT, viewport);

            winx = e.GetPosition(sender as OpenGLControl).X;
            winy = viewport[3] - e.GetPosition(sender as OpenGLControl).Y;

            double[] worldc = gl.UnProject(winx, winy, 0);

            Vector3 mouse = new Vector3((float) worldc[0], (float)worldc[1], 0);

            //float min = Vector3.distance(mouse, _System.Positions[0]), current;
            //_Selection = 0;

            //for (uint i = 1; i < _System.Count; i++)
            //{
            //    current = Vector3.distance(mouse, _System.Positions[i]);
            //    if (current < min)
            //    {
            //        min = current;
            //        _Selection = i;
            //    }
            //}

            //if (min >= SELECTION_RANGE_MAX)
            //{
            //    _Selection = null;
            //}
        }*/

        protected override void OnClosing(CancelEventArgs e)
        {
            _Terminate = true;

            base.OnClosing(e);
        }
    }
}
