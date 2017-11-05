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
using pasim.core;
using pasim.math;

namespace meshes
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public static float SELECTION_RANGE_MAX = 5f;


        private SimpleParticleSystem _System = null;
        private uint? _Selection = null;

        private Thread _PhysixThread = null;
        private volatile bool _Terminate = false;

        public MainWindow()
        {
            InitializeComponent();

            _System = new SimpleParticleSystem(100000);
            //_System.positions[0] = new Vector3(-50, 0, 0);
            //_System.positions[1] = new Vector3(50, 0, 0);
            //_System.momentums[0] = new Vector3(0, 10f, 0);
            //_System.momentums[1] = new Vector3(0, -10f, 0);
            _PhysixThread = new Thread(Computation);
            Loaded += (s,e) => _PhysixThread.Start();


            OpenGLControl control = new OpenGLControl();

            control.OpenGLInitialized += Control_OpenGLInitialized;
            control.OpenGLDraw += Control_OpenGLDraw;
            control.Resized += Control_Resized;
            control.MouseLeftButtonDown += Control_MouseLeftButtonDown; ;

            _RenderTarget.Content = control;
        }

        private void Computation()
        {
            DateTime start, end;
            int frames = 0;
            float averageTime = 0;

            while (!_Terminate)
            {
                start = DateTime.Now;

                _System.Tick(0.2f);

                end = DateTime.Now;

                averageTime = averageTime * frames + (float)(end - start).TotalMilliseconds;
                frames++;
                averageTime /= frames;

                string info = $"Frame: {frames}{Environment.NewLine}Average time: {averageTime}";

                Dispatcher.BeginInvoke(new Action(() => _Info.Text = info));
            }
        }

        protected override void OnClosing(CancelEventArgs e)
        {
            _Terminate = true;
            _PhysixThread.Join();

            base.OnClosing(e);
        }

        private void Control_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
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
        }

        private void Control_OpenGLDraw(object sender, OpenGLEventArgs args)
        {
            OpenGL gl = args.OpenGL;

            gl.Clear(OpenGL.GL_COLOR_BUFFER_BIT | OpenGL.GL_DEPTH_BUFFER_BIT);
            gl.LoadIdentity();

            //gl.Translate(_System.CenterOfMass.x, _System.CenterOfMass.y, 0);

            //DrawL1Mesh(gl);
            DrawParticles(gl);

            gl.Flush();
        }

        private void DrawParticles(OpenGL gl, bool usePoints = true)
        {
            IEnumerable<Vector2> positions = null;

            positions = new List<Vector2>(_System.positions);

            gl.Color(0f, 0.65f, 1f);

            if (usePoints)
            {
                gl.PointSize(3f);

                gl.Begin(BeginMode.Points);
                gl.Color(0f, 0.65f, 1f);

                for (uint i = 0; i < _System.count; i++)
                {
                    if (_Selection.HasValue && i == _Selection.Value)
                        continue;

                    gl.Vertex(_System.positions[i].x, _System.positions[i].y, 0);
                }

                gl.End();
            }
            else
            {
                foreach (Vector2 position in positions)
                {
                    gl.Begin(BeginMode.LineLoop);
                    gl.Vertex(position.x, position.y, 0);
                    gl.Vertex(position.x, position.y + 1, 0);
                    gl.Vertex(position.x + 1, position.y + 1, 0);
                    gl.Vertex(position.x, position.y + 1, 0);
                    gl.End();
                }
            }
            


            
            /*gl.PointSize(3f);

            gl.Begin(BeginMode.Points);
            gl.Color(0f, 0.65f, 1f);
            
            for (uint i = 0; i < _System.count; i++)
            {
                if (_Selection.HasValue && i == _Selection.Value)
                    continue;

                gl.Vertex(_System.positions[i].x, _System.positions[i].y, 0);
            }

            if (_Selection.HasValue)
            {
                gl.Color(1f, 1f, 1f);
                gl.Vertex(_System.positions[_Selection.Value].x, _System.positions[_Selection.Value].y, 0);
            }*/

            /*// Create a Vector Buffer Object that will store the vertices on video memory
2 	GLuint vbo;
3 	glGenBuffers(1, &vbo);
4 
5 	// Allocate space and upload the data from CPU to GPU
6 	glBindBuffer(GL_ARRAY_BUFFER, vbo);
7 	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_position), vertices_position, GL_STATIC_DRAW);
*/

            gl.End();
        }

        private void Control_Resized(object sender, OpenGLEventArgs args)
        {
            OpenGL gl = args.OpenGL;

            gl.Viewport(0, 0, (int)gl.RenderContextProvider.Width, (int)gl.RenderContextProvider.Height);
            gl.MatrixMode(OpenGL.GL_PROJECTION);
            gl.LoadIdentity();

            //gl.Perspective(90.0f, (float)gl.RenderContextProvider.Width / gl.RenderContextProvider.Height, 0.1, LENGTH);
            gl.Ortho(-SimpleParticleSystem.POSITION_MAX * 1.5, SimpleParticleSystem.POSITION_MAX * 1.5, -SimpleParticleSystem.POSITION_MAX * 1.5, SimpleParticleSystem.POSITION_MAX * 1.5, 1, -1);

            gl.MatrixMode(OpenGL.GL_MODELVIEW);
        }

        private void Control_OpenGLInitialized(object sender, OpenGLEventArgs args)
        {
            OpenGL gl = args.OpenGL;
            //gl.Enable(OpenGL.GL_DEPTH_TEST);
            gl.Enable(OpenGL.GL_LINE_SMOOTH);
            gl.Enable(OpenGL.GL_POINT_SMOOTH);
            gl.Enable(OpenGL.GL_PROGRAM_POINT_SIZE);
            gl.Enable(OpenGL.GL_BLEND);
            gl.Hint(OpenGL.GL_LINE_SMOOTH_HINT, OpenGL.GL_NICEST);
            gl.Hint(OpenGL.GL_POINT_SMOOTH_HINT, OpenGL.GL_NICEST);

            gl.BlendFunc(OpenGL.GL_SRC_ALPHA, OpenGL.GL_ONE_MINUS_SRC_ALPHA);
        }
    }
}
