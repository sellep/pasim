using SharpGL;
using SharpGL.SceneGraph;
using SharpGL.WPF;
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

    public partial class MainWindow : Window
    {
        public const float VIEW_MAX = 500;

        private volatile bool _Invalidated = true;
        private uint[] _VertexBuffer = new uint[1];
        private bool _UseVertexArray = false;

        public MainWindow()
        {
            InitializeComponent();

            OpenGLControl control = new OpenGLControl();

            control.OpenGLInitialized += Control_OpenGLInitialized;
            control.OpenGLDraw += Control_OpenGLDraw;
            control.Resized += Control_Resized;
            //control.MouseLeftButtonDown += Control_MouseLeftButtonDown;

            _RenderTarget.Content = control;
        }

        private void DrawBodiesVertexArray(OpenGL gl)
        {

        }

        private void Control_OpenGLDraw(object sender, OpenGLEventArgs args)
        {
            if (!_Invalidated)
                return;

            _Invalidated = false;

            OpenGL gl = args.OpenGL;

            gl.Clear(OpenGL.GL_COLOR_BUFFER_BIT | OpenGL.GL_DEPTH_BUFFER_BIT);
            gl.LoadIdentity();

            if (_UseVertexArray)
            {
                DrawBodiesVertexArray(gl);
            }

            gl.Flush();
        }

        private void Control_Resized(object sender, OpenGLEventArgs args)
        {
            OpenGL gl = args.OpenGL;

            gl.Viewport(0, 0, gl.RenderContextProvider.Width, gl.RenderContextProvider.Height);
            gl.MatrixMode(OpenGL.GL_PROJECTION);
            gl.LoadIdentity();

            //gl.Perspective(90.0f, (float)gl.RenderContextProvider.Width / gl.RenderContextProvider.Height, 0.1, LENGTH);
            gl.Ortho(-VIEW_MAX, VIEW_MAX, -VIEW_MAX, VIEW_MAX, 1, -1);

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

            if (_VertexBuffer[0] == 0 && gl.IsExtensionFunctionSupported("glGenVertexArrays"))
            {
                gl.GenVertexArrays(1000, _VertexBuffer);

                _UseVertexArray = true;
            }
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
    }
}
