using SharpGL;
using SharpGL.Enumerations;
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

namespace meshes
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private ParticleSystem _System = null;
        private uint _Selection;

        public MainWindow()
        {
            InitializeComponent();

            _System = new ParticleSystem(100);
            _System.Tick(12);

            _Selection = Rand.Next(_System.Count);

            OpenGLControl control = new OpenGLControl();

            control.OpenGLInitialized += Control_OpenGLInitialized;
            control.OpenGLDraw += Control_OpenGLDraw;
            control.Resized += Control_Resized;
            control.MouseLeftButtonDown += Control_MouseLeftButtonDown; ;

            _RenderTarget.Content = control;
        }

        private void Control_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            /*glGetDoublev(GL_MODELVIEW_MATRIX, modelview); //get the modelview info
            glGetDoublev(GL_PROJECTION_MATRIX, projection); //get the projection matrix info
            glGetIntegerv(GL_VIEWPORT, viewport); //get the viewport info*/

            OpenGL gl = (sender as OpenGLControl).OpenGL;

            double winx, winy;
            int[] viewport = new int[4];

            gl.GetInteger(OpenGL.GL_VIEWPORT, viewport);

            winx = e.GetPosition(sender as OpenGLControl).X;
            winy = viewport[3] - e.GetPosition(sender as OpenGLControl).Y;

            double[] worldc = gl.UnProject(winx, winy, 0);


        }

        private void Control_OpenGLDraw(object sender, OpenGLEventArgs args)
        {
            OpenGL gl = args.OpenGL;

            gl.Clear(OpenGL.GL_COLOR_BUFFER_BIT | OpenGL.GL_DEPTH_BUFFER_BIT);
            gl.LoadIdentity();

            gl.Translate(_System.CenterOfMass.x, _System.CenterOfMass.y, 0);

            DrawL1Mesh(gl);
            DrawCenterOfMass(gl);
            DrawParticles(gl);

            gl.Flush();
        }

        private void DrawParticles(OpenGL gl)
        {
            gl.Color(1f, 0f, 0f);
            gl.PointSize(1.5f);

            gl.Begin(BeginMode.Points);

            for (uint i = 0; i < _System.Count; i++)
            {
                if (i != _Selection)
                {
                    gl.Vertex(_System.Positions[i].x, _System.Positions[i].y, 0);
                }
            }

            gl.Color(1f, 1f, 1f);
            gl.Vertex(_System.Positions[_Selection].x, _System.Positions[_Selection].y, 0);

            gl.End();
        }

        private void DrawCenterOfMass(OpenGL gl)
        {
            gl.Color(1f, 1f, 0f, 0.7f);
            gl.PointSize(3f);

            gl.Begin(BeginMode.Points);

            gl.Vertex(_System.CenterOfMass.x, _System.CenterOfMass.y, 0);

            gl.End();
        }

        private void DrawL1Mesh(OpenGL gl)
        {
            float x, y;
            uint x1, y1;

            gl.Color(0f, 1f, 0f, 0.5f);
            gl.LineWidth(1f);

            for (y1 = 0; y1 < ParticleSystem.MESH_LENGTH; y1++)
            {
                for (x1 = 0; x1 < ParticleSystem.MESH_LENGTH; x1++)
                {
                    x = _System.MeshesL1[x1, y1].x * ParticleSystem.MESH_NODE_L1_LENGTH - ParticleSystem.MESH_WIDTH / 2 + _System.CenterOfMass.x;
                    y = _System.MeshesL1[x1, y1].y * ParticleSystem.MESH_NODE_L1_LENGTH - ParticleSystem.MESH_WIDTH / 2 + _System.CenterOfMass.y;

                    gl.Begin(BeginMode.LineStrip);

                    if (x1 == 0)
                    {
                        gl.Vertex(x, y, 0f);
                    }

                    gl.Vertex(x, y + ParticleSystem.MESH_NODE_L1_LENGTH, 0f);
                    gl.Vertex(x + ParticleSystem.MESH_NODE_L1_LENGTH, y + ParticleSystem.MESH_NODE_L1_LENGTH, 0f);
                    gl.Vertex(x + ParticleSystem.MESH_NODE_L1_LENGTH, y, 0f);

                    if (y1 == 0)
                    {
                        gl.Vertex(x, y, 0f);
                    }

                    gl.End();
                }
            }
        }

        private void Control_Resized(object sender, OpenGLEventArgs args)
        {
            OpenGL gl = args.OpenGL;

            gl.Viewport(0, 0, (int)gl.RenderContextProvider.Width, (int)gl.RenderContextProvider.Height);
            gl.MatrixMode(OpenGL.GL_PROJECTION);
            gl.LoadIdentity();

            //gl.Perspective(90.0f, (float)gl.RenderContextProvider.Width / gl.RenderContextProvider.Height, 0.1, LENGTH);
            gl.Ortho(-ParticleSystem.MESH_WIDTH, ParticleSystem.MESH_WIDTH, -ParticleSystem.MESH_WIDTH, ParticleSystem.MESH_WIDTH, 1, -1);

            gl.MatrixMode(OpenGL.GL_MODELVIEW);
        }

        private void Control_OpenGLInitialized(object sender, OpenGLEventArgs args)
        {
            OpenGL gl = args.OpenGL;
            gl.Enable(OpenGL.GL_DEPTH_TEST);
            gl.Enable(OpenGL.GL_LINE_SMOOTH);
            gl.Enable(OpenGL.GL_POINT_SMOOTH);
            gl.Enable(OpenGL.GL_BLEND);
            gl.Hint(OpenGL.GL_LINE_SMOOTH_HINT, OpenGL.GL_NICEST);
            gl.Hint(OpenGL.GL_POINT_SMOOTH_HINT, OpenGL.GL_NICEST);

            gl.BlendFunc(OpenGL.GL_SRC_ALPHA, OpenGL.GL_ONE_MINUS_SRC_ALPHA);
        }
    }
}
