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
        public static float SELECTION_RANGE_MAX = 5f;

        private ParticleSystem _System = null;
        private uint? _Selection = null;

        public MainWindow()
        {
            InitializeComponent();

            _System = new ParticleSystem(10000);
            _System.Tick(12);

            OpenGLControl control = new OpenGLControl();

            control.OpenGLInitialized += Control_OpenGLInitialized;
            control.OpenGLDraw += Control_OpenGLDraw;
            control.Resized += Control_Resized;
            control.MouseLeftButtonDown += Control_MouseLeftButtonDown; ;

            _RenderTarget.Content = control;
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

            float min = Vector3.Distance(mouse, _System.Positions[0]), current;
            _Selection = 0;

            for (uint i = 1; i < _System.Count; i++)
            {
                current = Vector3.Distance(mouse, _System.Positions[i]);
                if (current < min)
                {
                    min = current;
                    _Selection = i;
                }
            }

            if (min >= SELECTION_RANGE_MAX)
            {
                _Selection = null;
            }
        }

        private void Control_OpenGLDraw(object sender, OpenGLEventArgs args)
        {
            OpenGL gl = args.OpenGL;

            gl.Clear(OpenGL.GL_COLOR_BUFFER_BIT | OpenGL.GL_DEPTH_BUFFER_BIT);
            gl.LoadIdentity();

            gl.Translate(_System.CenterOfMass.x, _System.CenterOfMass.y, 0);

            //DrawL1Mesh(gl);
            DrawParticles(gl);

            gl.Flush();
        }

        private void DrawParticles(OpenGL gl)
        {
            gl.PointSize(3f);

            gl.Begin(BeginMode.Points);
            gl.Color(0f, 0.65f, 1f);
            
            for (uint i = 0; i < _System.Count; i++)
            {
                if (_Selection.HasValue && i == _Selection.Value)
                    continue;

                gl.Vertex(_System.Positions[i].x, _System.Positions[i].y, 0);
            }

            if (_Selection.HasValue)
            {
                gl.Color(1f, 1f, 1f);
                gl.Vertex(_System.Positions[_Selection.Value].x, _System.Positions[_Selection.Value].y, 0);
            }

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

        private void DrawCenterOfMass(OpenGL gl)
        {
            gl.Color(1f, 1f, 0f, 0.7f);
            gl.PointSize(2f);

            gl.Begin(BeginMode.Points);

            gl.Vertex(_System.CenterOfMass.x, _System.CenterOfMass.y, 0);

            gl.End();
        }

        //private void DrawL1Mesh(OpenGL gl)
        //{
        //    float x, y;
        //    uint x1, y1;

        //    gl.Color(0f, 1f, 0f, 0.5f);
        //    gl.LineWidth(10f);

        //    for (y1 = 0; y1 < ParticleSystem.MESH_LENGTH; y1++)
        //    {
        //        for (x1 = 0; x1 < ParticleSystem.MESH_LENGTH; x1++)
        //        {
        //            x = _System.MeshesL1[x1, y1].x * ParticleSystem.MESH_NODE_L1_LENGTH - ParticleSystem.MESH_WIDTH / 2 + _System.CenterOfMass.x;
        //            y = _System.MeshesL1[x1, y1].y * ParticleSystem.MESH_NODE_L1_LENGTH - ParticleSystem.MESH_WIDTH / 2 + _System.CenterOfMass.y;

        //            gl.Begin(BeginMode.LineStrip);

        //            if (x1 == 0)
        //            {
        //                gl.Vertex(x, y, 0f);
        //            }

        //            gl.Vertex(x, y + ParticleSystem.MESH_NODE_L1_LENGTH, 0f);
        //            gl.Vertex(x + ParticleSystem.MESH_NODE_L1_LENGTH, y + ParticleSystem.MESH_NODE_L1_LENGTH, 0f);
        //            gl.Vertex(x + ParticleSystem.MESH_NODE_L1_LENGTH, y, 0f);

        //            if (y1 == 0)
        //            {
        //                gl.Vertex(x, y, 0f);
        //            }

        //            gl.End();
        //        }
        //    }
        //}

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
