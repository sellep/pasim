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

            _RenderTarget.Content = control;
        }

        private void Control_OpenGLDraw(object sender, OpenGLEventArgs args)
        {
            OpenGL gl = args.OpenGL;

            gl.Clear(OpenGL.GL_COLOR_BUFFER_BIT | OpenGL.GL_DEPTH_BUFFER_BIT);
            gl.LoadIdentity();

            DrawParticles(gl);

            gl.Flush();
        }

        private void DrawCenterOfMass(OpenGL gl)
        {
            //Vector3 com = ParticleSystem.CenterOfMass(_Positions, _Masses);

            //gl.Begin(BeginMode.Lines);
            //gl.Color(0f, 0f, 1f);

            //gl.Vertex(-POSITION_MAX, com.y, 0);
            //gl.Vertex(+POSITION_MAX, com.y, 0);

            //gl.Vertex(com.x, -POSITION_MAX, 0);
            //gl.Vertex(com.x, +POSITION_MAX, 0);

            //gl.End();
        }

        private void DrawParticles(OpenGL gl)
        {
            gl.Begin(BeginMode.Points);
            gl.Color(1f, 0f, 0f);

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

        private void DrawMeshes(OpenGL gl)
        {
            //uint idx_l3 = _Mapping[_Selection];

            //Mesh l3 = _MeshesL3[idx_l3];

            //foreach (Mesh mesh in _MeshesL3.Where(m => m.parent == l3.parent))
            //{
            //    DrawL3Mesh(gl, mesh);
            //}

            //Mesh l2 = l3.parent;
            //foreach (Mesh mesh in _MeshesL2.Where(m => m.parent == l2.parent))
            //{
            //    DrawL2Mesh(gl, mesh);
            //}

            //foreach (Mesh mesh in _MeshesL1.Where(m => m != l2.parent))
            //{
            //    DrawL1Mesh(gl, mesh);
            //}
        }

        private void DrawL3Mesh(OpenGL gl, Mesh mesh)
        {
            //float x = mesh.x * MESH_L3_NODE_LENGTH - POSITION_MAX;
            //float y = mesh.y * MESH_L3_NODE_LENGTH - POSITION_MAX;

            //gl.Begin(BeginMode.LineLoop);

            //gl.Color(0f, 1f, 0f);

            //gl.Vertex(x, y, 0f);
            //gl.Vertex(x + MESH_L3_NODE_LENGTH, y, 0f);
            //gl.Vertex(x + MESH_L3_NODE_LENGTH, y + MESH_L3_NODE_LENGTH, 0f);
            //gl.Vertex(x, y + MESH_L3_NODE_LENGTH, 0f);

            //gl.End();
        }

        private void DrawL2Mesh(OpenGL gl, Mesh mesh)
        {
            //float x = mesh.x * MESH_L2_NODE_LENGTH - POSITION_MAX;
            //float y = mesh.y * MESH_L2_NODE_LENGTH - POSITION_MAX;

            //gl.Begin(BeginMode.LineLoop);

            //gl.Color(0f, 1f, 0f);

            //gl.Vertex(x, y, 0f);
            //gl.Vertex(x + MESH_L2_NODE_LENGTH, y, 0f);
            //gl.Vertex(x + MESH_L2_NODE_LENGTH, y + MESH_L2_NODE_LENGTH, 0f);
            //gl.Vertex(x, y + MESH_L2_NODE_LENGTH, 0f);

            //gl.End();
        }

        private void DrawL1Mesh(OpenGL gl, Mesh mesh)
        {
            float x = mesh.x * ParticleSystem.MESH_NODE_LENGTH - _System.CenterOfMass.x;
            float y = mesh.y * ParticleSystem.MESH_NODE_LENGTH - _System.CenterOfMass.y;

            gl.Begin(BeginMode.LineLoop);

            gl.Color(0f, 1f, 0f);

            gl.Vertex(x, y, 0f);
            gl.Vertex(x + ParticleSystem.MESH_NODE_LENGTH, y, 0f);
            gl.Vertex(x + ParticleSystem.MESH_NODE_LENGTH, y + ParticleSystem.MESH_NODE_LENGTH, 0f);
            gl.Vertex(x, y + ParticleSystem.MESH_NODE_LENGTH, 0f);

            gl.End();
        }

        private void Control_Resized(object sender, OpenGLEventArgs args)
        {
            //OpenGL gl = args.OpenGL;

            //gl.Viewport(0, 0, (int)gl.RenderContextProvider.Width, (int)gl.RenderContextProvider.Height);
            //gl.MatrixMode(OpenGL.GL_PROJECTION);
            //gl.LoadIdentity();

            ////gl.Perspective(90.0f, (float)gl.RenderContextProvider.Width / gl.RenderContextProvider.Height, 0.1, LENGTH);
            //gl.Ortho(-POSITION_MAX * 2, POSITION_MAX * 2, -POSITION_MAX * 2, POSITION_MAX * 2, 1, -1);

            //gl.MatrixMode(OpenGL.GL_MODELVIEW);
        }

        private void Control_OpenGLInitialized(object sender, OpenGLEventArgs args)
        {
            args.OpenGL.Enable(OpenGL.GL_DEPTH_TEST);
        }
    }
}
