using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace meshes
{

    public class Mesh
    {
        public uint x;
        public uint y;
        public float mass;
        public Vector3 center;
        public Mesh parent;

        public Mesh(uint x, uint y)
        {
            this.x = x;
            this.y = y;
        }

        public Mesh(uint x, uint y, Mesh parent)
        {
            this.x = x;
            this.y = y;
            this.parent = parent;
        }

        public void Reset()
        {
            Vector3.Zero(ref center);
            mass = 0;
        }

        public void AddParticle(ref Vector3 position, float mass)
        {
            center += position * mass;
            this.mass += mass;
        }

        //public static void CreateMeshes(Mesh[] l1, Mesh[] l2, Mesh[] l3)
        //{
        //    Mesh mesh;

        //    for (uint y = 0; y < MainWindow.MESH_L1_LENGTH; y++)
        //    {
        //        for (uint x = 0; x < MainWindow.MESH_L1_LENGTH; x++)
        //        {
        //            mesh = new Mesh(x, y);
        //            l1[mesh.y * MainWindow.MESH_L1_LENGTH + mesh.x] = mesh;
        //            CreateL2Meshes(mesh, l2, l3);
        //        }
        //    }
        //}

        //private static void CreateL2Meshes(Mesh parent, Mesh[] l2, Mesh[] l3)
        //{
        //    Mesh mesh;

        //    for (uint y = 0; y < MainWindow.MESH_L2_LENGTH; y++)
        //    {
        //        for (uint x = 0; x < MainWindow.MESH_L2_LENGTH; x++)
        //        {
        //            mesh = new Mesh(parent.x * MainWindow.MESH_L2_LENGTH + x, parent.y * MainWindow.MESH_L2_LENGTH + y, parent);
        //            l2[mesh.y * MainWindow.MESH_L2_LENGTH * MainWindow.MESH_L1_LENGTH + mesh.x] = mesh;
        //            CreateL3Meshes(mesh, l3);
        //        }
        //    }
        //}

        //private static void CreateL3Meshes(Mesh parent, Mesh[] l3)
        //{
        //    Mesh mesh;

        //    for (uint y = 0; y < MainWindow.MESH_L3_LENGTH; y++)
        //    {
        //        for (uint x = 0; x < MainWindow.MESH_L3_LENGTH; x++)
        //        {
        //            mesh = new Mesh(parent.x * MainWindow.MESH_L3_LENGTH + x, parent.y * MainWindow.MESH_L3_LENGTH + y, parent);
        //            l3[mesh.y * MainWindow.MESH_L3_LENGTH * MainWindow.MESH_L2_LENGTH * MainWindow.MESH_L1_LENGTH + mesh.x] = mesh;
        //        }
        //    }
        //}
    }
}
