using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace meshes
{

    public struct Mesh1
    {
        public uint x;
        public uint y;

        public Vector3 centerOfMass;
        public float mass;

        public Mesh parent;
        public Mesh2[,] meshes2;

        public Mesh1(Mesh parent, uint x, uint y)
        {
            uint sub_x, sub_y;

            this.parent = parent;
            this.x = x;
            this.y = y;

            centerOfMass = new Vector3();
            mass = 0;

            meshes2 = new Mesh2[Mesh.MESH_2_LENGTH, Mesh.MESH_2_LENGTH];

            for (sub_y = 0; sub_y < Mesh.MESH_2_LENGTH; sub_y++)
            {
                for (sub_x = 0; sub_x < Mesh.MESH_2_LENGTH; sub_x++)
                {
                    meshes2[sub_x, sub_y] = new Mesh2(this, sub_x, sub_y);
                }
            }
        }

        public void computeCenterOfMass(int[] mapping)
        {
            uint ix, iy;
            Mesh2 sub;

            Vector3.zero(ref centerOfMass);
            mass = 0;

            for (ix = 0; ix < Mesh.MESH_2_LENGTH; ix++)
            {
                for (iy = 0; iy < Mesh.MESH_2_LENGTH; iy++)
                {
                    sub = meshes2[ix, iy];
                    sub.computeCenterOfMass(mapping);

                    centerOfMass += sub.centerOfMass * sub.mass;
                    mass += sub.mass;
                }
            }

            centerOfMass /= mass;
        }
    }
}
