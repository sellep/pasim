using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace meshes
{

    public struct Mesh2
    {
        public uint x;
        public uint y;

        public Vector3 centerOfMass;
        public float mass;

        public Mesh1 parent;
        public Mesh3[,] meshes3;

        public Mesh2(Mesh1 parent, uint x, uint y)
        {
            uint sub_x, sub_y;

            this.parent = parent;
            this.x = x;
            this.y = y;

            centerOfMass = new Vector3();
            mass = 0;

            meshes3 = new Mesh3[Mesh.MESH_3_LENGTH, Mesh.MESH_3_LENGTH];
            for (sub_y = 0; sub_y < Mesh.MESH_3_LENGTH; sub_y++)
            {
                for (sub_x = 0; sub_x < Mesh.MESH_3_LENGTH; sub_x++)
                {
                    meshes3[sub_x, sub_y] = new Mesh3(this, sub_x, sub_y);
                }
            }
        }

        public void computeCenterOfMass(int[] mapping)
        {
            uint ix, iy;
            Mesh3 sub;

            Vector3.zero(ref centerOfMass);
            mass = 0;

            for (ix = 0; ix < Mesh.MESH_3_LENGTH; ix++)
            {
                for (iy = 0; iy < Mesh.MESH_3_LENGTH; iy++)
                {
                    sub = meshes3[ix, iy];
                    sub.computeCenterOfMass(mapping);

                    centerOfMass += sub.centerOfMass * sub.mass;
                    mass += sub.mass;
                }
            }

            centerOfMass /= mass;

        }
    }
}
