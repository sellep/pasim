﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace meshes
{

    public struct Mesh
    {
        public const uint MESH_1_LENGTH = 7;
        public const uint MESH_2_LENGTH = 3;
        public const uint MESH_3_LENGTH = 3;

        public Vector3 centerOfMass;
        public float mass;

        public Mesh1[,] meshes1;

        public Mesh(uint length)
        {
            uint sub_x, sub_y;

            centerOfMass = new Vector3();
            mass = 0;

            meshes1 = new Mesh1[MESH_1_LENGTH, MESH_1_LENGTH];

            for (sub_y = 0; sub_y < MESH_1_LENGTH; sub_y++)
            {
                for (sub_x = 0; sub_x < MESH_1_LENGTH; sub_x++)
                {
                    meshes1[sub_x, sub_y] = new Mesh1(this, sub_x, sub_y);
                }
            }
        }

        public void computeCenterOfMass(int[] mapping)
        {
            uint ix, iy;
            Mesh1 sub;

            Vector3.zero(ref centerOfMass);
            mass = 0;

            for (ix = 0; ix < MESH_1_LENGTH; ix++)
            {
                for (iy = 0; iy < MESH_1_LENGTH; iy++)
                {
                    sub = meshes1[ix, iy];
                    sub.computeCenterOfMass(mapping);

                    centerOfMass += sub.centerOfMass * sub.mass;
                    mass += sub.mass;
                }
            }

            centerOfMass /= mass;
        }
    }
}
