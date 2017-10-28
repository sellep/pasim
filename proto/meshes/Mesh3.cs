using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace meshes
{

    public struct Mesh3
    {
        public Mesh2 parent;
        public uint x;
        public uint y;

        public Vector3 centerOfMass;
        public float mass;

        public Mesh3(Mesh2 parent, uint x, uint y)
        {
            this.parent = parent;
            this.x = x;
            this.y = y;

            centerOfMass = new Vector3();
            mass = 0;
        }

        public void computeCenterOfMass(int[] mapping)
        {
            uint i;

            Vector3.zero(ref centerOfMass);
            mass = 0;

            for (i = 0; i < mapping.Length; i++)
            {

            }
        }
    }
}
