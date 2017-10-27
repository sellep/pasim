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
        public uint x_rel;
        public uint y_rel;
        public float mass;
        public Vector3 center;
        public Mesh parent;

        public Mesh(uint x, uint y)
        {
            this.x = x;
            this.y = y;
        }

        public Mesh(Mesh parent, uint x, uint y, uint x_rel, uint y_rel)
        {
            this.parent = parent;
            this.x = x;
            this.y = y;
            this.x_rel = x_rel;
            this.y_rel = y_rel;
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
    }
}
