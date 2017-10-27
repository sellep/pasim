using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace meshes
{
    public class ParticleSystem
    {
        public const float POSITION_MAX = 100;
        public const uint MESH_LENGTH = 3 * 3 * 9;
        public const float MESH_NODE_LENGTH = (2 * POSITION_MAX) / MESH_LENGTH;
        public const float MESH_WIDTH = MESH_LENGTH * MESH_NODE_LENGTH;

        public uint Count;

        public Vector3[] Positions;

        public float[] Masses;
        public Mesh[] Meshes;

        public Vector3 CenterOfMass;

        public int[] Mapping;

        public ParticleSystem(uint count)
        {
            Count = count;
            Positions = new Vector3[count];
            Masses = new float[count];
            Mapping = new int[count];

            for (uint i = 0; i < count; i++)
            {
                Positions[i] = new Vector3(POSITION_MAX);
                Masses[i] = Rand.NextSingle() / 2 + 0.5f;
            }

            Meshes = new Mesh[MESH_LENGTH * MESH_LENGTH];
            for (uint y = 0; y < MESH_LENGTH; y++)
            {
                for (uint x = 0; x < MESH_LENGTH; x++)
                {
                    Meshes[y * MESH_LENGTH + x] = new Mesh(x, y);
                }
            }
        }

        public void Tick(float dt)
        {
            ComputeCenterOfMass();

            MapPositionsToMesh();

            ComputeMeshCentersOfMass();

            //physix
        }

        private void ComputeCenterOfMass()
        {
            Vector3 com = new Vector3(0, 0, 0);

            for (int i = 0; i < Count; i++)
            {
                com += Positions[i] * Masses[i];
            }

            CenterOfMass = com / Masses.Sum();
        }

        private void MapPositionsToMesh()
        {
            Vector3 pos, norm;
            int j;

            for (uint i = 0; i < Count; i++)
            {
                pos = Positions[i];
                norm.x = pos.x - CenterOfMass.x;
                norm.y = pos.y - CenterOfMass.y;

                float x = norm.x + MESH_WIDTH / 2;
                if (x < 0 || x >= MESH_WIDTH)
                {
                    Mapping[i] = -1;
                    continue;
                }

                float y = norm.y + MESH_WIDTH / 2;
                if (y < 0 || y >= MESH_WIDTH)
                {
                    Mapping[i] = -1;
                    continue;
                }

                uint x_idx = (uint)(x / MESH_NODE_LENGTH);
                uint y_idx = (uint)(y / MESH_NODE_LENGTH);

                j = (int) (y_idx * MESH_LENGTH + x_idx);

                if (j >= Meshes.Length)
                    throw new Exception("index out of range");

                Mapping[i] = j;
            }
        }

        private void ComputeMeshCentersOfMass()
        {
            uint i;
            int mi;
            Mesh m;

            for (i = 0; i < MESH_LENGTH * MESH_LENGTH; i++)
            {
                Meshes[i].Reset();
            }

            for (i = 0; i < Count; i++)
            {
                mi = Mapping[i];
                if (mi == -1)
                    continue;

                m = Meshes[mi];
                m.AddParticle(ref Positions[i], Masses[i]);
            }

            for (i = 0; i < MESH_LENGTH * MESH_LENGTH; i++)
            {
                Meshes[i].center /= Meshes[i].mass;
            }
        }
    }
}
