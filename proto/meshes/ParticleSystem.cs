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

        public uint[] Mapping;

        public ParticleSystem(uint count)
        {
            Count = count;
            Positions = new Vector3[count];
            Masses = new float[count];
            Mapping = new uint[count];

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

            //compute meshed base center of mass

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
            for (uint i = 0; i < Count; i++)
            {
                float x = MESH_WIDTH / 2 + Positions[i].x - CenterOfMass.x;
                float y = MESH_WIDTH / 2 + Positions[i].y - CenterOfMass.y;

                uint x_idx = (uint)(x / MESH_NODE_LENGTH);
                uint y_idx = (uint)(y / MESH_NODE_LENGTH);

                Mapping[i] = y_idx * MESH_LENGTH + x_idx;
            }
        }

        private void ComputeMeshCentersOfMass()
        {
            //reset mass and vector from meshes

            for (uint i = 0; i < Count; i++)
            {
                //mesh.centerOfMass += Location[i] * Mass[i]
            }

            //Meshes.CenterOfMass /= Mesh.Mass;
        }
    }
}
