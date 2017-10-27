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
        public const uint MESH_L2_LENGTH = 9;
        public const uint MESH_L1_LENGTH = 3;
        public const uint MESH_LENGTH = MESH_L2_LENGTH * MESH_L1_LENGTH;
        public const float MESH_NODE_L1_LENGTH = (2 * POSITION_MAX) / MESH_LENGTH;
        public const float MESH_WIDTH = MESH_LENGTH * MESH_NODE_L1_LENGTH;

        public uint Count;

        public Vector3[] Positions;

        public float[] Masses;

        public Mesh[,] MeshesL1;
        public Mesh[,] MeshesL2;

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

            MeshesL1 = new Mesh[MESH_LENGTH, MESH_LENGTH];
            MeshesL2 = new Mesh[MESH_L2_LENGTH, MESH_L2_LENGTH];

            for (uint y2 = 0; y2 < MESH_L2_LENGTH; y2++)
            {
                for (uint x2 = 0; x2 < MESH_L2_LENGTH; x2++)
                {
                    Mesh l2 = new Mesh(x2, y2);

                    MeshesL2[y2, x2] = l2;

                    //generate L1 meshes
                    for (uint y1 = 0; y1 < MESH_L1_LENGTH; y1++)
                    {
                        for (uint x1 = 0; x1 < MESH_L1_LENGTH; x1++)
                        {
                            MeshesL1[x2 * MESH_L1_LENGTH + x1, y2 * MESH_L1_LENGTH + y1] = new Mesh(l2, x2 * MESH_L1_LENGTH + x1, y2 * MESH_L1_LENGTH + y1, x1, y1);
                        }
                    }
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

                uint x_idx = (uint)(x / MESH_NODE_L1_LENGTH);
                uint y_idx = (uint)(y / MESH_NODE_L1_LENGTH);

                Mapping[i] = (int)(y_idx * MESH_LENGTH + x_idx);
            }
        }

        private void ComputeMeshCentersOfMass()
        {
            uint i, x2, y2, x1, y1, x, y;
            int mi;

            for (y1 = 0; y1 < MESH_LENGTH ; y1++)
            {
                for (x1 = 0; x1 < MESH_LENGTH; x1++)
                {
                    MeshesL1[x1, y1].Reset();
                }
            }

            for (i = 0; i < Count; i++)
            {
                mi = Mapping[i];
                if (mi == -1)
                    continue;

                y = (uint)mi / MESH_LENGTH;
                x = (uint)mi - (y * MESH_LENGTH);

                MeshesL1[x, y].AddParticle(ref Positions[i], Masses[i]);
            }

            for (y2 = 0; y2 < MESH_L2_LENGTH; y2++)
            {
                for (x2 = 0; x2 < MESH_L2_LENGTH; x2++)
                {
                    MeshesL2[x2, y2].Reset();

                    for (y1 = 0; y1 < MESH_L1_LENGTH; y1++)
                    {
                        for (x1 = 0; x1 < MESH_L1_LENGTH; x1++)
                        {
                            x = x2 * MESH_L1_LENGTH + x1;
                            y = y2 * MESH_L1_LENGTH + y1;

                            MeshesL2[x2, y2].center += MeshesL1[x, y].center;
                            MeshesL2[x2, y2].mass   += MeshesL1[x, y].mass;

                            MeshesL1[x, y].center /= MeshesL1[x, y].mass;
                        }
                    }

                    MeshesL2[x2, y2].center /= MeshesL2[x2, y2].mass;
                }
            }
        }
    }
}
