using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace meshes
{

//    public class Mesh
//    {
//        public const float MESH_WIDTH = 250;
//        public const ushort LAYER_1_LENGTH = 7;
//        public const ushort LAYER_2_LENGTH = 3;
//        public const ushort LAYER_3_LENGTH = 3;
//        public const ushort MESH_LENGTH = LAYER_1_LENGTH * LAYER_2_LENGTH * LAYER_3_LENGTH;
//        public const float LAYER_1_WIDTH = MESH_WIDTH / LAYER_1_LENGTH;
//        public const float LAYER_2_WIDTH = LAYER_1_WIDTH / LAYER_2_LENGTH;
//        public const float LAYER_3_WIDTH = LAYER_2_WIDTH / LAYER_3_LENGTH;

//        public Vector3[] layer1 = new Vector3[LAYER_1_LENGTH *LAYER_1_LENGTH];
//        public Vector3[] layer2 = new Vector3[MESH_LENGTH / LAYER_1_LENGTH * MESH_LENGTH / LAYER_1_LENGTH];
//        public Vector3[] layer3 = new Vector3[MESH_LENGTH * MESH_LENGTH];

//        public Vector3[] mapping;

//        public Vector3 mid;

//        public Mesh(uint count)
//        {
//            mapping = new Vector3[count];
//            mid = new Vector3(0, 0, 0);
//        }

//        private int index_l3(Vector3 position)
//        {
//            int x = (int)((position.x + MESH_WIDTH / 2) / LAYER_3_WIDTH);
//            int y = (int)((position.y + MESH_WIDTH / 2) / LAYER_3_WIDTH);

//            //boundary check?

//            return y * MESH_LENGTH + x;
//        }

//        public void map(Vector3[] positions, float[] masses)
//        {
//            uint i;

//            //do center of mass reset

//            for (i = 0; i < positions.Length; i++)
//            {
//                mapping[i] = new Vector3();

//                mapping[i].x = index_l3(positions[i]);

//                layer3[(int)mapping[i].x].x += positions[i].x * masses[i];
//                layer3[(int)mapping[i].x].y += positions[i].y * masses[i];
//                layer3[(int)mapping[i].x].z += masses[i];
//            }
//        }

//        public float compute(Vector3[] positions, float[] masses, uint i)
//        {
//            uint j;

//            Vector3 map = mapping[i];
//            if (map.x < 0)
//                return 0;

//            for (j = 0; j < MESH_LENGTH * MESH_LENGTH; j++)
//            {
//                if (j == map.x)
//                {
//                    //compute for all particles in same mesh
//                }
//                else
//                {
//                    //compute with mesh

//                    Vector3 r = layer3[(int)map.x] - positions[i];
//                    Vector3.norm(ref r);

//                    //force += GRAVITIONAL_CONSTANT * layer3[(int)map.x].z

//                    /*
//                     for (j = 0; j < N; j++)
//	{
//		if (i == j)
//			continue;

//		//r = rs[j] - rs[i]
//		v3_sub(&tmp, &r[j], &self_r);

//		//norm = norm(r);
//		norm = v3_norm(&tmp);

//		//G * m1 * m2 * r / norm(r)^3
//		v3_idiv(&tmp, &tmp, norm * norm * norm);
//		v3_imul(&tmp, &tmp, GRAVITATION_CONSTANT * self_m * m[j]);
//		v3_add(&F, &F, &tmp);
//	}

//	v3_imul(&dp[i], &F, dt);
//                    */
//                }
//            }

//            return 0;
//        }
//    }
}
