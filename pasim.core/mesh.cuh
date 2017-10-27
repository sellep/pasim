#ifndef __PASIM_MESH_H
#define __PASIM_MESH_H 1

#include "defs.cuh"
#include "v3.cuh"

typedef struct
{
	v3 com;
	float m;
	mesh *parent;
} mesh;

/*
host

public static Vector3 root_mesh(Vector3 center_of_mesh)
{
return new Vector3(
center_of_mesh.x - (MESH_GRID_SIDE_NODES - 1) / 2 * MESH_NODE_SIDE_LENGTH - MESH_NODE_SIDE_LENGTH / 2,
center_of_mesh.y - (MESH_GRID_SIDE_NODES - 1) / 2 * MESH_NODE_SIDE_LENGTH - MESH_NODE_SIDE_LENGTH / 2,
center_of_mesh.z - (MESH_GRID_SIDE_NODES - 1) / 2 * MESH_NODE_SIDE_LENGTH - MESH_NODE_SIDE_LENGTH / 2);
}
*/

__device__ static inline uint node_index_of(
	v3 const * const center_of_mesh,
	v3 const * const position)
{
	//return (position.y - center_of_mesh.y) * MESH_GRID_SIDE_NODES + (position.y - center_of_mesh.y + MESH_GRID_SIDE_NODES / 2)

	return
		(position->y - center_of_mesh->y) 
		+ (position->x - center_of_mesh->x) / (MESH_GRID_SIDE_NODES / 2)
}

#endif