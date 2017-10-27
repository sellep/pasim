#include "kernel.cuh"


__device__ void assign_particles_to_mesh(
	uint       * const mesh_particle_mapping,
	v3   const * const positions,
	v3   const * const center_of_mesh,
	uint         const idx)
{
	//based on positions[idx], get idx of mesh, assign this index to par

	mesh_particle_mapping[idx] = mesh_index_of(center_of_mesh, positions[idx]);
}

__device__ void compute_meshes_center_of_mass(
	v3   const * const origin,
	uint       * const mesh_particle_mapping,
	uint         const idx)
{
	// one block should be level 3 node
	//so we can use shared memory in which we store level 1 (and level 2?) node infos to compute level 3 stuff

	int i;
	float mass = 0;
	
	//iterate over mesh_particle_mapping
	for (i = 0; i < )
}