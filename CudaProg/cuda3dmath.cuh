#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <stdio.h>
#include <vector_functions.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>


inline __device__ float dot(const float3& v1, const float3& v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline __device__ float3 cross(const float3& v1, const float3& v2) {
	return make_float3(
		v1.y * v2.z - v1.z * v2.y,
		v1.z * v2.x - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x);
}

inline __device__ float3 operator-(const float3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator*(const float3& a, const float& b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ bool intersect_ray_sphere(const float3& ray_origin, const float3& ray_direction, const float3& sphere_center, const float& sphere_radius)
{
    float3 dist = sphere_center - ray_origin;
    float B = dot(dist, ray_direction);
    float C = dot(dist, dist) - sphere_radius * sphere_radius;
    float D = B * B - C;
    //if (D > 0.0) return true;
    //else return false;
	return D > 0.0;
}

__device__ float RayTriangleIntersection(const float3& origin, const float3& r, const float3& v0, const float3& edge1, const float3& edge2)
{

	float3 tvec = origin - v0; //r.orig - v0;
	float3 pvec = cross(r, edge2);
	float  det = dot(edge1, pvec);

	det = 1.0f / det;

	float u = dot(tvec, pvec) * det;

	if (u < 0.0f || u > 1.0f)
		return -1.0f;

	float3 qvec = cross(tvec, edge1);

	float v = dot(r, qvec) * det;

	if (v < 0.0f || (u + v) > 1.0f)
		return -1.0f;

	return dot(edge2, qvec) * det;
}