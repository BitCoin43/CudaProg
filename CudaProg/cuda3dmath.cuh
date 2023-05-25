#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <stdio.h>
#include <vector_functions.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>


inline __host__ __device__ float dot(const float3& v1, const float3& v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline __host__ __device__ float3 cross(const float3& v1, const float3& v2) {
	return make_float3(
		v1.y * v2.z - v1.z * v2.y,
		v1.z * v2.x - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x);
}

inline __host__ __device__ float3 operator-(const float3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator*(const float3& a, const float& b)
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

__device__ inline void DPdev(int* dev_mem, int x, int y, int width, int height,
	unsigned char r, unsigned char g, unsigned char b) {
	if (x >= 0 && x < width && y >= 0 && y < height) {
		dev_mem[y * width + x] = (r << 16) | (g << 8) | b;
	}
}

__device__ inline void DPdev(int* dev_mem, int x, int y, int width, int height, int color) {
	if (x >= 0 && x < width && y >= 0 && y < height) {
		dev_mem[y * width + x] = color;
	}
}

__device__ inline void DP_ARGB_dev(int* dev_mem, int x, int y, int width, int height,
	unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
	if (x >= 0 && x < width && y >= 0 && y < height) {
		int color = dev_mem[y * width + x];

		float r_b = ((color >> 16) & 0x000000ff) / 255.0;
		float g_b = ((color >> 8) & 0x000000ff) / 255.0;
		float b_b = (color & 0x000000ff) / 255.0;

		float r_u = r / 255.0;
		float g_u = g / 255.0;
		float b_u = b / 255.0;

		float al = a / 255.0;

		unsigned char r_r = (r_u * al + r_b * (1 - al)) * 255;
		unsigned char g_r = (g_u * al + g_b * (1 - al)) * 255;
		unsigned char b_r = (b_u * al + b_b * (1 - al)) * 255;

		dev_mem[y * width + x] = (r_r << 16) | (g_r << 8) | b_r;
	}
}

__device__ int getColorOnBlakc(int color, unsigned char I) {
	float r_u = ((color >> 16) & 0x000000ff);
	float g_u = ((color >> 8) & 0x000000ff);
	float b_u = (color & 0x000000ff);

	float al = I / 255.0;

	unsigned char r_r = r_u * al;
	unsigned char g_r = g_u * al;
	unsigned char b_r = b_u * al;
	return (r_r << 16) | (g_r << 8) | b_r;
}

__global__ void DPglobal(int* dev_mem, int x, int y, int width, int height,
	unsigned char r, unsigned char g, unsigned char b) {
	if (x >= 0 && x < width && y >= 0 && y < height) {
		dev_mem[y * width + x] = (r << 16) | (g << 8) | b;
	}
}

__global__ void DP_ARGB_global(int* dev_mem, int x, int y, int width, int height,
	unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
	if (x >= 0 && x < width && y >= 0 && y < height) {
		int color = dev_mem[y * width + x];

		float r_b = ((color >> 16) & 0x000000ff) / 255.0;
		float g_b = ((color << 16 >> 24) & 0x000000ff) / 255.0;
		float b_b = ((color << 24 >> 24) & 0x000000ff) / 255.0;

		float r_u = r / 255.0;
		float g_u = g / 255.0;
		float b_u = b / 255.0;

		float al = a / 255.0;

		unsigned char r_r = (r_u * al + r_b * (1 - al)) * 255;
		unsigned char g_r = (g_u * al + g_b * (1 - al)) * 255;
		unsigned char b_r = (b_u * al + b_b * (1 - al)) * 255;

		dev_mem[y * width + x] = (r_r << 16) | (g_r << 8) | b_r;
	}
}

__global__ void renderCircle(int* dev_mem, int x1, int y1, int R, int width, int height,
	unsigned char r, unsigned char g, unsigned char b) {
	//DPdev(dev_mem, x + blockIdx.x, y + blockIdx.y, width, height, r, g, b);
	int x = blockIdx.x - R; int y = blockIdx.y - R;
	if (x * x + y * y <= (R - 0.5) * (R - 0.5)) {
		DPdev(dev_mem, blockIdx.x + x1, blockIdx.y + y1, width, height, r, g, b);
	}
}

__global__ void renderLine(int* dev_mem, const int width, const int height, const float x0, const float y0, const float x1, const float y1,
	unsigned char r, unsigned char g, unsigned char b)
{
	int x = blockIdx.x; int y = blockIdx.y;

	if (x >= width || y >= height)
		return;

	float AB = std::sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
	float AÑ = std::sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0));
	float ÑB = std::sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y));

	// adjust threshold to make the line thicker
	const float threshold = 0.1f;
	if (std::fabs(AB - (AÑ + ÑB)) <= threshold) {
		DPdev(dev_mem, x0 + x, y0 + y, width, height, r, g, b);
	}
}

__global__ void renderPoligon(int* dev_mem, int xmin, int ymin, int x1, int y1, int x2, int y2, int x3, int y3, int width, int height,
	unsigned char r, unsigned char g, unsigned char b) {
	int x = blockIdx.x + xmin; int y = blockIdx.y + ymin;
	int a = (x1 - x) * (y2 - y1) - (x2 - x1) * (y1 - y);
	int B = (x2 - x) * (y3 - y2) - (x3 - x2) * (y2 - y);
	int c = (x3 - x) * (y1 - y3) - (x1 - x3) * (y3 - y);

	if ((a >= 0 && B >= 0 && c >= 0) || (a <= 0 && B <= 0 && c <= 0)) {
		DPdev(dev_mem, blockIdx.x + xmin, blockIdx.y + ymin, width, height, r, g, b);
	}
}

__global__ void renderRect_dev(int* dev_mem, int xmin, int ymin, int width, int height,
	unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
	int x = blockIdx.x + xmin; int y = blockIdx.y + ymin;
	DP_ARGB_dev(dev_mem, x, y, width, height, r, g, b, a);
}

__global__ void clean(int* dev_mem, int width, int height,
	unsigned char r, unsigned char g, unsigned char b) {
	dev_mem[blockIdx.y * width + blockIdx.x] = (r << 16) | (g << 8) | b;
}

__global__ void getInt_dev(int* dev_mem, int* c, int width, int x, int y) {
	*c = dev_mem[y * width + x];
}

__device__ inline float squarOfTriangle(float a, float b, float c) {
	float p = (a + b + c) / 2;
	return sqrt(p * (p - a) * (p - b) * (p - c));
}

__device__ bool in_t(float3 p, float3 poly[3]) {
	if (p.x == 0 && p.y == 0 && p.z == 0) {
		return false;
	}
	bool inside = false;
	float AB = sqrt((poly[0].x - poly[1].x) * (poly[0].x - poly[1].x) + (poly[0].y - poly[1].y) * (poly[0].y -
		poly[1].y) + (poly[0].z - poly[1].z) * (poly[0].z - poly[1].z));
	float BC = sqrt((poly[1].x - poly[2].x) * (poly[1].x - poly[2].x) + (poly[1].y - poly[2].y) * (poly[1].y -
		poly[2].y) + (poly[1].z - poly[2].z) * (poly[1].z - poly[2].z));
	float CA = sqrt((poly[0].x - poly[2].x) * (poly[0].x - poly[2].x) + (poly[0].y - poly[2].y) * (poly[0].y -
		poly[2].y) + (poly[0].z - poly[2].z) * (poly[0].z - poly[2].z));

	float AP = sqrt((p.x - poly[0].x) * (p.x - poly[0].x) + (p.y - poly[0].y) * (p.y - poly[0].y) + (p.z - poly[0].z) * (p.z - poly[0].z));
	float BP = sqrt((p.x - poly[1].x) * (p.x - poly[1].x) + (p.y - poly[1].y) * (p.y - poly[1].y) + (p.z - poly[1].z) * (p.z - poly[1].z));
	float CP = sqrt((p.x - poly[2].x) * (p.x - poly[2].x) + (p.y - poly[2].y) * (p.y - poly[2].y) + (p.z - poly[2].z) * (p.z - poly[2].z));
	float diff = (squarOfTriangle(AP, BP, AB) + squarOfTriangle(AP, CP, CA) + squarOfTriangle(BP, CP, BC)) - squarOfTriangle(AB, BC, CA);
	if (fabs(diff) < 0.001) inside = true;
	return inside;

}

__device__ bool EQV(float3& f1, float3& f2) {
	return f1.x == f2.x && f1.y == f2.y && f1.z == f2.z;
}

__device__ bool fabs_f3(float3& f1, float3& f2, const float& s) {
	float x = f1.x - f2.x;	if (x < 0) x *= -1;
	float y = f1.y - f2.y;	if (y < 0) y *= -1;
	float z = f1.z - f2.z;	if (z < 0) z *= -1;
	return x < s&& y < s&& z < s;
}

__device__ float3 crossProduct_f3(float3& v1, float3& v2) {
	return make_float3(
		v1.y * v2.z - v1.z * v2.y,
		v1.z * v2.x - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x);
}

__device__ float getLength(float3& f) {
	return sqrt(f.x * f.x + f.y * f.y + f.z * f.z);
}

__device__ float3 getNormal(float3& p0, float3& p1, float3& p2) {
	/*
	float vx1 = p0.x - p1.x;
	float vy1 = p0.y - p1.y;
	float vz1 = p0.z - p1.z;
	float vx2 = p1.x - p2.x;
	float vy2 = p1.y - p2.y;
	float vz2 = p1.z - p2.z;
	float x = vy1 * vz2 - vz1 * vz2;
	float y = vz1 * vx2 - vx1 * vx2;
	float z = vx1 * vy2 - vy1 * vy2;

	return make_float3(
		x,
		y,
		z);*/
	float3 e1 = p1 - p0;
	float3 e2 = p2 - p1;
	float3 n = crossProduct_f3(e1, e2);
	return n;
}

__global__ void ray_tracing(int* dev_mem, int width, int height, float3* poly,
	int* colors, int countOfAllPolygons, float4* lites, int count_of_lites) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		float s = 999999999999999999.0;
		unsigned char Intens = 40;
		int color_index = 0;
		bool is = false;
		float3 end[3] = {};
		float3 dir = make_float3(x - (width >> 1), y - (height >> 1), width * 1.0);

		for (int i = 0; i < countOfAllPolygons; i++) {
			float3 polygon[3] = { poly[i * 3], poly[i * 3 + 1], poly[i * 3 + 2] };
			float p = RayTriangleIntersection(make_float3(0, 0, 0), dir, polygon[0], polygon[1], polygon[2]);
			if (p < s) {
				s = p;
				is = true;
				color_index = i;
				end[0] = polygon[0];
				end[1] = polygon[1];
				end[2] = polygon[2];
			}
		}

		if (is) {
			float3 G = dir * s;

			for (int j = 0; j < count_of_lites; j++) {
				float3 P = make_float3(lites[j].x, lites[j].y, lites[j].z);
				float3 L = P - G;
				float3 EndPoint;
				bool wall = true;
				for (int i = 0; i < countOfAllPolygons; i++) {
					if (i != color_index) {
						if (-1 != RayTriangleIntersection(G, L, poly[i * 3], poly[i * 3 + 1], poly[i * 3 + 2])) {
							wall = false;
							break;
						}
					}
				}
				if (wall) {
					float3 N = getNormal(end[0], end[1], end[2]);
					float n_dot = dot(N, L);
					if (n_dot > 0) {
						Intens += lites[0].w * n_dot / (getLength(N) * getLength(L));
					}
				}
			}
			DPdev(dev_mem, x, y, width, height, getColorOnBlakc(colors[color_index], Intens));
		}
	}
}

__global__ void path_tracing_1(int* dev_mem, int width, int height, float3* poly,
	int* colors, int countOfAllPolygons, float4* lites, int count_of_lites) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
}

__device__ void mi(Node* n, const float3& ray_origin, const float3& ray_direction, float3* poly, bool& intersect, int& poly_ind, float3& point) {
	for (int i = 0; i < n->count_of_child; i++) {
		if (n->poly != -1) {
			i = n->poly;
			float p;
			p = RayTriangleIntersection(ray_origin, ray_direction, poly[i * 3], poly[i * 3 + 1], poly[i * 3 + 2]);
			poly_ind = i;

		}
		else if (intersect_ray_sphere(ray_origin, ray_direction, n->sphere, n->radius)) {
			//mi(n, ray_origin, ray_direction, poly, intersect, poly_ind, point);
		}
	}
}