#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "DeviceKernel.cuh"
#include <cmath>
#include <algorithm>

__device__ inline void DPdev(int* dev_mem, int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b) {
	if (x >= 0 && x < width && y >= 0 && y < height) {
		dev_mem[y * width + x] = (r << 16) | (g << 8) | b;
	}
}

__device__ inline void DPdev(int* dev_mem, int x, int y, int width, int height, int color) {
	if (x >= 0 && x < width && y >= 0 && y < height) {
		dev_mem[y * width + x] = color;
	}
}

__device__ inline void DP_ARGB_dev(int* dev_mem, int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
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

__global__ void DPglobal(int* dev_mem, int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b) {
	if (x >= 0 && x < width && y >= 0 && y < height) {
		dev_mem[y * width + x] = (r << 16) | (g << 8) | b;
	}
}

__global__ void DP_ARGB_global(int* dev_mem, int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
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

__global__ void renderCircle(int* dev_mem, int x1, int y1, int R, int width, int height, unsigned char r, unsigned char g, unsigned char b) {
	//DPdev(dev_mem, x + blockIdx.x, y + blockIdx.y, width, height, r, g, b);
	int x = blockIdx.x - R; int y = blockIdx.y - R;
	if (x * x + y * y <= (R - 0.5) * (R - 0.5)) {
		DPdev(dev_mem, blockIdx.x + x1, blockIdx.y + y1, width, height, r, g, b);
	}
}

__global__ void renderLine(int* dev_mem, const int width, const int height, const float x0, const float y0, const float x1, const float y1, unsigned char r, unsigned char g, unsigned char b)
{
	int x = blockIdx.x; int y = blockIdx.y;

	if (x >= width || y >= height)
		return;

	float AB = std::sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
	float AС = std::sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0));
	float СB = std::sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y));

	// adjust threshold to make the line thicker
	const float threshold = 0.1f;
	if (std::fabs(AB - (AС + СB)) <= threshold) {
		DPdev(dev_mem, x0 + x, y0 + y, width, height, r, g, b);
	}
}

__global__ void renderPoligon(int* dev_mem, int xmin, int ymin, int x1, int y1, int x2, int y2, int x3, int y3, int width, int height, unsigned char r, unsigned char g, unsigned char b) {
	int x = blockIdx.x + xmin; int y = blockIdx.y + ymin;
	int a = (x1 - x) * (y2 - y1) - (x2 - x1) * (y1 - y);
	int B = (x2 - x) * (y3 - y2) - (x3 - x2) * (y2 - y);
	int c = (x3 - x) * (y1 - y3) - (x1 - x3) * (y3 - y);

	if ((a >= 0 && B >= 0 && c >= 0) || (a <= 0 && B <= 0 && c <= 0)) {
		DPdev(dev_mem, blockIdx.x + xmin, blockIdx.y + ymin, width, height, r, g, b);
	}
}

__global__ void renderRect_dev(int* dev_mem, int xmin, int ymin, int width, int height, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
	int x = blockIdx.x + xmin; int y = blockIdx.y + ymin;
	DP_ARGB_dev(dev_mem, x, y, width, height, r, g, b, a);
}

__global__ void clean(int* dev_mem, int width, int height, unsigned char r, unsigned char g, unsigned char b) {
	dev_mem[blockIdx.y * width + blockIdx.x] = (r << 16) | (g << 8) | b;
}

__device__ int NormalizePoint() {

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

__device__ inline float3 MIN_f3(float3& f1, float3& f2) {
	return make_float3(f1.x - f2.x, f1.y - f2.y, f1.z - f2.z);
}

__device__ inline float3 ADD_f3(float3& f1, float3& f2) {
	return make_float3(f1.x + f2.x, f1.y + f2.y, f1.z + f2.z);
}

__device__ inline float3 MLT_f3(float3& f, float s) {
	return make_float3(f.x * s, f.y * s, f.z * s);
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

__device__ float dot_f3(float3& v1, float3& v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
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
	float3 e1 = MIN_f3(p1, p0);
	float3 e2 = MIN_f3(p2, p1);
	float3 n = crossProduct_f3(e1, e2);
	return n;
}

__device__ float rayTriangleIntersect(float3& origin, float3& direction, float3& p0, float3& p1, float3& p2, float minT) {
	const float kNoIntersection = FLT_MAX;

	float3 e1 = MIN_f3(p1, p0);
	float3 e2 = MIN_f3(p2, p1);
	float3 n = crossProduct_f3(e1, e2);
	float dot = dot_f3(n, direction);

	if (!(dot < 0.0f)) return kNoIntersection;

	float d = dot_f3(n, p0);
	float t = d - dot_f3(n, origin);

	if (!(t <= 0.0f)) return kNoIntersection;
	if (!(t >= dot * minT)) return kNoIntersection;

	t /= dot;
	float3 p = ADD_f3(origin, MLT_f3(direction, t));

	float u0, u1, u2;
	float v0, v1, v2;
	if (fabs(n.x) > fabs(n.y)) {
		if (fabs(n.x) > fabs(n.z)) {
			u0 = p.y - p0.y;
			u1 = p1.y - p0.y;
			u2 = p2.y - p0.y;
			v0 = p.z - p0.z;
			v1 = p1.z - p0.z;
			v2 = p2.z - p0.z;
		}
		else {
			u0 = p.x - p0.x;
			u1 = p1.x - p0.x;
			u2 = p2.x - p0.x;
			v0 = p.y - p0.y;
			v1 = p1.y - p0.y;
			v2 = p2.y - p0.y;
		}
	}
	else {
		if (fabs(n.y) > fabs(n.z)) {
			u0 = p.x - p0.x;
			u1 = p1.x - p0.x;
			u2 = p2.x - p0.x;
			v0 = p.z - p0.z;
			v1 = p1.z - p0.z;
			v2 = p2.z - p0.z;
		}
		else {
			u0 = p.x - p0.x;
			u1 = p1.x - p0.x;
			u2 = p2.x - p0.x;
			v0 = p.y - p0.y;
			v1 = p1.y - p0.y;
			v2 = p2.y - p0.y;
		}
	}
	float temp = u1 * v2 - v1 * u2;
	if (!(temp != 0.0f)) {
		return kNoIntersection;
	}
	temp = 1.0f / temp;

	float alpha = (u0 * v2 - v0 * u2) * temp;
	if (!(alpha >= 0.0f))	return kNoIntersection;
	float beta = (u1 * v0 - v1 * u0) * temp;
	if (!(beta >= 0.0f))	return kNoIntersection;
	float gamma = 1.0f - alpha - beta;
	if (!(gamma >= 0.0f)) 	return kNoIntersection;

	return t;
}

__device__ inline float4 cross(float3 origin, float3 dir, float3 poly[3]) {
	float3 E1 = MIN_f3(poly[1], poly[0]);
	float3 E2 = MIN_f3(poly[2], poly[0]);
	float3 T = MIN_f3(origin, poly[0]);
	float3 P = crossProduct_f3(dir, E2);
	float3 Q = crossProduct_f3(T, E1);
	float t = (1 / dot_f3(P, E1)) * dot_f3(Q, E2);
	float u = (1 / dot_f3(P, E1)) * dot_f3(P, T);
	float v = (1 / dot_f3(P, E1)) * dot_f3(Q, dir);
	float3 r = ADD_f3(MLT_f3(poly[1], (1 - u - v)), ADD_f3(MLT_f3(poly[2], u), MLT_f3(poly[0], v)));
	float3 l = ADD_f3(origin, MLT_f3(dir, t));
	if (fabs_f3(r, l, FLT_MAX)) {
		return make_float4(r.x, r.y, r.z, t);
	}
	return make_float4(0, 0, 0, 0);
}

__global__ void ray_tracing(int* dev_mem, int width, int height, float3* poly, int* colors, int countOfAllPolygons, float4* lites, int count_of_lites) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x <= width && y <= height) {

		float s = FLT_MAX;
		unsigned char Intens = 10;
		int color_index = 0;
		bool is = false;
		float3 end[3] = {};
		float3 dir = make_float3(x - width / 2, y - height / 2, width * 1.0);
		for (int i = 0; i < countOfAllPolygons; i++) {
			float3 polygon[3] = { poly[i * 3], poly[i * 3 + 1], poly[i * 3 + 2] };
			//float4 p = cross(make_float3(0, 0, 0), dir, polygon);
			float p = rayTriangleIntersect(make_float3(0, 0, 0), dir, polygon[0], polygon[1], polygon[2], 10);
			if (p != FLT_MAX) {
				if (p < s) {
					s = p;
					is = true;
					color_index = i;
					
					end[0] = polygon[0];
					end[1] = polygon[1];
					end[2] = polygon[2];
				}
			}
		}
		if (is) {
			float3 P = make_float3(lites[0].x, lites[0].y, lites[0].z);
			float3 G = MLT_f3(dir, s);
			float3 L = MIN_f3(P, G);
			bool wall = true;
			for (int i = 0; i < countOfAllPolygons; i++) {
				if (i != color_index) {

					//float t =  rayTriangleIntersect(G,P ,  poly[i * 3], poly[i * 3 + 1], poly[i * 3 + 2], 0);
					float3 polygon[3] = { poly[i * 3], poly[i * 3 + 1], poly[i * 3 + 2] };

					if (rayTriangleIntersect(G, L, polygon[0], polygon[1], polygon[2], 10) != FLT_MAX) {
						wall = false;
						break;
					}
				}
			}
			if (wall) {
				
				float3 N = getNormal(end[0], end[1], end[2]);
				float n_dot = dot_f3(N, L);
				if (n_dot > 0) {
					Intens += lites[0].w * n_dot / (getLength(N) * getLength(L));
				}

			}
			
			DPdev(dev_mem, x, y, width, height, getColorOnBlakc(colors[color_index], Intens));
			
		}
		//DP_ARGB_dev(dev_mem, x, y, width, height, 0, 0, 255, 100);
	}
}

Device::Device(int height, int width, int* Colors)
	:
	dev_width(width),
	dev_height(height)
{
	cudaMalloc((void**)&dev_mem, sizeof(int) * height * width);
}

Device::~Device()
{
	cudaFree(dev_mem);
}

void Device::copyDeviceToHost(int& Colors)
{
	cudaMemcpy(&Colors, dev_mem, sizeof(int) * dev_height * dev_width, cudaMemcpyDeviceToHost);
}

void Device::cleanDeviceMem(unsigned char r, unsigned char g, unsigned char b)
{
	dim3 grid(dev_width, dev_height);
	clean << <grid, 1 >> > (dev_mem, dev_width, dev_height, r, g, b);
}

void Device::drawPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
	DPglobal << <1, 1 >> > (dev_mem, x, y, dev_width, dev_height, r, g, b);
}

void Device::drawCircle(int x, int y, int R, unsigned char r, unsigned char g, unsigned char b)
{
	dim3 grid(R * 2, R * 2);
	renderCircle << <grid, 1 >> > (dev_mem, x, y, R, dev_width, dev_height, r, g, b);
}

void Device::drawLine(int x0, int y0, int x1, int y1, unsigned char r, unsigned char g, unsigned char b)
{
	dim3 grid(x1 - x0, y1 - y0);
	renderLine << <grid, 1 >> > (dev_mem, dev_width, dev_height, x0, y0, x1, y1, r, g, b);
}

void Device::drawPoligon(int x1, int y1, int x2, int y2, int x3, int y3, unsigned char r, unsigned char g, unsigned char b)
{
	int xmin = 0;
	int xmax = 0;
	int ymin = 0;
	int ymax = 0;

	if (x1 <= x2 && x1 <= x3) xmin = x1;
	if (x2 <= x1 && x2 <= x3) xmin = x2;
	if (x3 <= x2 && x3 <= x1) xmin = x3;

	if (y1 <= y2 && y1 <= y3)	ymin = y1;
	if (y2 <= y1 && y2 <= y3)	ymin = y2;
	if (y3 <= y2 && y3 <= y1)	ymin = y3;

	if (x1 >= x2 && x1 >= x3) xmax = x1;
	if (x2 >= x1 && x2 >= x3) xmax = x2;
	if (x3 >= x2 && x3 >= x1) xmax = x3;

	if (y1 >= y2 && y1 >= y3)	ymax = y1;
	if (y2 >= y1 && y2 >= y3)	ymax = y2;
	if (y3 >= y2 && y3 >= y1)	ymax = y3;

	dim3 grid(xmax - xmin, ymax - ymin);
	renderPoligon << <grid, 1 >> > (dev_mem, xmin, ymin, x1, y1, x2, y2, x3, y3, dev_width, dev_height, r, g, b);
}

void Device::drawRect(int x1, int x2, int y1, int y2, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	dim3 grid(x2 - x1, y2 - y1);
	renderRect_dev << <grid, 1 >> > (dev_mem, x1, y1, dev_width, dev_height, r, g, b, a);
}

float Device::normalizePointX(float p, float z)
{
	if (z != 0) p /= -(z * 0.5);
	return dev_width / 2 + dev_height / 2 * p;
}

float Device::normalizePointY(float p, float z)
{
	if (z != 0) p /= (-z * 0.5);
	return dev_height / 2 + dev_height / 2 * p;
}

void Device::drawPlane(plane pl, unsigned char r, unsigned char g, unsigned char b)
{
	if (pl.p1.z > 0.4 && pl.p3.z > 0.4 && pl.p2.z > 0.4 && pl.p4.z > 0.4) {
		float x1 = normalizePointX(pl.p1.x, pl.p1.z);
		float x2 = normalizePointX(pl.p2.x, pl.p2.z);
		float x3 = normalizePointX(pl.p3.x, pl.p3.z);
		float x4 = normalizePointX(pl.p4.x, pl.p4.z);

		float y1 = normalizePointY(pl.p1.y, pl.p1.z);
		float y2 = normalizePointY(pl.p2.y, pl.p2.z);
		float y3 = normalizePointY(pl.p3.y, pl.p3.z);
		float y4 = normalizePointY(pl.p4.y, pl.p4.z);

		drawPoligon(x1, y1, x2, y2, x3, y3, r, g, b);
		drawPoligon(x3, y3, x2, y2, x4, y4, r, g, b);
	}
}

void Device::drawMap_p(Map map, Camera cam)
{

}

void Device::ray_render(Map& map, Camera& cam, float angle)
{
	polygon* polygons = { new polygon[map.count_of_all_polygons] {} };
	int* colors = { new int[map.count_of_all_polygons] {} };


	Matrix hp(translate(cam.pos));
	Matrix m = rotateZ(180);
	Matrix l = rotateY(cam.angleX);
	Matrix k = rotateX(cam.angleY);
	m *= k;
	m *= l;
	m *= hp;

	int coo = map.count_of_objects;

	int counter = 0;
	for (int i = 0; i < coo; i++) {
		Object b = *(map.object + i);
		int cofp = b.mesh->count_of_polygons;
		for (int j = 0; j < cofp; j++) {
			polygon p = b.mesh->polygons[j];
			int color = b.mesh->colors_of_polygons[j];
			for (int k = 0; k < 3; k++) {

				p.facets[k] = multyply(p.facets[k], rotateY(b.rotation.x));
				p.facets[k] += b.position;
				p.facets[k] = multyply(p.facets[k], m);
			}
			polygons[counter] = p;
			colors[counter] = color;
			counter++;

		}
	}

	float3* host_end_poly = { new float3[3 * map.count_of_all_polygons] {} };
	float3* dev_end_poly = nullptr;

	for (int i = 0; i < map.count_of_all_polygons; i++) {
		polygon p = polygons[i];
		host_end_poly[i * 3 + 0] = make_float3(p.facets[0].x, p.facets[0].y, p.facets[0].z);
		host_end_poly[i * 3 + 1] = make_float3(p.facets[1].x, p.facets[1].y, p.facets[1].z);
		host_end_poly[i * 3 + 2] = make_float3(p.facets[2].x, p.facets[2].y, p.facets[2].z);
	}

	cudaMalloc((void**)&dev_end_poly, sizeof(float3) * 3 * map.count_of_all_polygons);
	cudaMemcpy(dev_end_poly, host_end_poly, sizeof(float3) * 3 * map.count_of_all_polygons, cudaMemcpyHostToDevice);

	int* dev_colors = nullptr;

	cudaMalloc((void**)&dev_colors, sizeof(int) * map.count_of_all_polygons);
	cudaMemcpy(dev_colors, colors, sizeof(int) * map.count_of_all_polygons, cudaMemcpyHostToDevice);

	float4* dev_lites = nullptr;

	float4* lites = { new float4[map.count_of_lites]{} };
	for (int i = 0; i < map.count_of_lites; i++) {
		Lite lite = *(map.lites + i);
		Vector3D pos = multyply(lite.position, m);
		lites[i] = make_float4(pos.x, pos.y, pos.z, lite.Intensity);
	}
	cudaMalloc((void**)&dev_lites, sizeof(float4) * map.count_of_lites);
	cudaMemcpy(dev_lites, lites, sizeof(float4) * map.count_of_lites, cudaMemcpyHostToDevice);

	int w = 1080 / 32;
	int h = dev_height / 32;
	if (dev_width % 32 != 0) w++;
	if (dev_height % 32 != 0) h++;

	dim3 blocks(w, h);
	dim3 theads(32, 32);

	ray_tracing << <blocks, theads >> > (dev_mem, dev_width, dev_height, dev_end_poly, dev_colors, map.count_of_all_polygons, dev_lites, map.count_of_lites);
	cudaFree(dev_end_poly);
	cudaFree(dev_colors);
	cudaFree(dev_lites);
	delete host_end_poly;
	delete polygons;
	delete colors;
	delete lites;
}

