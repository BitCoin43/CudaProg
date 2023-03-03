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

__device__ inline void DP_ARGB_dev(int* dev_mem, int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
	if (x >= 0 && x < width && y >= 0 && y < height){
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
	
	if ((a >= 0 && B >= 0 && c >= 0) || (a <= 0 && B <= 0 && c <= 0)){
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

__global__ void getInt_dev(int* dev_mem, int *c, int width, int x, int y) {
	*c = dev_mem[y * width + x];
}

__global__ void ray_tracing(int* dev_mem, int width, int height, polygon* poly) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x <= width && y <= height) {
		DP_ARGB_dev(dev_mem, x, y, width, height, 0, 0, 255, 100);
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
	clean <<<grid, 1 >>> (dev_mem, dev_width, dev_height, r, g, b);
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
	renderLine <<<grid, 1>>>(dev_mem, dev_width, dev_height, x0, y0, x1, y1, r, g, b);
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
	renderRect_dev<<<grid, 1>>>(dev_mem, x1, y1, dev_width, dev_height, r, g, b, a);
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

void Device::drawMap(Map map, Camera cam)
{
	Matrix p(translate(cam.pos));
	Matrix m = rotateZ(180);
	Matrix l = rotateY(cam.angleX);
	Matrix k = rotateX(cam.angleY);
	m *= l;
	m *= p;

	plane mpl [3];

	for (int i = 0; i < 3; i++) {
		mpl[i] = plane(
			multyply(map.pl[i].p1, m),
			multyply(map.pl[i].p2, m),
			multyply(map.pl[i].p3, m),
			multyply(map.pl[i].p4, m), map.pl[i].r, map.pl[i].g, map.pl[i].b);
	}

	float f1[3] = {};
	for (int i = 0; i < 3; i++) {
		f1[i] = (mpl[i].p1.z + mpl[i].p2.z + mpl[i].p3.z + mpl[i].p4.z) / 4;
	}

	float f2[3] = { f1[0], f1[1], f1[2]};
	std::sort(std::begin(f2), std::end(f2), std::greater<>());


	int f3[3] = {};
	for (int i = 0; i < 3; i++) {
		float count = 0;
		for (int j = 0; j < 4; j++) {
			if (f2[j] == f1[i]) {
				count = j;
			}
		}
		f3[i] = count;
	}

	for (int i = 0; i < 3; i++) {
		drawPlane(mpl[f3[i]], mpl[f3[i]].r, mpl[f3[i]].g, mpl[f3[i]].b);
	}
}

void Device::drawMap_p(Map map, Camera cam)
{

}

void Device::ray_render(Map map, Camera cam)
{
	polygon* polygons = { new polygon[map.count_of_all_polygons] {} };

	Matrix hp(translate(cam.pos));
	Matrix m = rotateZ(180);
	Matrix l = rotateY(cam.angleX);
	Matrix k = rotateX(cam.angleY);
	m *= l;
	m *= hp;

	int coo = map.count_of_objects;

	for (int i = 0; i < coo; i++) {
		Object b = *(map.object + i);
		int cofp = b.mesh->count_of_polygons;
		for (int j = 0; j < cofp; j++) {
			polygon p = b.mesh->polygons[j];
			for (int k = 0; k < 3; k++) {
				p.facets[k] += b.position;
				p.facets[k] = multyply(p.facets[k], m);
			}
			polygons[(i * b.mesh->count_of_polygons) + j] = p;
		}
	}

	polygon* end_poly = nullptr;

	cudaMalloc((void**)&end_poly, sizeof(polygon) * map.count_of_all_polygons);

	for (int i = 0; i < map.count_of_all_polygons; i++) {
		polygon p = *(polygons + i);

		polygon p2 = *(polygons + 0);
		polygon p3 = *(polygons + 1);

		float x1 = normalizePointX(p.facets[0].x, p.facets[0].z);
		float x2 = normalizePointX(p.facets[1].x, p.facets[1].z);
		float x3 = normalizePointX(p.facets[2].x, p.facets[2].z);
								   
		float y1 = normalizePointY(p.facets[0].y, p.facets[0].z);
		float y2 = normalizePointY(p.facets[1].y, p.facets[1].z);
		float y3 = normalizePointY(p.facets[2].y, p.facets[2].z);


		drawPoligon(x1, y1, x2, y2, x3, y3, 255, 90, 250);
	}

	int w = dev_width / 32;
	int h = dev_height / 32;
	if (dev_width % 32 != 0) w++;
	if (dev_height % 32 != 0) h++;

	dim3 blocks(w, h);
	dim3 theads(32, 32);

	ray_tracing<<<blocks, theads>>>(dev_mem, dev_width, dev_height, end_poly);
}

