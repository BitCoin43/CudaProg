#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "DeviceKernel.cuh"
#include <cmath>

__device__ void DPdev(int* dev_mem, int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b) {
	if (x >= 0 && x < width && y >= 0 && y < height) {
		dev_mem[y * width + x] = (r << 16) | (g << 8) | b;
	}
}

__global__ void DPglobal(int* dev_mem, int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b) {
	if (x >= 0 && x < width && y >= 0 && y < height) {
		dev_mem[y * width + x] = (r << 16) | (g << 8) | b;
	}
}

__global__ void renderCircle(int* dev_mem, int x1, int y1, int R, int width, int height, unsigned char r, unsigned char g, unsigned char b) {
	//DPdev(dev_mem, x + blockIdx.x, y + blockIdx.y, width, height, r, g, b);
	int x = blockIdx.x - R; int y = blockIdx.y - R;
	if (x * x + y * y <= (R - 0.5) * (R - 0.5)) {
		DPdev(dev_mem, blockIdx.x + x1, blockIdx.y + y1, width, height, r, g, b);
	}
}

__global__ void renderLine(int* dev_mem, const int width, const int height, const float x0, const float y0, const float x1, const float y1, unsigned char r, unsigned char g, unsigned char b
)
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

__global__ void clean(int* dev_mem, int width, int height, unsigned char r, unsigned char g, unsigned char b) {
	dev_mem[blockIdx.y * width + blockIdx.x] = (r << 16) | (g << 8) | b;
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

	if (x1 < x2 && x1 < x3) xmin = x1;
	if (x2 < x1 && x2 < x3) xmin = x2;
	if (x3 < x2 && x3 < x1) xmin = x3;

	if (y1 < y2 && y1 < y3)	ymin = y1;
	if (y2 < y1 && y2 < y3)	ymin = y2;
	if (y3 < y2 && y3 < y1)	ymin = y3;

	if (x1 > x2 && x1 > x3) xmax = x1;
	if (x2 > x1 && x2 > x3) xmax = x2;
	if (x3 > x2 && x3 > x1) xmax = x3;
							 
	if (y1 > y2 && y1 > y3)	ymax = y1;
	if (y2 > y1 && y2 > y3)	ymax = y2;
	if (y3 > y2 && y3 > y1)	ymax = y3;

	dim3 grid(xmax - xmin, ymax - ymin);
	renderPoligon << <grid, 1 >> > (dev_mem, xmin, ymin, x1, y1, x2, y2, x3, y3, dev_width, dev_height, r, g, b);
}
