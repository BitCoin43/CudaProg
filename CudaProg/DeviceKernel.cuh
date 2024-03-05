#include "Map.h"
#include "Vector.h"
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

class Node {
public:
	Node(int poly, float3 sphere, float radius) : sphere(sphere), radius(radius), poly(poly) {};
	Node(float3 sphere, float radius, Node* n, int count_of_child) : sphere(sphere), radius(radius), child(n), count_of_child(count_of_child) {};
	Node() {};
	~Node() {};
	float3 sphere = {};
	float radius = 0;
	int count_of_child = 0;
	Node* child = nullptr;
	int poly = -1;
};

class Device {
public:
	Device(int height, int width, int* Colors);
	~Device();
	void copyDeviceToHost(int& Colors);
	void cleanDeviceMem(unsigned char r, unsigned char g, unsigned char b);
	void drawPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b);
	void drawPoligon(int x1, int y1, int x2, int y2, int x3, int y3, unsigned char r, unsigned char g, unsigned char b);
	float normalizePointX(float p, float z)
	{
		if (z != 0) p /= -(z * 0.5);
		return dev_width / 2 + dev_height / 2 * p;
	}
	float normalizePointY(float p, float z)
	{
		if (z != 0) p /= (-z * 0.5);
		return dev_height / 2 + dev_height / 2 * p;
	}
	void activateMap(Map& map, Camera& cam);
	void ray_render(Map& map, Camera& cam, float angle);
	void path_tracing(Map& map, Camera& cam, float angle);
	void buil_sphere_tree(Map& map, Camera& cam);

public:
	int* dev_mem;
	int* dev_colors = nullptr;
	float4* dev_lites = nullptr;
	float3* dev_poly = nullptr;
	//float3* dev_end_poly = nullptr;
	int dev_height;
	int dev_width;
	float* intersection_matrix = nullptr;
	float3* ray_matrix_direction = nullptr;
	float3* ray_matrix_origin = nullptr;
public:
	Node* main_tree = nullptr;
};

/*void uploadLites(Map& map, Matrix m, float4* dev_lites) {
	float4* lites = { new float4[map.count_of_lites]{} };
	for (int i = 0; i < map.count_of_lites; i++) {
		Lite lite = *(map.lites + i);
		Vector3D pos = multyply(lite.position, m);
		lites[i] = make_float4(pos.x, pos.y, pos.z, lite.Intensity);
	}
	cudaMalloc((void**)&dev_lites, sizeof(float4) * map.count_of_lites);
	cudaMemcpy(dev_lites, lites, sizeof(float4) * map.count_of_lites, cudaMemcpyHostToDevice);
}*/
