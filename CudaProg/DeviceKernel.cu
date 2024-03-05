#include "DeviceKernel.cuh"
#include "cuda3dmath.cuh"

Device::Device(int height, int width, int* Colors) :dev_width(width), dev_height(height)
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

void Device::cleanDeviceMem(unsigned char r, unsigned char g, unsigned char b) {
	dim3 grid(dev_width, dev_height);
	clean << < grid, 1 >> > (dev_mem, dev_width, dev_height, r, g, b);
}

void Device::drawPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
	DPglobal << < 1, 1 >> > (dev_mem, x, y, dev_width, dev_height, r, g, b);
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

void Device::activateMap(Map& map, Camera& cam) {
	Matrix hp(translate(cam.pos));
	Matrix m = rotateZ(180);
	Matrix l = rotateY(cam.angleX);
	Matrix k = rotateX(cam.angleY);
	m *= k;
	m *= l;
	m *= hp;

	int* colors = { new int[map.count_of_all_polygons] {} };

	int counter = 0;
	for (int i = 0; i < map.count_of_objects; i++) {
		Object b = *(map.object + i);
		int cofp = b.mesh->count_of_polygons;
		for (int j = 0; j < cofp; j++) {
			colors[counter] = b.mesh->colors_of_polygons[j];
			counter++;

		}
	}
	float4* lites = { new float4[map.count_of_lites]{} };
	for (int i = 0; i < map.count_of_lites; i++) {
		Lite lite = *(map.lites + i);
		Vector3D pos = multyply(lite.position, m);
		lites[i] = make_float4(pos.x, pos.y, pos.z, lite.Intensity);
	}
	cudaMalloc((void**)&dev_lites, sizeof(float4) * map.count_of_lites);
	cudaMemcpy(dev_lites, lites, sizeof(float4) * map.count_of_lites, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&intersection_matrix, sizeof(float) * map.count_of_all_polygons * dev_width * dev_height);
	cudaMalloc((void**)&ray_matrix_direction, sizeof(float3) * dev_width * dev_height);
	cudaMalloc((void**)&ray_matrix_origin, sizeof(float3) * dev_width * dev_height);

	cudaMalloc((void**)&dev_colors, sizeof(int) * map.count_of_all_polygons);
	cudaMemcpy(dev_colors, colors, sizeof(int) * map.count_of_all_polygons, cudaMemcpyHostToDevice);
	polygon* polygons = { new polygon[map.count_of_all_polygons] {} };
	//int* colors = { new int[map.count_of_all_polygons] {} };




	//std::thread th(uploadLites, std::ref(map), m, std::cref(dev_lites));


	int coo = map.count_of_objects;

	int counter2 = 0;
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
			polygons[counter2] = p;
			counter2++;

		}
	}
	float3* host_end_poly = { new float3[3 * map.count_of_all_polygons] {} };
	dev_poly = nullptr;

	for (int i = 0; i < map.count_of_all_polygons; i++) {
		polygon p = polygons[i];
		host_end_poly[i * 3 + 0] = make_float3(p.facets[0].x, p.facets[0].y, p.facets[0].z);
		host_end_poly[i * 3 + 1] = make_float3(p.facets[1].x, p.facets[1].y, p.facets[1].z);
		host_end_poly[i * 3 + 2] = make_float3(p.facets[2].x, p.facets[2].y, p.facets[2].z);
	}

	cudaMalloc((void**)&dev_poly, sizeof(float3) * 3 * map.count_of_all_polygons);
	cudaMemcpy(dev_poly, host_end_poly, sizeof(float3) * 3 * map.count_of_all_polygons, cudaMemcpyHostToDevice);
	delete[] colors;
}

void Device::ray_render(Map& map, Camera& cam, float angle) {
	int r = 32;
	int w = dev_width / r;
	int h = dev_height / r;
	if (dev_width % r != 0) w++;
	if (dev_height % r != 0) h++;

	dim3 blocks(w, h);
	dim3 theads(r, r);

	ray_tracing << < blocks, theads >> > (dev_mem, dev_width, dev_height, dev_poly,
		dev_colors, map.count_of_all_polygons, dev_lites, map.count_of_lites);
}

void Device::path_tracing(Map& map, Camera& cam, float angle) {
	Matrix hp(translate(cam.pos));
	Matrix m = rotateZ(180);
	Matrix l = rotateY(cam.angleX);
	Matrix k = rotateX(cam.angleY);
	m *= k;
	m *= l;
	m *= hp;

	//========================================================================
	float4* lites = { new float4[map.count_of_lites]{} };
	for (int i = 0; i < map.count_of_lites; i++) {
		Lite lite = *(map.lites + i);
		Vector3D pos = multyply(lite.position, m);
		lites[i] = make_float4(pos.x, pos.y, pos.z, lite.Intensity);
	}
	cudaMalloc((void**)&dev_lites, sizeof(float4) * map.count_of_lites);
	cudaMemcpy(dev_lites, lites, sizeof(float4) * map.count_of_lites, cudaMemcpyHostToDevice);
	//========================================================================

	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	int coo = map.count_of_objects;
	polygon* polygons = { new polygon[map.count_of_all_polygons] {} };
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
			counter++;

		}
	}
	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	float3* host_end_poly = { new float3[3 * map.count_of_all_polygons] {} };
	float3* dev_end_poly = nullptr;

	for (int i = 0; i < map.count_of_all_polygons; i++) {
		polygon p = polygons[i];
		host_end_poly[i * 3 + 0] = make_float3(p.facets[0].x, p.facets[0].y, p.facets[0].z);
		host_end_poly[i * 3 + 1] = make_float3(p.facets[1].x, p.facets[1].y, p.facets[1].z);
		host_end_poly[i * 3 + 2] = make_float3(p.facets[2].x, p.facets[2].y, p.facets[2].z);
	}
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	cudaMalloc((void**)&dev_end_poly, sizeof(float3) * 3 * map.count_of_all_polygons);
	cudaMemcpy(dev_end_poly, host_end_poly, sizeof(float3) * 3 * map.count_of_all_polygons, cudaMemcpyHostToDevice);

	int w = dev_width / 32;
	int h = dev_height / 32;
	if (dev_width % 32 != 0) w++;
	if (dev_height % 32 != 0) h++;

	dim3 blocks(w, h);
	dim3 theads(32, 32);

	//newtrace << < blocks, theads >> > (intersection_matrix, ray_matrix_direction, dev_width, dev_height, dev_end_poly, 0, map.count_of_all_polygons);




	//ray_tracing << <blocks, theads >> > (dev_mem, dev_width, dev_height, dev_end_poly, dev_colors, map.count_of_all_polygons, dev_lites, map.count_of_lites);
	delete[] host_end_poly;
	delete[] polygons;
}

void findCircumSphere(const float3& A, const float3& B, const float3& C, Node& n) {
	float3 a = make_float3(B.x - A.x, B.y - A.y, B.z - A.z);
	float3 b = make_float3(C.x - A.x, C.y - A.y, C.z - A.z);
	float3 c = make_float3(a.x * b.y - a.y * b.x, a.z * b.x - a.x * b.z, a.y * b.z - a.z * b.y);
	float d = -1 * (c.x * A.x + c.y * A.y + c.z * A.z);
	float x = -1 * c.x / (2 * c.z);
	float y = -1 * c.y / (2 * c.z);
	float z = -1 * (c.x * x + c.y * y + d) / c.z;
	float3 center = make_float3(x, y, z);
	float radius = sqrt(pow(A.x - x, 2) + pow(A.y - y, 2) + pow(A.z - z, 2));
	n.sphere = center;
	n.radius = radius;
}

float3 getMidPoint(const float3& a, const float3& b) {
	return make_float3( (a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2 );
}

void findCircumsphere_2(const float3& A, const float3& B, const float3& C, Node& n) {
	float3 AB = { B.x - A.x, B.y - A.y, B.z - A.z };
	float3 AC = { C.x - A.x, C.y - A.y, C.z - A.z };

	float3 N = cross(AB, AC);
	double D = -dot(N, A);

	float3 M_AB = getMidPoint(A, B);
	float3 M_BC = getMidPoint(B, C);
	
	float3 M_AB_N = cross(N, cross(AB, N));
	float3 M_BC_N = cross(N, cross(M_BC - B, N));

	double det = M_AB_N.x * M_BC_N.y - M_AB_N.y * M_BC_N.x;

	double t = (M_BC_N.y * (M_AB.x - M_BC.x) - M_BC_N.x * (M_AB.y - M_BC.y)) / det;
	float3 center = { M_AB.x + t * M_AB_N.x, M_AB.y + t * M_AB_N.y, M_AB.z + t * M_AB_N.z };
	double radius = sqrt((A.x - center.x, 2) + pow(A.y - center.y, 2) + pow(A.z - center.z, 2));

	n.sphere = center;
	n.radius = radius;
}

void Device::buil_sphere_tree(Map& map, Camera& cam)
{
	Matrix hp(translate(cam.pos));
	Matrix m = rotateZ(180);
	Matrix l = rotateY(cam.angleX);
	Matrix k = rotateX(cam.angleY);
	m *= k;
	m *= l;
	m *= hp;
	int coo = map.count_of_objects;
	polygon* polygons = { new polygon[map.count_of_all_polygons] {} };
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
			counter++;
		}
	}
	Node* nodes = new Node[map.count_of_all_polygons]{};
	for (int i = 0; i < map.count_of_all_polygons; i++) {
		float3 sphere = make_float3(0, 0, 0);
		Node n(i, sphere, 8);
		float3 p1 = make_float3(polygons[i].facets[0].x, polygons[i].facets[0].y, polygons[i].facets[0].z);
		float3 p2 = make_float3(polygons[i].facets[1].x, polygons[i].facets[1].y, polygons[i].facets[1].z);
		float3 p3 = make_float3(polygons[i].facets[2].x, polygons[i].facets[2].y, polygons[i].facets[2].z);
		findCircumsphere_2(p1, p2, p3, n);
		nodes[i] = n;
	}
	Node n0();
	








	delete[] polygons;
}