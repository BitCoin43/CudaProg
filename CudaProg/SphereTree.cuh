#include "Render.cuh"

class Node {
public:
    Node(int poly, float3 sphere, float radius) : sphere(sphere), radius(radius), poly(poly){};
    Node(float3 sphere, float radius, Node* n, int count_of_child) : sphere(sphere), radius(radius), child(n), count_of_child(count_of_child){};
    ~Node() {};
	float3 sphere;
    float radius;
    int count_of_child = 0;
	Node* child = nullptr;
	int poly = -1;
};

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


