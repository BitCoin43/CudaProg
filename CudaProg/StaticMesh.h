#pragma once
#include "Polygon.h"

class StaticMesh {
public:
	StaticMesh(polygon* p, int* colors, int count);
	~StaticMesh();
	int count_of_polygons;
	polygon* polygons;
	int* colors_of_polygons = nullptr;
};