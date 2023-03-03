#pragma once
#include "Polygon.h"

class StaticMesh {
public:
	StaticMesh(polygon* p, int count);
	int count_of_polygons;
	polygon* polygons;
};