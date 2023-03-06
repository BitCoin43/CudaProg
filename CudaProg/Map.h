#pragma once
#include "plane.h"
#include "Object.h"

class Map {
public:
	Map(Object* obj, int countOfPolygons);
	Map();
	~Map();

	Object* object;
	int count_of_objects = 2;
	int count_of_all_polygons = 8;
	//void renderScene(Device dev, Camera cam);
};