#pragma once
#include "plane.h"
#include "Object.h"

class Map {
public:
	Map(Object* obj);
	Map();
	plane pl[3];

	Object* object;
	int count_of_objects = 1;
	int count_of_all_polygons = 6;
	//void renderScene(Device dev, Camera cam);
};