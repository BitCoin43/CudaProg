#pragma once
#include "plane.h"
#include "Object.h"
#include "Lite.h"

class Map {
public:
	Map(Object* obj, int countOfPolygons, Lite* lites);
	~Map();

	Object* object;
	int count_of_objects = 2;
	int count_of_all_polygons = 8;

	Lite* lites;
	int count_of_lites = 1;
};