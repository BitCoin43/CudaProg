#pragma once
#include "plane.h"
#include "Object.h"
#include "Lite.h"

class Map {
public:
	Map(Object* obj, int countOfPolygons, int countOfObjects, Lite* lites);
	~Map();

	Object* object;
	int count_of_objects = 5;
	int count_of_all_polygons =  12;

	Lite* lites;
	int count_of_lites = 2;
};