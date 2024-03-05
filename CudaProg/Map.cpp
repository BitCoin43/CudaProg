#include "Map.h"

Map::Map(Object* obj, int countOfPolygons, int countOfObjects, Lite* lites)
	:
	object(obj),
	count_of_all_polygons(countOfPolygons),
	lites(lites),
	count_of_objects(countOfObjects)
{
}

Map::~Map()
{
	//delete object;
}
