#include "Map.h"

Map::Map(Object* obj, int countOfPolygons, Lite* lites)
	:
	object(obj),
	count_of_all_polygons(countOfPolygons),
	lites(lites)
{
}

Map::~Map()
{
	//delete object;
}
