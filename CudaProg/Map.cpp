#include "Map.h"

Map::Map(Object* obj, int countOfPolygons)
	:
	object(obj),
	count_of_all_polygons(countOfPolygons)
{
}

Map::Map()
{
}

Map::~Map()
{
	//delete object;
}
