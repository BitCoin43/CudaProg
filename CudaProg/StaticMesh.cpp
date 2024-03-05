#include "StaticMesh.h"

StaticMesh::StaticMesh(polygon* p, int* colors, int count)
	:
	count_of_polygons(count),
	colors_of_polygons(colors),
	polygons(p)
{
}

StaticMesh::~StaticMesh()
{
	//delete polygons;
	//delete colors_of_polygons;
}
