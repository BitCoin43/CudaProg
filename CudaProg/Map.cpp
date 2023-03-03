#include "Map.h"

Map::Map(Object* obj)
	:
	object(obj)
{
	pl[0] = plane(Vector3D(-0.5, 0.5, 0), Vector3D(0.5, 0.5, 0), Vector3D(-0.5, -0.5, 0), Vector3D(0.5, -0.5, 0), 10, 200, 2);
	pl[1] = plane(Vector3D(-0.5, 0.5, 0), Vector3D(-0.5, 0.5, 1), Vector3D(-0.5, -0.5, 0), Vector3D(-0.5, -0.5, 1), 200, 10, 2);
	pl[2] = plane(Vector3D(-0.5, -0.5, 1), Vector3D(0.5, -0.5, 1), Vector3D(-0.5, -0.5, 0), Vector3D(0.5, -0.5, 0), 10, 10, 200);
	
}

Map::Map()
{
}
