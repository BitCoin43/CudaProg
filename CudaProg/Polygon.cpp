#include "Polygon.h"

polygon::polygon(Vector3D p[3])
	:
	facets{ p[0], p[1], p[2] }
{
}

polygon::polygon(Vector3D p0, Vector3D p1, Vector3D p2)
	:
	facets{ p0, p1, p2 }
{
}

polygon::polygon()
	:
	facets{ Vector3D(0, 0, 0), Vector3D(0, 0, 0), Vector3D(0, 0, 0) }
{
}
