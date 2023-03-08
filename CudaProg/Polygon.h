#pragma once
#include "Vector.h"

class polygon {
public:
	polygon(Vector3D p[3]);
	polygon(Vector3D p0, Vector3D p1, Vector3D p2);
	polygon();
	Vector3D facets[3];
};
