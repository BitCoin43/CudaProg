#include "Vector.h"
#include <cmath>

Vector3D::Vector3D(float x, float y, float z)
	:
	x(x),
	y(y),
	z(z)
{
	w = 1;
}
