#include "Camera.h"

Camera::Camera()
	:
	pos(-14, -10, -78), //pos = { x = -287.656006 y = -206.683044 z = 241.765259 ... }pos(-14, -10, -78)
	up(0, -1, 0),
	angleX(-33),
	angleY(-18)
{
}

Vector3D Camera::getDirection()
{
	Matrix m = rotateX(-angleY);
	m *= rotateY(-angleX);
	return multyply(Vector3D(0, 0, 1), m);
}
