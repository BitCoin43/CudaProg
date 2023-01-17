#include "Camera.h"

Camera::Camera()
	:
	pos(0, 0, 3),
	up(0, -1, 0),
	angleX(0),
	angleY(0)
{
}

Vector3D Camera::getDirection()
{
	Matrix m = rotateX(-angleY);
	m *= rotateY(-angleX);
	return multyply(Vector3D(0, 0, 1), m);
}
