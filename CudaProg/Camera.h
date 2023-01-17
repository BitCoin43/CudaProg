#pragma once
#include "Matrix.h"

class Camera {
public:
	Camera();
	Vector3D getDirection();

public:
	Vector3D pos;
	//Vector3D direction;
	Vector3D up;
	float angleX;
	float angleY;
};
