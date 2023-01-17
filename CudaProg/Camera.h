#pragma once
#include "Vector.h"

class Camera {
public:
	Camera();


public:
	Vector3D pos;
	Vector3D direction;
	Vector3D up;
};
