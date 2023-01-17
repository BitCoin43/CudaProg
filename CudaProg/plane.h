#pragma once
#include "Matrix.h"
#include "Camera.h"

class plane {
public:
	plane(Vector3D p1, Vector3D p2, Vector3D p3, Vector3D p4, unsigned char r, unsigned char g, unsigned char b);
	~plane();
	plane getPlane(float angle, Camera cam);
public:
	Vector3D p1;
	Vector3D p2;
	Vector3D p3;
	Vector3D p4;
	unsigned char r;
	unsigned char g;
	unsigned char b;
};