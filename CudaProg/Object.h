#pragma once
#include "StaticMesh.h"

class Object {
public:
	Object(Vector3D position, Vector3D rotation, StaticMesh* mesh);
	Vector3D position;
	Vector3D rotation;
	StaticMesh* mesh;
};