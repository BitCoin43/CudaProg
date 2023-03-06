#include "Object.h"

Object::Object(Vector3D position, Vector3D rotation, StaticMesh* mesh)
	:
	position(position),
	rotation(rotation),
	mesh(mesh)
{
}

Object::~Object()
{
	//delete mesh;
}
