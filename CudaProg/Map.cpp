#include "Map.h"

Map::Map()
{
	pl[0] = plane(Vector3D(-0.5, 0.5, 0), Vector3D(0.5, 0.5, 0), Vector3D(-0.5, -0.5, 0), Vector3D(0.5, -0.5, 0), 10, 200, 2);
	pl[1] = plane(Vector3D(-0.5, 0.5, 0), Vector3D(-0.5, 0.5, 1), Vector3D(-0.5, -0.5, 0), Vector3D(-0.5, -0.5, 1), 200, 10, 2);
	pl[2] = plane(Vector3D(-0.5, -0.5, 1), Vector3D(0.5, -0.5, 1), Vector3D(-0.5, -0.5, 0), Vector3D(0.5, -0.5, 0), 10, 10, 200);
	//pl[3] = plane(Vector3D(-3, 3, 1), Vector3D(3, 3, 1), Vector3D(-3, -3, 1), Vector3D(3, -3, 1), 60, 60, 60);
}
/*
void Map::renderScene(Device dev, Camera cam)
{
	Matrix p(translate(cam.pos));
	Matrix m = rotateZ(180);
	Matrix l = rotateY(cam.angleX);
	Matrix k = rotateX(cam.angleY);
	m *= l;
	m *= p;
	int u = sizeof(pl);
	for (int i = 0; i < 4; i++) {
		plane pk(multyply(pl[i].p1, m), multyply(pl[i].p2, m), multyply(pl[i].p3, m), multyply(pl[i].p4, m), 0, 0, 0);

		dev.drawPlane(pk, pl[i].r, pl[i].g, pl[i].b);
	}
}
*/