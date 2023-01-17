#include "plane.h"


plane::plane(Vector3D p1, Vector3D p2, Vector3D p3, Vector3D p4, unsigned char r, unsigned char g, unsigned char b)
	:
	p1(p1),
	p2(p2),
	p3(p3),
	p4(p4),
	r(r),
	g(g),
	b(b)
{
}

plane::~plane()
{
}

plane plane::getPlane(float angle, Camera cam)
{
	//Matrix p(Look(Vector3D(cam.pos), Vector3D(cam.direction), Vector3D(1, 0, 0)));
	Matrix p (translate(cam.pos));
	Matrix m = rotateZ(180);
	Matrix l = rotateY(cam.angleX);
	Matrix k = rotateX(cam.angleY);
	m *= l;
	m *= p;
	

	plane pk(multyply(p1, m),
			 multyply(p2, m),
			 multyply(p3, m),
			 multyply(p4, m), r, g, b);
	 
	return pk;
}



