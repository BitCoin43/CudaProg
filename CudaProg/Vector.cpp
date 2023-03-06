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

Vector3D::Vector3D(float x, float y, float z, float w)
	:
	x(x),
	y(y),
	z(z),
	w(w)
{
}

Vector3D::Vector3D(const Vector3D& v)
	:
	x(v.x),
	y(v.y),
	z(v.z)
{
	w = 1;
}

Vector3D& Vector3D::operator+=(const Vector3D& v)
{
	x += v.x;
	y += v.y;
	z += v.z;
	return *this;
}

Vector3D& Vector3D::operator-=(const Vector3D& v)
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
	return *this;
}

Vector3D& Vector3D::operator*=(const Vector3D& v)
{
	x *= v.x;
	y *= v.y;
	z *= v.z;
	return *this;
}

Vector3D& Vector3D::operator/=(const Vector3D& v)
{
	x /= v.x;
	y /= v.y;
	z /= v.z;
	return *this;
}

Vector3D& Vector3D::operator*=(float v)
{
	x *= v;
	y *= v;
	z *= v;
	return *this;
}

Vector3D& Vector3D::operator/=(float v)
{
	x /= v;
	y /= v;
	z /= v;
	return *this;
}

Vector3D Vector3D::getRightVector(Vector3D v)
{
	return Vector3D(
		y * v.z - z * v.y,
		z * v.x - x * v.z,
		x * v.y - y * v.x,
		1);
}

void Vector3D::normal()
{
	float l = this->length();
	x /= l;
	z /= l;
	x /= l;
}

float Vector3D::length()
{
	return sqrt(x * x + y * y + z * z);
}

Vector3D operator*(Vector3D& v, float& f)
{
	v *= f;
	return v;
}

Vector3D crossProduct(const Vector3D& v1, const Vector3D& v2)
{
	
	return Vector3D(
		v1.y * v2.z - v1.z * v2.y,
		v1.z * v2.x - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x,
		1);
	
}

Vector3D subtraction(const Vector3D& v1, const Vector3D& v2)
{
	return  Vector3D(
		v1.x - v2.x,
		v1.y - v2.y,
		v1.z - v2.z,
		1);
}

Vector3D normalize(const Vector3D& v)
{
	Vector3D f(v);
	f /= f.length();
	return f;
}

Vector3D multiple(Vector3D v, float f)
{
	v *= f;
	return v;
}

Vector3D addVector3D(Vector3D v1, Vector3D v2)
{
	return Vector3D(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

float dot(Vector3D v1, Vector3D v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
