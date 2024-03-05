#pragma once

class Vector3D {
public:
	Vector3D(float x, float y, float z);
	Vector3D(float x, float y, float z, float w);
	Vector3D(const Vector3D& v);
	Vector3D& operator += (const Vector3D& v);
	Vector3D& operator -= (const Vector3D& v);
	Vector3D& operator *= (const Vector3D& v);
	Vector3D& operator /= (const Vector3D& v);
	Vector3D& operator *= (float v);
	Vector3D& operator /= (float v);

	Vector3D getRightVector(Vector3D v);
	
	void normal();
	float length();

	float x;
	float y;
	float z;
	float w;
};

Vector3D crossProduct(const Vector3D& v, const Vector3D& k);

Vector3D subtraction(const Vector3D& v, const Vector3D& k);

Vector3D normalize(const Vector3D& v);

Vector3D multiple(Vector3D v, float f);

Vector3D addVector3D(Vector3D v1, Vector3D v2);

float dot(Vector3D v1, Vector3D v2);



