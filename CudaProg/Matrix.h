#pragma once
#include "Vector.h"
#include <cmath>

class Matrix {
public:
	double x [4][4];
	Matrix(float v);
	Matrix(const Matrix& m);
	Matrix& operator += (const Matrix& a);
	Matrix& operator -= (const Matrix& a);
	Matrix& operator *= (const Matrix& a);
	Matrix& operator *= (float f);
	Matrix& operator /= (float f);
	void invert();
	void tramspose();

};
Matrix translate(const Vector3D& loc);
Matrix scale(const Vector3D& v);
Matrix rotateX(float angle);
Matrix rotateY(float angle);
Matrix rotateZ(float angle);
Matrix rotate(const Vector3D& axis, float angle);
Matrix Look(const Vector3D& pos, const Vector3D& direction, const Vector3D& up);
Matrix mirrorX();
Matrix mirrorY();
Matrix mirrorZ();
Vector3D multyply(const Vector3D& v,const Matrix& m);
