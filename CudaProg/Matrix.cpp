#include "Matrix.h"


Matrix::Matrix(float v)
{
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			x[i][j] = (i == j) ? v : 0.0;
		}
	}
}

Matrix::Matrix(const Matrix& m)
{
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			x[i][j] = m.x[i][j];
		}
	}
}

Matrix& Matrix::operator+=(const Matrix& a)
{
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			x[i][j] += a.x[i][j];
		}
	}
	return *this;
}

Matrix& Matrix::operator-=(const Matrix& a)
{
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			x[i][j] -= a.x[i][j];
		}
	}
	return *this;
}

Matrix& Matrix::operator*=(const Matrix& a)
{
	Matrix res(*this);
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			float sum = 0;
			for (int k = 0; k < 4; k++){
				sum += res.x[i][k] * a.x[k][j];
			}
			x[i][j] = sum;
		}
	}
	return *this;
}

Matrix& Matrix::operator*=(float f)
{
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			x[i][j] *= f;
		}
	}
	return *this;
}

Matrix& Matrix::operator/=(float f)
{
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			x[i][j] /= f;
		}
	}
	return *this;
}

void Matrix::invert()
{
	Matrix out(1);
	for (int i = 0; i < 4; i++) {
		float d = x[i][i];
		if (d != 1.0) {
			for (int j = 0; j < 4; j++) {
				out.x[i][j] /= d;
				x[i][j] /= d;
			}
		}
		for (int j = 0; j < 4; j++) {
			if (j != i) {
				if (x[j][i] != 0.0) {
					float mulBy = x[j][i];
					for (int k = 0; k < 4; k++) {
						x[j][k] -= mulBy * x[i][k];
						out.x[j][k] -= mulBy * out.x[i][k];
					}
				}
			}
		}
	}
	*this = out;
}

void Matrix::tramspose()
{
	float t;
	for (int i = 0; i < 4; i++) {
		for (int j = i; j < 4; j++) {
			if (i != j) {
				t = x[i][j];
				x[i][j] = x[j][i];
				x[j][i] = t;
			}
		}
	}
}

Matrix translate(const Vector3D& loc)
{
	Matrix res(1);
	res.x[0][3] = loc.x;
	res.x[1][3] = loc.y;
	res.x[2][3] = loc.z;
	return res;
}

Matrix scale(const Vector3D& v)
{
	Matrix res(1);
	res.x[0][0] = v.x;
	res.x[1][1] = v.y;
	res.x[2][2] = v.z;
	return res;
}

Matrix rotateX(float angle)
{
	float a = 3.1415926535 / 180 * angle;
	Matrix res(1);
	float cosine = cos(a);
	float sine = sin(a);
	res.x[1][1] = cosine;
	res.x[1][2] = -sine;
	res.x[2][1] = sine;
	res.x[2][2] = cosine;
	return res;
}

Matrix rotateY(float angle)
{
	float a = 3.1415926535 / 180 * angle;
	Matrix res(1);
	float cosine = cos(a);
	float sine = sin(a);
	res.x[0][0] = cosine;
	res.x[0][2] = -sine;
	res.x[2][0] = sine;
	res.x[2][2] = cosine;
	return res;
}

Matrix rotateZ(float angle)
{
	float a = 3.1415926535 / 180 * angle;
	Matrix res(1);
	float cosine = cos(a);
	float sine = sin(a);
	res.x[0][0] = cosine;
	res.x[0][1] = -sine;
	res.x[1][0] = sine;
	res.x[1][1] = cosine;
	return res;
}

Matrix rotate(const Vector3D& axis, float angle)
{
	float a = 3.1415926535 / 180 * angle;
	Matrix res(1);
	float cosine = cos(a);
	float sine = sin(a);

	res.x[0][0] = axis.x * axis.x * (1 - axis.x * axis.x) * cosine;
	res.x[1][0] = axis.x * axis.y * (1 - cosine) + axis.z * sine;
	res.x[2][0] = axis.x * axis.z * (1 - cosine) - axis.y * sine;
	res.x[3][0] = 0;

	res.x[0][1] = axis.x * axis.y * (1 - cosine) - axis.z * sine;
	res.x[1][1] = axis.y * axis.y + (1 - axis.y * axis.y) * cosine;
	res.x[2][1] = axis.y * axis.z * (1 - cosine) + axis.x * sine;
	res.x[3][1] = 0;

	res.x[0][2] = axis.x * axis.z * (1 - cosine) + axis.y * sine;
	res.x[1][2] = axis.y * axis.z * (1 - cosine) - axis.x * sine;
	res.x[2][2] = axis.z * axis.z + (1 - axis.z * axis.z) * cosine;
	res.x[3][2] = 0;

	res.x[0][3] = 0;
	res.x[1][3] = 0;
	res.x[2][3] = 0;
	res.x[3][3] = 1;
	return res;
}

Matrix Look(const Vector3D& pos, const Vector3D& direction, const Vector3D& up)
{
	Vector3D vz = normalize(subtraction(pos, direction));
	Vector3D vx = normalize(crossProduct(up, vz));
	Vector3D vy = normalize(crossProduct(vz, vx));

	Matrix t = translate(Vector3D(-pos.x, -pos.y, -pos.z));

	Matrix m(1);
	m.x[0][0] = vx.x;	m.x[1][0] = vx.y;	m.x[2][0] = vx.z;

	m.x[0][1] = vy.x;	m.x[1][1] = vy.y;	m.x[2][1] = vy.z;

	m.x[0][2] = vz.x;	m.x[1][2] = vz.y;	m.x[2][2] = vz.z;

	t *= m;

	return t;
}

Matrix mirrorX()
{
	Matrix res(1);
	res.x[0][0] = -1;
	return res;
}

Matrix mirrorY()
{
	Matrix res(1);
	res.x[1][1] = -1;
	return res;
}

Matrix mirrorZ()
{
	Matrix res(1);
	res.x[2][2] = -1;
	return res;
}

Vector3D multyply(const Vector3D& v, const Matrix& m)
{
	return Vector3D(
		m.x[0][0] * v.x + m.x[0][1] * v.y + m.x[0][2] * v.z + m.x[0][3] * v.w,
		m.x[1][0] * v.x + m.x[1][1] * v.y + m.x[1][2] * v.z + m.x[1][3] * v.w,
		m.x[2][0] * v.x + m.x[2][1] * v.y + m.x[2][2] * v.z + m.x[2][3] * v.w,
		m.x[3][0] * v.x + m.x[3][1] * v.y + m.x[3][2] * v.z + m.x[3][3] * v.w
	);
}
