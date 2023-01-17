#include "plane.h"
#include "Vector.h"

class Device {
public:
	Device(int height, int width, int* Colors);
	~Device();
	void copyDeviceToHost(int& Colors);
	void cleanDeviceMem(unsigned char r, unsigned char g, unsigned char b);
	void drawPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b);
	void drawCircle(int x, int y, int R, unsigned char r, unsigned char g, unsigned char b);
	void drawLine(int x0, int y0, int x1, int y1, unsigned char r, unsigned char g, unsigned char b);
	void drawPoligon(int x1, int y1, int x2, int y2, int x3, int y3, unsigned char r, unsigned char g, unsigned char b);
	float normalizePointX(float p, float z);
	float normalizePointY(float p, float z);
	void drawPlane(plane pl, unsigned char r, unsigned char g, unsigned char b);
private:
	int* dev_mem;
	int dev_height;
	int dev_width;
};


