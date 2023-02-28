#pragma once

class Graphics
{
public:
	Graphics(int WindowWidth, int WindowHeight);
	Graphics(const Graphics&) = delete;
	Graphics operator=(const Graphics&) = delete;
	~Graphics();

	void FillScreenFast(int* Colors, unsigned char r, unsigned char g, unsigned char b);
	void ClearScreenSuperFast(int* Colors);
	void DrawPixel(int* Colors, int x, int y, unsigned char r, unsigned char g, unsigned char b);
	void DrawAlphaPixel(int* Colors, int x, int y, unsigned char r, unsigned char g, unsigned char b, unsigned char a);
	void DrawRectancle(int* Colors, int xMin, int xMax, int yMin, int yMax, unsigned char r, unsigned char g, unsigned char b);
	void DrawAlphaRectangle(int* Colors, int xMin, int xMax, int yMin, int yMax, unsigned char r, unsigned char g, unsigned char b, unsigned char a);
	void DrawElips(int* Colors, int x0, int y0, int R, unsigned char r, unsigned char g, unsigned char b);
	void DrawLine(int* Colors, int x1, int x2, int y1, int y2, unsigned char r, unsigned char g, unsigned char b);
	int GetWindowWidth();
	int GetWindowHeight();

private:
	int WinWidth;
	int WinHeight;
};