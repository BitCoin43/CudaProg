#include "Graphics.h"
#include <iostream>
#include <assert.h>

Graphics::Graphics(int WindowWidth, int WindowHeight)
	:
	WinWidth(WindowWidth),
	WinHeight(WindowHeight)
{
}

Graphics::~Graphics()
{
}

void Graphics::FillScreenFast(int* Colors, unsigned char r, unsigned char g, unsigned char b)
{
	for (int* i = Colors, *end = &Colors[WinWidth * WinHeight]; i < end; i++)
	{
		*i = (r << 16) | (g << 8) | b;
	}
}

void Graphics::ClearScreenSuperFast(int* Colors)
{
	std::memset(Colors, 0, WinWidth * WinHeight * 4);
}

void Graphics::DrawPixel(int* Colors, int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
	if (x > 0 && x < WinWidth && y > 0 && y < WinHeight)
	{
		Colors[y * WinWidth + x] = (r << 16) | (g << 8) | b;
	}
}

void Graphics::DrawRectancle(int* Colors, int xMin, int xMax, int yMin, int yMax, unsigned char r, unsigned char g, unsigned char b)
{
	for (int y = yMin; y < yMax; y++)
	{
		for (int x = xMin; x < xMax; x++)
		{
			DrawPixel(Colors, x, y, r, g, b);
		}
	}
}

void Graphics::DrawElips(int* Colors, int x0, int y0, int R, unsigned char r, unsigned char g, unsigned char b)
{
	int x = 0;
	int y = R;
	int delta = 1 - 2 * R;
	int error = 0;
	bool wt = true;
	int q1 = x0;
	int q2 = x0;
	while (y >= 0)
	{
		DrawPixel(Colors, x0 + x, y0 + y, r, g, b);
		if (wt)
		{
			q1 = x0 + x;
			wt = false;
		}
		else
		{
			q2 = x0 + x;
			wt = false;
		}

		if (q2 - q1 != 1)
		{

		}


		if (true)
		{
			int x1 = 0;
			while (x1 < R + 1)
			{
				if ((x0 - x + x1) < x0)
				{
					DrawPixel(Colors, x0 + x - x1, y0 + y, r, g, b);
				}
				x1++;
			}
		}

		DrawPixel(Colors, x0 + x, y0 - y, r, g, b);
		if (true)
		{
			int y1 = 0;
			while (y1 < R)
			{
				if ((y0 - y + y1) < y0 + 1)
				{
					DrawPixel(Colors, x0 + x, y0 - y + y1, r, g, b);
					y1++;
				}
				else
				{
					break;
				}
			}
		}

		DrawPixel(Colors, x0 - x, y0 + y, r, g, b);
		if (true)
		{
			int y1 = 0;
			while (y1 < R)
			{
				if ((y0 - y + y1) < y0 + 1)
				{
					DrawPixel(Colors, x0 - x, y0 + y - y1, r, g, b);
					y1++;
				}
				else
				{
					break;
				}
			}
		}
		DrawPixel(Colors, x0 - x, y0 - y, r, g, b);
		if (true)
		{
			int x1 = 0;
			while (x1 < R + 1)
			{
				if ((x0 - x + x1) < x0 + 1)
				{
					DrawPixel(Colors, x0 - x + x1, y0 - y, r, g, b);
				}
				x1++;
			}
		}
		DrawPixel(Colors, x0, y0, r, g, b);
		error = 2 * (delta + y) * 2 - 1;
		if (delta < 0 && error <= 0) {
			++x;
			delta += 2 * x + 1;
			continue;
		}
		error = 2 * (delta - x) * 2 - 1;
		if (delta > 0 && error > 0) {
			--y;
			delta += 1 - 2 * y;
			continue;
		}
		++x;
		delta += 2 * (x - y);
		--y;
	}
}

void Graphics::DrawLine(int* Colors, int x1, int x2, int y1, int y2, unsigned char r, unsigned char g, unsigned char b)
{
	const int deltaX = abs(x2 - x1);
	const int deltaY = abs(y2 - y1);
	const int signX = x1 < x2 ? 1 : -1;
	const int signY = y1 < y2 ? 1 : -1;
	int error = deltaX - deltaY;
	DrawPixel(Colors, x2, y2, r, g, b);
	while (x1 != x2 || y1 != y2)
	{
		DrawPixel(Colors, x1, y1, r, g, b);
		int error2 = error * 2;
		if (error2 > -deltaY)
		{
			error -= deltaY;
			x1 += signX;
		}
		if (error2 < deltaX)
		{
			error += deltaX;
			y1 += signY;
		}
	}
}

int Graphics::GetWindowWidth()
{
	return WinWidth;
}

int Graphics::GetWindowHeight()
{
	return WinHeight;
}
