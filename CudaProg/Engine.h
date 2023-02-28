#pragma once
#include "Window.h"
#include "Graphics.h"
#include "Timer.h"
#include <string>
#include <thread>
#include "Map.h"
#include "DeviceKernel.cuh"

class Engine
{
public:
	Engine(class Window& wnd);
	Engine(const Engine&) = delete;
	Engine operator=(const Engine&) = delete;
	~Engine();

	void Run(class Window& wnd);
private:
	void ComposeFrame();
	void Update(class Window& wnd);
	LARGE_INTEGER EngineGetWallClock() const;
	float EngineGetSecondsElapsed(LARGE_INTEGER Start, LARGE_INTEGER End) const;
private:
	Graphics gfx;
	Device dev;
	int* Colors;
	Timer ft;
private:
	LARGE_INTEGER PerfCountFrequecyResult;
	bool SleepIsGranular = true;
	float PerfCountFrequency;
	const float FPSMS = 1.0f / 144.0f;
	float in = 0;

	float cX = 0;
	Camera cam;
	Map map;
private:
	//Game stuff here
	float playerx = 0;
	float playery = 0;
	float playerz = 0;

	float angleX = 0;
};
