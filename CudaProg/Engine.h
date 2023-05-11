#pragma once
#include "Window.h"
#include "Graphics.h"
#include <string>
#include <thread>
#include "Map.h"
#include "DeviceKernel.cuh"
#include "FPScontroller.h"

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
	FPScontroller ft;
private:
	LARGE_INTEGER PerfCountFrequecyResult;
	bool SleepIsGranular = true;
	float PerfCountFrequency;
	const float FPSMS = 1.0f / 144.0f;
	float tick = 0;
	

	float cX = 0;
	Camera cam;
	Map* map = nullptr;
private:
	//Game stuff here
	float speed = 0.8f;
	float r_speed = 1;
};
