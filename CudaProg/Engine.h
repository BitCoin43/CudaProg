#pragma once
#include "Window.h"
#include "Graphics.h"
#include "Timer.h"
#include <string>
#include <thread>
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
	int in = 0;

	float cX = 0;
private:
	//Game stuff here
	int playerx = 0;

};
