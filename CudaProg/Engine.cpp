#include "Engine.h"

Engine::Engine(Window& wnd)
	:
	gfx(wnd.GetWindowWidth(), wnd.GetWindowHeight()),
	dev(wnd.GetWindowHeight(), wnd.GetWindowWidth(), wnd.GetColorBuffer())
{
	Colors = wnd.GetColorBuffer();
	QueryPerformanceFrequency(&PerfCountFrequecyResult);
	PerfCountFrequency = (float)(PerfCountFrequecyResult.QuadPart);
	//SleepIsGranular = (timeBeginPeriod(1) == TIMERR_NOERROR);
	SetWindowTextA(wnd.GetCustomWindow(), "Winframework");
}

Engine::~Engine()
{
}

void Engine::Run(Window& wnd)
{
	//Thread sleep to stop burning cycles
	LARGE_INTEGER LastCounter = EngineGetWallClock();

	LARGE_INTEGER WorkCounter = EngineGetWallClock();

	float WorkSecondsElapsed = EngineGetSecondsElapsed(LastCounter, WorkCounter);
	float SecondsElapsedForFrame = WorkSecondsElapsed;

	while (SecondsElapsedForFrame < FPSMS)
	{
		if (SleepIsGranular)
		{
			DWORD SleepMS = (DWORD)(1000.0f * (FPSMS - SecondsElapsedForFrame));
			Sleep(SleepMS);
		}
		SecondsElapsedForFrame = EngineGetSecondsElapsed(LastCounter, EngineGetWallClock());
	}



	cX = 1.0f / SecondsElapsedForFrame;
	Update(wnd);
	ComposeFrame();

	LARGE_INTEGER EndCounter = EngineGetWallClock();
	LastCounter = EndCounter;
}

void Engine::Update(Window& wnd)
{

	if (wnd.kbd.KeyIsPressed('D'))	cam.pos -= multiple(crossProduct(normalize(cam.getDirection()), cam.up) , 0.02);
	
	if (wnd.kbd.KeyIsPressed('A'))	cam.pos += multiple(crossProduct(normalize(cam.getDirection()), cam.up), 0.02);

	if (wnd.kbd.KeyIsPressed('F'))	cam.pos.y -= 0.02;

	if (wnd.kbd.KeyIsPressed('R'))	cam.pos.y += 0.02;

	if (wnd.kbd.KeyIsPressed('S'))	cam.pos += multiple(normalize(cam.getDirection()), 0.02);

	if (wnd.kbd.KeyIsPressed('W'))	cam.pos -= multiple(normalize(cam.getDirection()), 0.02);

	if (wnd.kbd.KeyIsPressed('Z'))  cam.angleX -= 2;

	if (wnd.kbd.KeyIsPressed('C'))	cam.angleX += 2;
}

LARGE_INTEGER Engine::EngineGetWallClock() const
{
	LARGE_INTEGER Result;
	QueryPerformanceCounter(&Result);
	return Result;
}

float Engine::EngineGetSecondsElapsed(LARGE_INTEGER Start, LARGE_INTEGER End) const
{
	float Result = ((float)(End.QuadPart - Start.QuadPart) / PerfCountFrequency);
	return Result;
}

void Engine::ComposeFrame()
{
	//gfx.ClearScreenSuperFast(Colors);
	//map.renderScene(dev, cam);
	dev.drawMap(map, cam);

	//dev.drawPlane(flor.getPlane(90, cam), 60, 60, 60);
	
	//Vector3D v = cam.getDirection();

	dev.drawPixel(in, 0, 255, 255, 255);
	dev.drawPixel(144, 1, 255, 255, 255);
	dev.drawPixel(540, 360, 255, 255, 255);
	
	in += 1;
	if (in > 255) in = 0;

	//*********************************************************************************
	dev.copyDeviceToHost(*Colors);
	dev.cleanDeviceMem(30, 30, 30);
	//*********************************************************************************
	
	


	//gfx.DrawAlphaRectangle(Colors, 100, 300, 100, 300, 0, 255, 0, in);
	
}
