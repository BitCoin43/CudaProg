#include "Engine.h"

Engine::Engine(Window& wnd)
	:
	gfx(wnd.GetWindowWidth(), wnd.GetWindowHeight()),
	dev(wnd.GetWindowHeight(), wnd.GetWindowWidth(), wnd.GetColorBuffer()),
	pl(Vector3D(-0.5, 0.5, 0), Vector3D(0.5, 0.5, 0), Vector3D(-0.5, -0.5, 0), Vector3D(0.5, -0.5, 0), 255, 200, 2),
	plan(Vector3D(-0.5, 0.5, 0), Vector3D(-0.5, 0.5, 1), Vector3D(-0.5, -0.5, 0), Vector3D(-0.5, -0.5, 1), 255, 200, 2),
	pl2(Vector3D(-0.5, -0.5, 1), Vector3D(0.5, -0.5, 1), Vector3D(-0.5, -0.5, 0), Vector3D(0.5, -0.5, 0), 255, 200, 2),
	flor(Vector3D(-3, 3, 1), Vector3D(3, 3, 1), Vector3D(-3, -3, 1), Vector3D(3, -3, 1), 255, 200, 2)
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

	if (wnd.kbd.KeyIsPressed('Z'))  cam.angleX += 2;

	if (wnd.kbd.KeyIsPressed('C'))	cam.angleX -= 2;
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

	
	
	dev.drawPlane(pl2.getPlane(in, cam), 10, 10, 200);
	dev.drawPlane(pl.getPlane(in, cam), 200, 10, 10);
	dev.drawPlane(plan.getPlane(in, cam), 10, 200, 10);
	//dev.drawPlane(flor.getPlane(90, cam), 60, 60, 60);
	
	Vector3D v = cam.getDirection();

	dev.drawPixel(in, 0, 255, 255, 255);
	dev.drawPixel(144, 1, 255, 255, 255);
	dev.drawPixel(540, 360, 255, 255, 255);
	
	in += 1;
	if (in > 144) in = 0;

	//*********************************************************************************
	dev.copyDeviceToHost(*Colors);
	dev.cleanDeviceMem(30, 30, 30);
}
