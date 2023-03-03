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
	SetWindowTextA(wnd.GetCustomWindow(), "Ray-Tracing");

	polygon p1(Vector3D(-1, 2, -1), Vector3D(-1, 2, 1), Vector3D(1, 2, 1));
	polygon p2(Vector3D(-1, 2, -1), Vector3D(1, 2, -1), Vector3D(1, 2, 1));
	polygon p3(Vector3D(-1, 2, -1), Vector3D(1, 0, -1), Vector3D(-1, 0, -1));
	polygon p4(Vector3D(-1, 2, -1), Vector3D(1, 0, -1), Vector3D(1, 2, -1));
	polygon p5(Vector3D(1, 0, -1), Vector3D(1, 2, 1), Vector3D(1, 2, -1));
	polygon p6(Vector3D(1, 0, -1), Vector3D(1, 2, 1), Vector3D(1, 0, 1));
	
	polygon* polygons = { new polygon[6] {p1, p2, p3, p4, p5, p6}};
	StaticMesh* mesh = new StaticMesh(polygons, 6);
	Object* obj = { new Object[1] {Object(Vector3D(0, 0, 4), Vector3D(0, 0, 0), mesh)} };
	map.object = obj;
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
	tick += 1;
	if (tick > 144) tick = 0;
	ComposeFrame();

	LARGE_INTEGER EndCounter = EngineGetWallClock();
	LastCounter = EndCounter;
}

void Engine::Update(Window& wnd)
{

	if (wnd.kbd.KeyIsPressed('D'))	cam.pos -= multiple(crossProduct(normalize(cam.getDirection()), cam.up) , 0.04);
	
	if (wnd.kbd.KeyIsPressed('A'))	cam.pos += multiple(crossProduct(normalize(cam.getDirection()), cam.up), 0.04);

	if (wnd.kbd.KeyIsPressed('F'))	cam.pos.y -= 0.02;

	if (wnd.kbd.KeyIsPressed('R'))	cam.pos.y += 0.02;

	if (wnd.kbd.KeyIsPressed('S'))	cam.pos += multiple(normalize(cam.getDirection()), 0.04);

	if (wnd.kbd.KeyIsPressed('W'))	cam.pos -= multiple(normalize(cam.getDirection()), 0.04);

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
	//dev.drawMap(map, cam);

	dev.ray_render(map, cam);
	
	//Vector3D v = cam.getDirection();

	dev.drawPixel(tick, 0, 255, 255, 255);
	dev.drawPixel(144, 1, 255, 255, 255);
	dev.drawPixel(540, 360, 255, 255, 255);

	//*********************************************************************************
	dev.copyDeviceToHost(*Colors);
	dev.cleanDeviceMem(30, 30, 30);
	//*********************************************************************************
	
	//gfx.DrawAlphaRectangle(Colors, 100, 300, 100, 300, 0, 255, 0, in);
	
}
