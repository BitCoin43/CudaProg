#include "Engine.h"

void o(Device dev, Map map, Camera cam) {
	dev.ray_render(map, cam);
}

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
	polygon p2(Vector3D(-1, 2, -1), Vector3D(1, 2, 1), Vector3D(1, 2, -1));
	polygon p3(Vector3D(-1, 2, -1), Vector3D(1, 0, -1), Vector3D(-1, 0, -1));
	polygon p4(Vector3D(1, 2, -1), Vector3D(1, 0, -1), Vector3D(-1, 2, -1));
	polygon p5(Vector3D(1, 2, 1), Vector3D(1, 0, -1), Vector3D(1, 2, -1));
	polygon p6(Vector3D(1, 0, -1), Vector3D(1, 2, 1), Vector3D(1, 0, 1));
	polygon p7(Vector3D(-1, 0, -1),Vector3D(-1, 2, 1),  Vector3D(-1, 2, -1));
	polygon p8( Vector3D(-1, 2, 1),Vector3D(-1, 0, -1), Vector3D(-1, 0, 1));
	polygon p9( Vector3D(-1, 2, 1),Vector3D(-1, 0, 1),   Vector3D(1, 0, 1) );
	polygon p10(Vector3D(1, 2, 1) ,Vector3D(-1, 2, 1),   Vector3D(1, 0, 1) );
	polygon* polygons = { new polygon[10] {p1, p2, p3, p4, p5, p6, p7, p8, p9, p10} };
	int* colors_of_plygons = { new int[10] {13109770, 13109770, 706570, 706570, 658120, 658120, 658120, 658120, 706570, 706570} };
	StaticMesh* mesh = new StaticMesh(polygons, colors_of_plygons, 10);

	polygon flor1(Vector3D(5, 0, 5), Vector3D(-5, 0, -5), Vector3D(-5, 0, 5));
	polygon flor2(Vector3D(-5, 0, -5), Vector3D(5, 0, 5), Vector3D(5, 0, -5));
	polygon* flor_polygons = { new polygon[2] {flor1, flor2} };
	int* flor_colors_of_polygons = { new int[2] {9408399, 9408399} };
	StaticMesh* flor_mesh = new StaticMesh(flor_polygons, flor_colors_of_polygons, 2);

	Object* obj = { new Object[2] {Object(Vector3D(0, 0, 100), Vector3D(0, 0, 0), mesh), Object(Vector3D(0, 0, 100), Vector3D(0, 0, 0), flor_mesh)}};

	map = new Map(obj, 12);
}

Engine::~Engine()
{
	//delete map;
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
			std::string str = std::to_string(1000 / (SleepMS + SecondsElapsedForFrame));
			LPCSTR name = str.c_str();
			SetWindowTextA(wnd.GetCustomWindow(), name);
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
	if (wnd.kbd.KeyIsPressed('D'))	cam.pos += multiple(crossProduct(normalize(cam.getDirection()), cam.up) , speed);
	
	if (wnd.kbd.KeyIsPressed('A'))	cam.pos -= multiple(crossProduct(normalize(cam.getDirection()), cam.up), speed);

	if (wnd.kbd.KeyIsPressed('F'))	cam.pos.y += speed;

	if (wnd.kbd.KeyIsPressed('R'))	cam.pos.y -= speed;

	if (wnd.kbd.KeyIsPressed('S'))	cam.pos += multiple(normalize(cam.getDirection()), speed);

	if (wnd.kbd.KeyIsPressed('W'))	cam.pos -= multiple(normalize(cam.getDirection()), speed);

	if (wnd.kbd.KeyIsPressed('Z'))  cam.angleX += r_speed;
												 
	if (wnd.kbd.KeyIsPressed('C'))	cam.angleX -= r_speed;
												 
	if (wnd.kbd.KeyIsPressed('G'))	cam.angleY += r_speed;
												 
	if (wnd.kbd.KeyIsPressed('T'))	cam.angleY -= r_speed;

	if (wnd.kbd.KeyIsPressed('N')) {
		cam.angleX += 0.6;
		cam.pos += multiple(crossProduct(normalize(cam.getDirection()), cam.up), speed);
	}
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
	dev.ray_render(*map, cam);
	
	//Vector3D v = cam.getDirection();

	dev.drawPixel(tick, 0, 255, 255, 255);
	dev.drawPixel(144, 1, 255, 255, 255);
	//dev.drawPixel(540, 360, 255, 255, 255);

	//*********************************************************************************
	dev.copyDeviceToHost(*Colors);
	dev.cleanDeviceMem(30, 30, 30);
	////*********************************************************************************
	
	//gfx.DrawAlphaRectangle(Colors, 100, 300, 100, 300, 0, 255, 0, in);
	
}
