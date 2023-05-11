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

	polygon p1( Vector3D(-1, 2, 1), Vector3D(1, 2, 1),Vector3D(-1, 2, -1));
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
	//int* colors_of_plygons = { new int[10] {13109770, 13109770, 706570, 706570, 658120, 658120, 658120, 658120, 706570, 706570} };
	int* colors_of_plygons = { new int[10] {
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255} }; //16777215
	StaticMesh* mesh = new StaticMesh(polygons, colors_of_plygons, 10);
	int* colors_of_plygons2 = { new int[10] {
		6430250, 6430250, 6430250, 6430250, 6430250, 6430250, 6430250, 6430250, 6430250, 6430250} };
	StaticMesh* mesh2 = new StaticMesh(polygons, colors_of_plygons2, 10);
	int* colors_of_plygons3 = { new int[10] {
		5258121, 5258121, 5258121, 5258121, 5258121, 5258121, 5258121, 5258121, 5258121, 5258121} };
	StaticMesh* mesh3 = new StaticMesh(polygons, colors_of_plygons3, 10);

	polygon flor1(Vector3D(10, 0, 10), Vector3D(-10, 0, -10), Vector3D(-10, 0, 10));
	polygon flor2(Vector3D(-10, 0, -10), Vector3D(10, 0, 10), Vector3D(10, 0, -10));
	polygon* flor_polygons = { new polygon[2] {flor1, flor2} };
	int* flor_colors_of_polygons = { new int[2] {9408399, 9408399} };
	StaticMesh* flor_mesh = new StaticMesh(flor_polygons, flor_colors_of_polygons, 2);

	polygon flor12(Vector3D(10,  20,  -10), Vector3D(-10, 0, -10), Vector3D(-10, 20, -10));
	polygon flor22(Vector3D(-10, 0, -10), Vector3D(10, 20, -10), Vector3D(10, 0, -10));
	polygon* flor_polygons2 = { new polygon[2] {flor12, flor22} };
	StaticMesh* flor_mesh2 = new StaticMesh(flor_polygons2, flor_colors_of_polygons, 2);

	polygon flor13(Vector3D(10, 0, 10), Vector3D(10, 20, -10), Vector3D(10, 20, 10));
	polygon flor23(Vector3D(10, 20, -10), Vector3D(10, 0, 10), Vector3D(10, 0, -10));
	polygon* flor_polygons3 = { new polygon[2] {flor13, flor23} };
	StaticMesh* flor_mesh3 = new StaticMesh(flor_polygons3, flor_colors_of_polygons, 2);

	Object* obj = { new Object[6] { 
		Object(Vector3D(0.8, 0, 100), Vector3D(0, 0, 0), mesh), 
		Object(Vector3D(0, 0, 105), Vector3D(0, 0, 0), mesh2), 
		Object(Vector3D(0.4, 2, 105.4), Vector3D(0, 0, 0), mesh3), 
		Object(Vector3D(0, 0, 100), Vector3D(0, 0, 0), flor_mesh), 
		Object(Vector3D(0, 0, 120), Vector3D(0, 0, 0), flor_mesh2), 
		Object(Vector3D(-20, 0, 100), Vector3D(0, 0, 0), flor_mesh3)}
	};

	Lite* lites = { new Lite[2] {Lite(Vector3D(80, 48, 100), 150), Lite(Vector3D(-5, 50, -100), 150) }};

	map = new Map(obj, 36, 6, lites);
	dev.activateMap(*map, cam);
}

Engine::~Engine()
{
	//delete map;
}

void Engine::Run(Window& wnd)
{
	//Thread sleep to stop burning cycles
	//LARGE_INTEGER LastCounter = EngineGetWallClock();

	//LARGE_INTEGER WorkCounter = EngineGetWallClock();

	//float WorkSecondsElapsed = EngineGetSecondsElapsed(LastCounter, WorkCounter);
	//float SecondsElapsedForFrame = WorkSecondsElapsed;

		
		
	//DWORD SleepMS = (DWORD)(1000.0f * (FPSMS - SecondsElapsedForFrame));
	//Sleep(SleepMS);
	std::thread t(&Engine::ComposeFrame, this);
	//ComposeFrame();

	float dt = ft.Go();
	cX += dt;
	std::chrono::duration<float> SleepTime(FPSMS - cX);

	if (cX < FPSMS) {
		std::this_thread::sleep_for(SleepTime);
	}
	else {
		cX = 0.0f;
	}

	std::string str = std::to_string(1.0f / dt);
	LPCSTR name = str.c_str();
	SetWindowTextA(wnd.GetCustomWindow(), str.c_str());
		
	//SecondsElapsedForFrame = EngineGetSecondsElapsed(LastCounter, EngineGetWallClock());
	
	
	Update(wnd);
	tick += 1;
	if (tick > 144) tick = 0;

	t.join();
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
	//dev.copyDeviceToHost(*Colors);
	//dev.cleanDeviceMem(30, 30, 30);
	gfx.DrawAlphaRectangle(Colors, 10, 100, 10, 100, 255, 90, 30, 125);

	//dev.ray_render(*map, cam, 1);
	//map->object->rotation.x += 1.5;
	//dev.path_tracing(*map, cam, 1);

}
