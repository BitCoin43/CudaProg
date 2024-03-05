#include "Engine.h"

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd)
{
	Window wnd(1080, 720, hInst);

	Engine engine(wnd);

	while (wnd.RunMessageLoop()) engine.Run(wnd);
	
	return 0;
}