#include "FPScontroller.h"

FPScontroller::FPScontroller()
{
    last = std::chrono::steady_clock::now();
}

float FPScontroller::Go()
{
    std::chrono::steady_clock::time_point old = last;
    last = std::chrono::steady_clock::now();
    std::chrono::duration<float> FrameTime = last - old;

    return FrameTime.count();
}
