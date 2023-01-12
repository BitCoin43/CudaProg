#pragma once
#include <chrono>

class Timer
{
public:
	Timer();
	float Go();
private:
	std::chrono::steady_clock::time_point last;
};