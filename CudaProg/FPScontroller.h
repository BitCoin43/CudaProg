#pragma once
#include <chrono>

class FPScontroller {
public:
	FPScontroller();
	float Go();
private:
	std::chrono::steady_clock::time_point last;
};