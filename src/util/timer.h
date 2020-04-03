//
// Created by yuhailin.
//

#ifndef VLOCTIO_TIMER_H
#define VLOCTIO_TIMER_H

#include <chrono>

namespace PE {

class Timer {
public:
    Timer();

    void Start();
    void Restart();
    void Pause();
    void Resume();
    void Reset();

    double ElapsedMicroSeconds() const;
    double ElapsedSeconds() const;
    double ElapsedMinutes() const;
    double ElapsedHours() const;
    double ElapsedMilliSeconds() const;

    void PrintSeconds() const;
    void PrintMinutes() const;
    void PrintHours() const;

private:
    bool started_;
    bool paused_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point pause_time_;
};

}

#endif //VLOCTIO_TIMER_H
