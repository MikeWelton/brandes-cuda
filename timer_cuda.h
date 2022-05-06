#ifndef __TIMER_CUDA_H__
#define __TIMER_CUDA_H__

class TimerCuda {
private:
    float timeMilis{};
    float timeSeconds{};
    float timeMinutes{};
    float timeMinutesRemSeconds{};

public:
    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};

    TimerCuda() {
        cudaCheck(cudaEventCreate(&startEvent));
        cudaCheck(cudaEventCreate(&stopEvent));
    }

    ~TimerCuda() {
        cudaCheck(cudaEventDestroy(startEvent));
        cudaCheck(cudaEventDestroy(stopEvent));
    }

    void start() {
        cudaCheck(cudaEventRecord(startEvent, nullptr));
    }

    float stop(bool print=false) {
        cudaCheck(cudaEventRecord(stopEvent, nullptr));
        cudaCheck(cudaEventSynchronize(stopEvent));
        cudaCheck(cudaEventElapsedTime(&timeMilis, startEvent, stopEvent));

        if (print) {
            timeSeconds = timeMilis / 1000;
            timeMinutesRemSeconds = (float) fmod(timeSeconds, 60);
            timeMinutes = (timeSeconds - timeMinutesRemSeconds) / 60;
            printf("Elapsed time: %3.1f ms | %.1f s | %.0f min %.1f s\n",
                   timeMilis, timeSeconds, timeMinutes, timeMinutesRemSeconds);
        }

        return timeMilis;
    }
};

#endif // __TIMER_CUDA_H__
