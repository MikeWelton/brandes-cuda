#include "errors.h"
#include "timer_cuda.h"
#include "graph.cu"


#define BLOCKS 512
#define THREADS 1024


// This kernel works only when run with single cuda block.
__global__ void brandesCudaSingleBlock(Graph *graph, double *centrality, double *delta, int *d,
                                       int *sigma) {
    if (threadIdx.x >= max(graph->numVertices, graph->vmapSize)) {
        return;
    }

    __shared__ int s, l;
    __shared__ bool cont;
    int w, v;
    double sum;

    if (threadIdx.x == 0) {
        s = -1;
    }
    __syncthreads();  // Sync for s

    while (s < graph->numVertices - 1) {
        if (threadIdx.x == 0) {
            ++s;
            l = 0;
            cont = true;
        }
        __syncthreads();  // Sync for s

        // Init arrays in parallel
        for (v = (int) threadIdx.x; v < graph->numVertices; v += (int) blockDim.x) {
            if (v == s) {
                d[s] = 0;
                sigma[s] = 1;
            }
            else {
                d[v] = -1;
                sigma[v] = 0;
            }
        }
        __syncthreads();  // Sync for initial values

        // Forward pass
        while (cont) {
            cont = false;

            // Forward step in parallel
            for (int i = (int) threadIdx.x; i < graph->vmapSize; i += (int) blockDim.x) {
                w = graph->vmap[i];

                if (d[w] == l) {
                    for (int j = (int) graph->vptrs[i]; j < graph->vptrs[i + 1]; ++j) {
                        v = graph->adjs[j];

                        if (d[v] == -1) {
                            d[v] = l + 1;  // d[w] + 1
                            cont = true;
                        }

                        if (d[v] == l + 1) {  // (d[v] == d[w] + 1)
                            atomicAdd(&sigma[v], sigma[w]);
                        }
                    }
                }
            }
            __syncthreads();  // Sync computations

            if (threadIdx.x == 0) {
                ++l;
            }
            __syncthreads();  // Sync for l and arrays
        }

        for (v = (int) threadIdx.x; v < graph->numVertices; v += (int) blockDim.x) {
            delta[v] = (sigma[v] != 0) ? (1 / (double) sigma[v]) : 0.0;
        }

        // Backward pass
        while (l > 1) {
            __syncthreads();  // Sync for l
            if (threadIdx.x == 0) {
                --l;
            }
            __syncthreads();  // Sync for l

            // Backward step in parallel
            for (int i = (int) threadIdx.x; i < graph->vmapSize; i += (int) blockDim.x) {
                w = graph->vmap[i];

                if (d[w] == l) {
                    sum = 0;

                    for (int j = (int) graph->vptrs[i]; j < graph->vptrs[i + 1]; ++j) {
                        v = graph->adjs[j];

                        if (d[v] == l + 1) {  // (d[v] == d[w] + 1)
                            sum += delta[v];
                        }
                    }
                    atomicAdd(&delta[w], sum);

                }

            }
            __syncthreads();
        }

        for (v = (int) threadIdx.x; v < graph->numVertices; v += (int) blockDim.x) {
            if (v != s && delta[v] != 0.0) {
                centrality[v] += delta[v] * (double) sigma[v] - 1;
            }
        }
    }
}


// Code for brandes algorithm run with multiple cuda blocks.
__constant__ int NUM_THREADS = BLOCKS * THREADS;
__device__ bool devCont;

__global__ void initArrays(Graph *graph, int *d, int *sigma, int s) {
    int idx = (int) (threadIdx.x + blockIdx.x * blockDim.x);

    for (int v = idx; v < graph->numVertices; v += NUM_THREADS) {
        if (v == s) {
            d[s] = 0;
            sigma[s] = 1;
        }
        else {
            d[v] = -1;
            sigma[v] = 0;
        }
    }
}

__global__ void forwardStep(Graph *graph, int *d, int *sigma, int l) {
    int idx = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int v, w;

    for (int i = idx; i < graph->vmapSize; i += NUM_THREADS) {
        w = graph->vmap[i];

        if (d[w] == l) {
            for (int j = (int) graph->vptrs[i]; j < graph->vptrs[i + 1]; ++j) {
                v = graph->adjs[j];

                if (d[v] == -1) {
                    d[v] = l + 1;  // d[w] + 1
                    devCont = true;
                }

                if (d[v] == l + 1) {  // (d[v] == d[w] + 1)
                    atomicAdd(&sigma[v], sigma[w]);
                }
            }
        }
    }
}

__global__ void initDelta(Graph *graph, int *sigma, double *delta) {
    int idx = (int) (threadIdx.x + blockIdx.x * blockDim.x);

    for (int v = idx; v < graph->numVertices; v += NUM_THREADS) {
        delta[v] = (sigma[v] != 0) ? (1 / (double) sigma[v]) : 0.0;
    }
}

__global__ void backwardStep(Graph *graph, int *d, double *delta, int l) {
    int idx = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int v, w;
    double sum;

    for (int i = idx; i < graph->vmapSize; i += NUM_THREADS) {
        w = graph->vmap[i];

        if (d[w] == l) {
            sum = 0;

            for (int j = (int) graph->vptrs[i]; j < graph->vptrs[i + 1]; ++j) {
                v = graph->adjs[j];

                if (d[v] == l + 1) {  // (d[v] == d[w] + 1)
                    sum += delta[v];
                }
            }
            atomicAdd(&delta[w], sum);
        }
    }
}

__global__ void updateCentrality(Graph *graph, int *sigma, double *centrality, double *delta, int s) {
    int idx = (int) (threadIdx.x + blockIdx.x * blockDim.x);

    for (int v = idx; v < graph->numVertices; v += NUM_THREADS) {
        if (v != s && delta[v] != 0.0) {
            centrality[v] += delta[v] * (double) sigma[v] - 1;
        }
    }
}


static TimerCuda timer;

// Returns cumulative execution time of all kernels.
pair<float, float> brandesCudaMultipleBlocks(Graph *graph, Graph *deviceGraph, int *d, int *sigma,
                                             double *deviceCentrality, double *delta) {
    int s = 0, l;
    bool cont;
    float kernelsTime = 0.0, memoryTime = 0.0;

    while (s < graph->numVertices) {
        l = 0;
        cont = true;

        // Init arrays in parallel
        timer.start();
        initArrays<<<BLOCKS, THREADS>>>(deviceGraph, d, sigma, s);
        kernelsTime += timer.stop();

        // Forward pass
        while (cont) {
            cont = false;

            // Forward step in parallel
            timer.start();
            cudaCheck(cudaMemcpyToSymbol(devCont, &cont, sizeof(bool)));
            memoryTime += timer.stop();

            timer.start();
            forwardStep<<<BLOCKS, THREADS>>>(deviceGraph, d, sigma, l);
            kernelsTime += timer.stop();

            timer.start();
            cudaCheck(cudaMemcpyFromSymbol((void *) &cont, devCont, sizeof(bool)));
            memoryTime += timer.stop();

            ++l;
        }

        // Init delta values
        timer.start();
        initDelta<<<BLOCKS, THREADS>>>(deviceGraph, sigma, delta);
        kernelsTime += timer.stop();

        // Backward pass
        while (l > 1) {
            --l;

            // Backward step in parallel
            timer.start();
            backwardStep<<<BLOCKS, THREADS>>>(deviceGraph, d, delta, l);
            kernelsTime += timer.stop();
        }

        // Update centrality values
        timer.start();
        updateCentrality<<<BLOCKS, THREADS>>>(deviceGraph, sigma, deviceCentrality, delta, s);
        kernelsTime += timer.stop();

        ++s;
    }

    return {kernelsTime, memoryTime};
}

Graph *copyGraphToCuda(Graph *graph, vector<int*> &devGraphArrays) {
    Graph *deviceGraph;

    // Allocate and copy graph object to cuda
    cudaCheck(cudaMalloc((void **) &deviceGraph, sizeof(Graph)));
    cudaCheck(cudaMemcpy(deviceGraph, graph, sizeof(Graph), cudaMemcpyHostToDevice));

    // Copy arrays that graph stores pointers to
    int *vmap, *vptrs, *adjs;
    cudaCheck(cudaMalloc((void **) &vmap, sizeof(int) * graph->vmapSize));
    cudaCheck(cudaMemcpy(vmap, graph->vmap, sizeof(int) * graph->vmapSize, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(&(deviceGraph->vmap), &vmap, sizeof(int *), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc((void **) &vptrs, sizeof(int) * graph->vptrsSize));
    cudaCheck(cudaMemcpy(vptrs, graph->vptrs, sizeof(int) * graph->vptrsSize, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(&(deviceGraph->vptrs), &vptrs, sizeof(int *), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc((void **) &adjs, sizeof(int) * graph->adjsSize));
    cudaCheck(cudaMemcpy(adjs, graph->adjs, sizeof(int) * graph->adjsSize, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(&(deviceGraph->adjs), &adjs, sizeof(int *), cudaMemcpyHostToDevice));

    devGraphArrays = {vmap, vptrs, adjs};

    return deviceGraph;
}

void freeGraphFromCuda(Graph *deviceGraph, vector<int *> &devGraphArrays) {
    for (auto arrPtr: devGraphArrays) {
        cudaCheck(cudaFree(arrPtr));
    }
    cudaCheck(cudaFree(deviceGraph));
}

double *runBrandesCuda(int numVertices, int numEdges, int **edges) {
    // Pointers to arrays stored in graph that should be freed from cuda (vmap, vptrs, adjs)
    vector<int *> devGraphArrays;
    Graph graph = Graph(numVertices, numEdges, edges);
    Graph *deviceGraph;
    // All arrays that are needed for algorithm
    int *d, *sigma;
    double *centrality = (double *) calloc(graph.numVertices, sizeof(double)),
           *deviceCentrality, *delta;

    float kernelsTime, memoryTime = 0.0;

    timer.start();
    deviceGraph = copyGraphToCuda(&graph, devGraphArrays);

    cudaCheck(cudaMalloc((void **)&sigma, sizeof(int) * graph.numVertices));
    cudaCheck(cudaMalloc((void **)&d, sizeof(int) * graph.numVertices));
    cudaCheck(cudaMalloc((void **)&deviceCentrality, sizeof(double) * graph.numVertices));
    cudaCheck(cudaMalloc((void **)&delta, sizeof(double) * graph.numVertices));
    cudaCheck(cudaMemcpy(deviceCentrality, centrality, sizeof(double) * graph.numVertices, cudaMemcpyHostToDevice));
    memoryTime += timer.stop();

    pair<float, float> algoTimes = brandesCudaMultipleBlocks(&graph, deviceGraph, d, sigma, deviceCentrality, delta);
    kernelsTime = algoTimes.first;
    memoryTime += algoTimes.second;

    // brandesCudaSingleBlock<<<1, THREADS>>>(deviceGraph, deviceCentrality, delta, d, sigma);

    timer.start();
    cudaCheck(cudaMemcpy(centrality, deviceCentrality, sizeof(double) * graph.numVertices, cudaMemcpyDeviceToHost));
    memoryTime += timer.stop();

    // Clean memory
    freeGraphFromCuda(deviceGraph, devGraphArrays);
    cudaCheck(cudaFree(sigma));
    cudaCheck(cudaFree(d));
    cudaCheck(cudaFree(deviceCentrality));
    cudaCheck(cudaFree(delta));

    cerr << (int) kernelsTime << '\n'
         << (int) (kernelsTime + memoryTime) << '\n';

    return centrality;
}
