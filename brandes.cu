#include <iostream>
#include <fstream>
#include <vector>
#include "errors.h"
#include "brandes_cpu.cu"


#define MDEG ((int) 4)  // Use same value as suggested in paper
#define THREADS 1024


using namespace std;


class Graph {
public:
    int *vmap{};
    int *vptrs{};
    int *adjs{};
    int vmapSize = 0;
    int vptrsSize = 0;
    int adjsSize = 0;
    int numVertices;

    Graph(int numVertices, int numEdges, int **edges) {
        this->numVertices = numVertices;
        createVirtualCSRRepresentation(numEdges, edges);
    }

    ~Graph() {
        free(vmap);
        free(vptrs);
        free(adjs);
    }

    void createVirtualCSRRepresentation(int numEdges, int **edges) {
        int *numNeighbors = (int *) calloc(numVertices, sizeof(int));
        for (int i = 0; i < numEdges; ++i) {
            numNeighbors[edges[0][i]]++;
            numNeighbors[edges[1][i]]++;
        }

        // Calculate sizes of three arrays used in representation and allocate them.
        for (int i = 0; i < numVertices; ++i) {
            vmapSize += (numNeighbors[i] + MDEG - 1) / MDEG;
        }
        vptrsSize = vmapSize + 1;
        adjsSize = 2 * numEdges;  // Every edge is counted for both vertices.
        vmap = (int *) calloc(vmapSize, sizeof(int));
        vptrs = (int *) calloc(vptrsSize, sizeof(int));
        adjs = (int *) calloc(adjsSize, sizeof(int));

        // Fill vmap and vptrs.
        // firstVptrs array stores first vptr of every vertex.
        int *firstVptrs = (int *) calloc(numVertices, sizeof(int));
        int vi = 0, neighborsCount = 0;
        for (int i = 0; i < numVertices; ++i) {
            firstVptrs[i] = neighborsCount;
            for (int j = 0; j < (numNeighbors[i] + MDEG - 1) / MDEG; ++j) {
                vmap[vi] = i;
                vptrs[vi] = neighborsCount;
                vi++;
                neighborsCount += min(numNeighbors[i] - j * MDEG, MDEG);
            }
        }
        vptrs[vptrsSize - 1] = adjsSize;

        // Fill adjs.
        // adjsOffset is offset in adjs from firstVptr of given vertex.
        int *adjsOffset = (int *) calloc(numVertices, sizeof(int));
        int vert1, vert2;
        for (int i = 0; i < numEdges; ++i) {
            vert1 = edges[0][i];
            vert2 = edges[1][i];
            adjs[firstVptrs[vert1] + adjsOffset[vert1]] = vert2;
            adjs[firstVptrs[vert2] + adjsOffset[vert2]] = vert1;
            adjsOffset[vert1]++;
            adjsOffset[vert2]++;
        }

        free(edges[0]);
        free(edges[1]);
        free(edges);
        free(numNeighbors);
        free(firstVptrs);
        free(adjsOffset);
    }

    void print() const {
        for (int i = 0; i < vmapSize; ++i) {
            cout << vmap[i] << ' ';
        }
        cout << '\n';

        for (int i = 0; i < vptrsSize; ++i) {
            cout << vptrs[i] << ' ';
        }
        cout << '\n';

        for (int i = 0; i < adjsSize; ++i) {
            cout << adjs[i] << ' ';
        }
        cout << '\n';
    }
};

int **readInputFile(const string &inputFile, int &numEdges, int &numVertices) {
    int i = 0, mem = 1, pos, node1, node2;
    int *nodes1 = (int *) calloc(1, sizeof(int)),
        *nodes2 = (int *) calloc(1, sizeof(int));
    string input;
    ifstream file(inputFile);

    while (getline (file, input)) {
        if (i >= mem) {
            mem = 2 * mem;
            nodes1 = (int *) realloc(nodes1, mem * sizeof(int));
            nodes2 = (int *) realloc(nodes2, mem * sizeof(int));
        }
        pos = (int) input.find_first_of(' ');
        node1 = stoi(input.substr(0, pos));
        node2 = stoi(input.substr(pos + 1, input.length() - pos - 1));
        nodes1[i] = node1;
        nodes2[i] = node2;
        numVertices = max(numVertices, max(node1, node2));
        ++i;
    }

    file.close();

    numEdges = i;
    numVertices += 1;  // Vertices are numbered from 0.
    auto **edges = (int **) calloc(2, sizeof(int *));
    edges[0] = nodes1;
    edges[1] = nodes2;
    return edges;
}

void writeOutputToFile(const string &outputFile, const double *centrality, int numVertices) {
    ofstream file(outputFile);
    for (int i = 0; i < numVertices; ++i) {
        file << centrality[i] << '\n';
    }
}

__global__ void brandesCuda(Graph *graph, double *centrality, double *delta, int *d, int *sigma) {
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
            delta[v] = 0.0;
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

        for (v = 0; v < graph->numVertices; ++v) {
            if (sigma[v] != 0) {
                delta[v] = 1 / (double) sigma[v];
            }
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
            if (s == 4 || s == 5) {
            }
            if (v != s && delta[v] != 0.0) {
                centrality[v] += delta[v] * (double) sigma[v] - 1;
            }
        }
    }
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

void printArray(double *arr, int size) {
    for (int i = 0; i < size; ++i) {
        cout << arr[i] << '\n';
    }
}

double *runBrandesCuda(int numVertices, int numEdges, int **edges) {
    // Pointers to arrays stored in graph that should be freed from cuda.
    vector<int *> devGraphArrays;
    // Create graph and copy it to device
    Graph graph = Graph(numVertices, numEdges, edges);
    Graph *deviceGraph = copyGraphToCuda(&graph, devGraphArrays);

    // Create all necessary arrays, allocate them and copy centrality to device
    int *d, *sigma;
    double *centrality = (double *) calloc(graph.numVertices, sizeof(double)),
           *deviceCentrality, *delta;

    cudaCheck(cudaMalloc((void **)&sigma, sizeof(int) * graph.numVertices));
    cudaCheck(cudaMalloc((void **)&d, sizeof(int) * graph.numVertices));
    cudaCheck(cudaMalloc((void **)&deviceCentrality, sizeof(double) * graph.numVertices));
    cudaCheck(cudaMalloc((void **)&delta, sizeof(double) * graph.numVertices));
    cudaCheck(cudaMemcpy(deviceCentrality, centrality, sizeof(double) * graph.numVertices, cudaMemcpyHostToDevice));

    // Timer
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    cudaCheck(cudaEventRecord(start, nullptr));

    brandesCuda<<<1, THREADS>>>(deviceGraph, deviceCentrality, delta, d, sigma);
    cudaCheck(cudaDeviceSynchronize());

    // Record elapsed time and destroy events
    cudaCheck(cudaEventRecord(stop, nullptr));
    cudaCheck(cudaEventSynchronize(stop));

    float elapsedTime;
    cudaCheck(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Elapsed time: %3.1f ms\n", elapsedTime);

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));

    // Copy centrality and clean memory
    cudaCheck(cudaMemcpy(centrality, deviceCentrality, sizeof(double) * graph.numVertices, cudaMemcpyDeviceToHost));
    freeGraphFromCuda(deviceGraph, devGraphArrays);
    cudaCheck(cudaFree(sigma));
    cudaCheck(cudaFree(d));
    cudaCheck(cudaFree(deviceCentrality));
    cudaCheck(cudaFree(delta));

    cudaDeviceReset();  // TODO For cuda-memcheck

    return centrality;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        cerr << "Incorrect number of arguments.\n"
                "Usage: ./brandes <input-file> <output-file>\n";
        return 1;
    }
    string inputFile = argv[1];
    string outputFile = argv[2];

    int numEdges = 0, numVertices = 0;
    int **edges = readInputFile(inputFile, numEdges, numVertices);

    double *centrality = runBrandesCuda(numVertices, numEdges, edges);

    printArray(centrality, numVertices);
    writeOutputToFile(outputFile, centrality, numVertices);

    free(centrality);
    return 0;
}
