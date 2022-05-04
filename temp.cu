#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <stack>
#include <queue>
#include <cstring>
#include "brandes_cpu.cu"

#define MDEG ((uint32_t) 4)  // Use same value as suggested in paper.
#define THREADS 1024

using namespace std;

static void handleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck( err ) (handleError(err, __FILE__, __LINE__ ))


class Graph {
public:
    uint32_t *vmap{};
    uint32_t *vptrs{};
    uint32_t *adjs{};
    uint32_t vmapSize = 0;
    uint32_t vptrsSize = 0;
    uint32_t adjsSize = 0;
    uint32_t numVertices;

    Graph(uint32_t numVertices, uint32_t numEdges, uint32_t **edges) {
        this->numVertices = numVertices;
        createVirtualCSRRepresentation(numEdges, edges);
    }

    ~Graph() {
        free(vmap);
        free(vptrs);
        free(adjs);
    }

    void createVirtualCSRRepresentation(uint32_t numEdges, uint32_t **edges) {
        uint32_t *numNeighbors = (uint32_t *) calloc(numVertices, sizeof(uint32_t));
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
        vmap = (uint32_t *) calloc(vmapSize, sizeof(uint32_t));
        vptrs = (uint32_t *) calloc(vptrsSize, sizeof(uint32_t));
        adjs = (uint32_t *) calloc(adjsSize, sizeof(uint32_t));

        // Fill vmap and vptrs.
        // firstVptrs array stores first vptr of every vertex.
        uint32_t *firstVptrs = (uint32_t *) calloc(numVertices, sizeof(uint32_t));
        uint32_t vi = 0, neighborsCount = 0;
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
        uint32_t *adjsOffset = (uint32_t *) calloc(numVertices, sizeof(uint32_t));
        uint32_t vert1, vert2;
        for (int i = 0; i < numEdges; ++i) {
            vert1 = edges[0][i];
            vert2 = edges[1][i];
//            cout << "Vertex: " << vert1 << " vptr: " << vptrs[vert1] << " offset: " << adjsOffset[vert1] << " neighbor: " << vert2 << '\n';
//            cout << "Vertex: " << vert2 << " vptr: " << vptrs[vert2] << " offset: " << adjsOffset[vert2] << " neighbor: " << vert1 << '\n';
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

uint32_t **readInputFile(const string &inputFile, uint32_t &numEdges, uint32_t &numVertices) {
    uint32_t i = 0, mem = 1, pos, node1, node2;
    uint32_t *nodes1 = (uint32_t *) calloc(1, sizeof(uint32_t)),
            *nodes2 = (uint32_t *) calloc(1, sizeof(uint32_t));
    string input;
    ifstream file(inputFile);

    while (getline (file, input)) {
        if (i >= mem) {
            mem = 2 * mem;
            nodes1 = (uint32_t *) realloc(nodes1, mem * sizeof(uint32_t));
            nodes2 = (uint32_t *) realloc(nodes2, mem * sizeof(uint32_t));
        }
        pos = input.find_first_of(' ');
        node1 = stoul(input.substr(0, pos));
        node2 = stoul(input.substr(pos + 1, input.length() - pos - 1));
        nodes1[i] = node1;
        nodes2[i] = node2;
        numVertices = max(numVertices, max(node1, node2));
        ++i;
    }

    file.close();

    numEdges = i;
    numVertices += 1;  // Vertices are numbered from 0.
    auto **edges = (uint32_t **) calloc(2, sizeof(uint32_t *));
    edges[0] = nodes1;
    edges[1] = nodes2;
    return edges;
}

void writeOutputToFile(const string &outputFile, const float *centrality, uint32_t numVertices) {
    ofstream file(outputFile);
    for (int i = 0; i < numVertices; ++i) {
        file << centrality[i] << '\n';
    }
}

float *brandesVirtualVerticesSequential(Graph &graph) {
    float *centrality = (float *) calloc(graph.numVertices, sizeof(float)),
            *delta = (float *) calloc(graph.numVertices, sizeof(float));
    int *d = (int *) calloc(graph.numVertices, sizeof(int)),
            *sigma = (int *) calloc(graph.numVertices, sizeof(int));
    float sum;
    int l = 0;
    uint32_t u, v;
    bool cont = true;

    memset(d, -1, graph.numVertices * sizeof(int));
    for (int i = 0; i < graph.numVertices; ++i) {
        cout << delta[i] << ' ' << sigma[i] << ' ' << d[i] << '\n';
    }

    // Forward pass
    while (cont) {
        cont = false;

        // Forward step kernel.
        for (int i = 0; i < graph.vmapSize; ++i) {
            u = graph.vmap[i];

            if (d[u] == l) {
                for (int j = (int) graph.vptrs[i]; j < graph.vptrs[i + 1]; ++j) {
                    v = graph.adjs[j];

                    if (d[v] == -1) {
                        d[v] = l + 1;  // d[u] + 1
                        cont = true;
                    }

                    if (d[v] == l + 1) {  // (d[v] == d[u] + 1)
                        sigma[v] += sigma[u]; // (atomic)
                    }
                }
            }
        }
        // synchronize
        ++l;
    }

    // Backward pass
    while (l > 1) {
        --l;

        // Backward step kernel
        for (int i = 0; i < graph.vmapSize; ++i) {
            u = graph.vmap[i];

            if (d[u] == l) {
                sum = 0;
                for (int j = (int) graph.vptrs[i]; j < graph.vptrs[i + 1]; ++j) {
                    v = graph.adjs[j];

                    if (d[v] == l + 1 && sigma[v] != 0.0) {  // (d[v] == d[u] + 1)
                        delta[u] += ((float) sigma[u] / (float) sigma[v]) * (1 + delta[v]);  // ??? (atomic)
                        sum += delta[v];
                    }
                }
                if (u != s) {
                    centrality[u] += delta[u];
                }
            }
        }
        // synchronize
    }

    free(delta);
    free(sigma);
    free(d);

    return centrality;
}

__global__ void brandesCuda(Graph *graph, float *centrality, float *delta, int *d, int *sigma) {
    __shared__ int s, l;
    __shared__ bool cont;

    if (threadIdx.x == 0) {
        s = -1;
    }

    while (s < graph->numVertices) {
        if (threadIdx.x == 0) {
            ++s;
            l = 0;
            cont = false;
        }

        while (cont) {
            cont = false;

            // Forward step kernel.
            for (int i = 0; i < graph->vmapSize; ++i) {
                u = graph->vmap[i];

                if (d[u] == l) {
                    for (int j = (int) graph.vptrs[i]; j < graph.vptrs[i + 1]; ++j) {
                        v = graph.adjs[j];

                        if (d[v] == -1) {
                            d[v] = l + 1;  // d[u] + 1
                            cont = true;
                        }

                        if (d[v] == l + 1) {  // (d[v] == d[u] + 1)
                            sigma[v] += sigma[u]; // (atomic)
                        }
                    }
                }
            }
            // synchronize
            ++l;
        }

        // Backward pass
        while (l > 1) {
            --l;

            // Backward step kernel
            for (int i = 0; i < graph.vmapSize; ++i) {
                u = graph.vmap[i];

                if (d[u] == l) {
                    sum = 0;
                    for (int j = (int) graph.vptrs[i]; j < graph.vptrs[i + 1]; ++j) {
                        v = graph.adjs[j];

                        if (d[v] == l + 1 && sigma[v] != 0.0) {  // (d[v] == d[u] + 1)
                            delta[u] += ((float) sigma[u] / (float) sigma[v]) * (1 + delta[v]);  // ??? (atomic)
                            sum += delta[v];
                        }
                    }
                    if (u != s) {
                        centrality[u] += delta[u];
                    }
                }
            }
            // synchronize
        }

        ++s;
    }
}

float *runBrandesCuda(uint32_t numVertices, uint32_t numEdges, uint32_t **edges) {
    // Create graph and copy it to device
    Graph graph = Graph(numVertices, numEdges, edges);
    Graph *deviceGraph;

    cudaMalloc((void **)&deviceGraph, sizeof(Graph));
    cudaMemcpy(deviceGraph, &graph, sizeof(Graph), cudaMemcpyHostToDevice);

    // Create all necessary arrays, allocate then and copy
    int *d, *sigma;
    float *centrality = (float *) calloc(graph.numVertices, sizeof(float)),
            *deviceCentrality, *delta;

    cudaMalloc((void **)&sigma, sizeof(int) * graph.numVertices);
    cudaMalloc((void **)&d, sizeof(int) * graph.numVertices);
    cudaMalloc((void **)&deviceCentrality, sizeof(float) * graph.numVertices);
    cudaMalloc((void **)&delta, sizeof(float) * graph.numVertices);
    cudaMemcpy(deviceCentrality, centrality, sizeof(float) * graph.numVertices, cudaMemcpyHostToDevice);

    // Timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, nullptr);

    brandesCuda<<<1, THREADS>>>(deviceGraph, centrality, delta, d, sigma);
    cudaDeviceSynchronize();

    // Record elapsed time and destroy events
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time: %3.1f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy centrality and clean
    cudaMemcpy(centrality, deviceCentrality, sizeof(double) * graph.numVertices, cudaMemcpyDeviceToHost);
    cudaFree(deviceCentrality);
    cudaFree(sigma);
    cudaFree(delta);
    cudaFree(d);

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

    uint32_t numEdges = 0, numVertices = 0;
    uint32_t **edges = readInputFile(inputFile, numEdges, numVertices);

    float *centrality = runBrandesCuda(numVertices, numEdges, edges);

    writeOutputToFile(outputFile, centrality, numVertices);

    free(centrality);
    return 0;
}

// TODO error handling
