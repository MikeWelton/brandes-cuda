#include <vector>
#include <stack>
#include <queue>
#include <iostream>


using namespace std;


int **createNeighborsLists(
        int **edges,
        int numEdges,
        int numVertices,
        int *numNeighbors) {
    vector<vector<int>> neighbors(numVertices);

    for (int i = 0; i < numEdges; ++i) {
        // Add neighbor for both nodes
        neighbors[edges[0][i]].push_back(edges[1][i]);
        neighbors[edges[1][i]].push_back(edges[0][i]);
    }

    free(edges[0]);
    free(edges[1]);
    free(edges);

    int **ret = (int **) calloc(numVertices, sizeof(int *));
    for (int i = 0; i < numVertices; ++i) {
        numNeighbors[i] = (int) neighbors[i].size();
        ret[i] = (int *) calloc(numNeighbors[i], sizeof(int));
        memcpy(ret[i], neighbors[i].data(), numNeighbors[i] * sizeof(int));
    }

    return ret;
}

double *brandesCpu(int numVertices, int **neighbors, const int *numNeighbors) {
    double *centrality = (double *) calloc(numVertices, sizeof(double)),
           *delta = (double *) calloc(numVertices, sizeof(double));
    int *d = (int *) calloc(numVertices, sizeof(int)),
        *sigma = (int *) calloc(numVertices, sizeof(int));

    vector<vector<int>> p;
    int w;

    for (int s = 0; s < numVertices; ++s) {
        stack<int> stack;
        queue<int> queue;

        p = vector<vector<int>>(numVertices);
        memset(sigma, 0, numVertices * sizeof(int));
        sigma[s] = 1;
        memset(d, -1, numVertices * sizeof(int));
        d[s] = 0;
        memset(delta, 0.0, numVertices * sizeof(double));

        queue.push(s);

        while (!queue.empty()) {
            int v = queue.front();
            queue.pop();
            stack.push(v);

            for (int i = 0; i < numNeighbors[v]; ++i) {
                w = neighbors[v][i];
                // w found for the first time?
                if (d[w] < 0) {
                    queue.push(w);
                    d[w] = d[v] + 1;
                }

                // shortest path to w via v?
                if (d[w] == d[v] + 1) {
                    sigma[w] += sigma[v];
                    p[w].push_back(v);
                }
            }
        }

        for (int v = 0; v < numVertices; ++v) {
            if (sigma[v] != 0) {
                delta[v] = 1 / (double) sigma[v];
            }
        }

        while (!stack.empty()) {
            w = stack.top();
            stack.pop();

            for (auto v: p[w]) {
                delta[v] += delta[w];
            }
        }

        for (int v = 0; v < numVertices; ++v) {
            if (v != s && delta[v] != 0.0) {
                centrality[v] += delta[v] * (double) sigma[v] - 1;
            }
        }
    }

    free(d);
    free(sigma);
    free(delta);

    return centrality;
}

double *runBrandesCpu(int numVertices, int numEdges, int **edges) {
    int *numNeighbors = (int *) calloc(numVertices, sizeof(int));
    int **neighbors = createNeighborsLists(edges, numEdges, numVertices, numNeighbors);

    double *centrality = brandesCpu(numVertices, neighbors, numNeighbors);

    free(numNeighbors);
    for (int i = 0; i < numVertices; ++i) {
        free(neighbors[i]);
    }
    free(neighbors);

    return centrality;
}
