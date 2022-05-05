#define MDEG 4  // Use same value as suggested in paper


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
