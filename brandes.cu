#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "brandes_cpu.cu"
#include "brandes_cuda.cu"


using namespace std;


void printArray(double *arr, int size) {
    for (int i = 0; i < size; ++i) {
        cout << arr[i] << '\n';
    }
}

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

    // printArray(centrality, numVertices); // TODO remove later
    writeOutputToFile(outputFile, centrality, numVertices);

    free(centrality);
    return 0;
}
