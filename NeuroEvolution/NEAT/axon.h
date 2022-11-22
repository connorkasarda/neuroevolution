// Header file for the Axon (Edge) class
#ifndef AXON_H
#define AXON_H
// Dependencies for Axon class
#include <neuron.h>
// Blueprint for an edge, or axon, of the ANN
// Otherwise dubbed the phenotype of the organism
class Axon
{
    int id;
    float weight;
    Neuron *origin;
    Neuron *target;
    public:
        Axon(int id);
        void print();
};
#endif