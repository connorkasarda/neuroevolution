// Header file for the Synapse (Edge) class
#ifndef SYNAPSE_H
#define SYNAPSE_H
// Dependencies for Synapse class
#include <neuron.h>
// Blueprint for an edge, or synapse, of the ANN
// Otherwise dubbed the phenotype of the organism
class Synapse
{
    int id;
    float weight;
    Neuron *origin;
    Neuron *target;
    public:
        Synapse(int id, float weight, Neuron *origin, Neuron *target);
        void print();
};
#endif