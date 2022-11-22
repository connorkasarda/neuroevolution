// Header file for the Phenotype (Neural Network) class
#ifndef PHENOTYPE_H
#define PHENOTYPE_H
// Dependencies for Phenotype class
#include <neuron.h>
#include <synapse.h>
#include <vector>
// Blueprint for an ANN
class Phenotype
{
    int id;
    std::vector<Neuron*> sensors;
    std::vector<Neuron*> thinkers;
    std::vector<Neuron*> actors;
    std::vector<Synapse*> synapses;
    public:
        Phenotype(int id, std::vector<Neuron*> sensors, std::vector<Neuron*> thinkers, std::vector<Neuron*> actors, std::vector<Synapse*> synapses);
        void print();
};
#endif