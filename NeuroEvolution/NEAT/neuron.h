// Header file for the Neuron (Node) class
#ifndef NEURON_H
#define NEURON_H
// Dependencies for Neuron class
#include <axon.h>
#include <vector>
// Enumeration of the 3 possible roles a Neuron can be assigned to
// 1) Sensor --> recieves INPUT from the environment and sends to Thinkers (or directly to Actors)
// 2) Thinker --> HIDDEN between the Sensors and Actors, simply performs activation function on incoming weighted sums
// 3) Actor --> Produces OUTPUT after recieving calculations from Thinkers, deciding what action is taken
enum NeuronType {
    sensor,
    thinker,
    actor
};
// Blueprint for a Neuron, or neuron, of the ANN
// Otherwise dubbed the phenotype of the organism
class Neuron 
{
    int id;
    NeuronType *type;
    std::vector<Axon*> inputs;
    std::vector<Axon*> outputs;
    public:
        Neuron(int id, NeuronType type);
        void print();
};
#endif