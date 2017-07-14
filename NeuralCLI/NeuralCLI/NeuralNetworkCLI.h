#pragma once
#include "neural/neural/Classifier.h"
#include "neural/neural/NeuralNetwork.h";
#include "NeuronCLI.h"
namespace NeuralCLI {
  using namespace System;
  public ref class NeuralNetwork {
  private:
    neural::NeuralNetwork* network;
  protected:
    NeuralNetwork(NeuralNetwork^ network);
  public:
    NeuralNetwork(size_t inputCount, array<int>^ neuralCounts);
    NeuralNetwork::~NeuralNetwork();
    NeuralNetwork::!NeuralNetwork();
    size_t layer_size();
    array<Neuron^>^ layer(size_t layerIndex);
    double threshold_to_result_in_zero();
    double inactive_neuron_weight();
    void randomize_weights(double min, double max);
    array<double>^ create_inactive_neuron_weights(size_t weightCount);
    array<double>^ raw_outputs(array<double>^ inputs);
    array<array<double>^>^ all_outputs(array<double>^ inputs);
    size_t neuron_size(size_t layerIndex);
    /**
    * Inserts a layer into the given index with the same neuron count as the previous layer.
    * Each neuron in the layer only acknowledges the input of the neuron from the previous layer with the same index.
    * Warning: This will affect the output values of the network.
    **/
    void insert_after(size_t layerIndex);
    void remove_neuron(size_t layerIndex, size_t neuronIndex);
    void remove_layer(size_t layerIndex);
    void insert_layer(size_t layerIndex, array<Neuron^>^ layer);
    void add_output_neuron(Neuron^ neuron);
    void add_non_output_neuron(size_t layerIndex, Neuron^ neuron, array<double>^ outputWeights);
    /**
    * Splits a neuron into 2 that produce half the output of the original neuron.
    * This effectively adds a neuron without causing the network's behaviour/outputs to change.
    **/
    void split_neuron_non_destructive(size_t layerIndex, size_t neuronIndex);
    /**
    * Adds a new neuron that is not affected by any neurons from the previous layer and outputs 0 (i.e. always outputs 0).
    * This effectively adds a neuron without causing the network's behaviour/outputs to change.
    */
    void add_neuron_non_destructive(size_t layerIndex);
    NeuralNetwork^ produce_new_neural_network();
  };
}