#pragma once
#include "stdafx.h"
#include "neural/neural/Classifier.h"
#include "neural/neural/NeuralNetwork.h"
#include "NeuronCLI.h"
using namespace System;
namespace NeuralCLI {
  public ref class NeuralNetwork {
  protected:
	NeuralNetwork(NeuralNetwork^ network);
  public:
	neural::NeuralNetwork* network;
	NeuralNetwork(neural::NeuralNetwork& network);
	NeuralNetwork(size_t inputCount, cli::array<size_t>^ neuralCounts);
    NeuralNetwork::~NeuralNetwork();
    NeuralNetwork::!NeuralNetwork();
	/**
	* The weight of a neuron.
	* @param layerIndex
	* The index of the layer that the weight resides within.
	* @param neuronIndex
	* The index of the neruon that the weight resides within.
	* @param weightIndex
	* The index of the weight for the given neuron (This includes the threshold weight).
	*/
	double weight(size_t layerIndex, size_t neuronIndex, size_t weightIndex);
	/**
	* Sets the weight for a given neuron on a given layer of the neural network.
	*/
	void set_weight(size_t layerIndex, size_t neuronIndex, size_t weightIndex, double weight);
    size_t layer_size();
	size_t weight_size(size_t layerIndex, size_t neuronIndex);
    double threshold_to_result_in_zero();
    double inactive_neuron_weight();
    void randomize_weights(double min, double max);
	cli::array<double>^ raw_outputs(cli::array<double>^ inputs);
	cli::array<cli::array<double>^>^ all_outputs(cli::array<double>^ inputs);
    size_t neuron_size(size_t layerIndex);
    /**
    * Inserts a layer into the given index with the same neuron count as the previous layer.
    * Each neuron in the layer only acknowledges the input of the neuron from the previous layer with the same index.
    * Warning: This will affect the output values of the network.
    **/
    void insert_after(size_t layerIndex);
    void remove_neuron(size_t layerIndex, size_t neuronIndex);
    void remove_layer(size_t layerIndex);
    void insert_layer(size_t layerIndex, cli::array<Neuron^>^ layer);
    void add_output_neuron(Neuron^ neuron);
    void add_non_output_neuron(size_t layerIndex, Neuron^ neuron, cli::array<double>^ outputWeights);
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
	cli::array<double>^ classify(cli::array<double>^ inputs);
  };
}