#include "NeuralNetworkCLI.h"
#include "Conversion.h"
#include <vector>
#include <memory>
namespace NeuralCLI {
  NeuralNetwork::~NeuralNetwork() {
    this->!NeuralNetwork();
  }

  NeuralNetwork::!NeuralNetwork() {
    delete this->network;
  }

  NeuralNetwork::NeuralNetwork(NeuralNetwork ^ network){
	this->network = new neural::NeuralNetwork(*network->network);
  }

  NeuralNetwork::NeuralNetwork(size_t inputCount, array<size_t>^ neuralCounts) {
    this->network = new neural::NeuralNetwork(inputCount, *Conversion::array_to_vector(neuralCounts));
  }

  size_t NeuralNetwork::layer_size() {
    return this->network->layer_size();
  }

  double NeuralNetwork::threshold_to_result_in_zero() {
    return this->network->threshold_to_result_in_zero();
  }

  double NeuralNetwork::inactive_neuron_weight() {
    return this->network->inactive_neuron_weight();
  }

  void NeuralNetwork::randomize_weights(double min, double max) {
    this->network->randomize_weights(min, max);
  }

  array<double>^ NeuralNetwork::raw_outputs(array<double>^ inputs) {
    return Conversion::vector_to_array(*this->network->raw_outputs(*Conversion::array_to_vector(inputs)));
  }

  array<array<double>^>^ NeuralNetwork::all_outputs(array<double>^ inputs) {
	array<array<double>^>^ outputs = gcnew array<array<double>^>(inputs->Length);
    std::shared_ptr<std::vector<std::shared_ptr<std::vector<double>>>> stdOutputs = this->network->all_outputs(*Conversion::array_to_vector(inputs));
	for (size_t l = 0; l < stdOutputs->size(); l++) {
		std::shared_ptr<std::vector<double>> layerOutput = stdOutputs->at(l);
		outputs[l] = Conversion::vector_to_array(*layerOutput);
	}
	return outputs;
  }

  size_t NeuralNetwork::neuron_size(size_t layerIndex) {
    return this->network->neuron_size(layerIndex);
  }

  /**
  * Inserts a layer into the given index with the same neuron count as the previous layer.
  * Each neuron in the layer only acknowledges the input of the neuron from the previous layer with the same index.
  * Warning: This will affect the output values of the network.
  **/
  void NeuralNetwork::insert_after(size_t layerIndex) {
    this->network->insert_after(layerIndex);
  }

  void NeuralNetwork::remove_neuron(size_t layerIndex, size_t neuronIndex) {
    this->network->remove_neuron(layerIndex, neuronIndex);
  }

  void NeuralNetwork::remove_layer(size_t layerIndex) {
    this->network->remove_layer(layerIndex);
  }

  void NeuralNetwork::insert_layer(size_t layerIndex, array<Neuron^>^ layer) {
	std::shared_ptr<std::vector<std::shared_ptr<neural::Neuron>>> stdLayer = std::make_shared<std::vector<std::shared_ptr<neural::Neuron>>>(layer->Length);
	for (size_t i = 0; i < stdLayer->size(); i++) {
		*stdLayer->at(i) = *layer[i]->neuron;
	}
    this->network->insert_layer(
		layerIndex,
		stdLayer
	);
  }

  void NeuralNetwork::add_output_neuron(Neuron^ neuron) {
    this->network->add_output_neuron(std::make_shared<neural::Neuron>(*neuron->neuron));
  }

  void NeuralNetwork::add_non_output_neuron(size_t layerIndex, Neuron^ neuron, array<double>^ outputWeights) {
    this->network->add_non_output_neuron(layerIndex, std::make_shared<neural::Neuron>(*neuron->neuron), *Conversion::array_to_vector(outputWeights));
  }

  /**
  * Splits a neuron into 2 that produce half the output of the original neuron.
  * This effectively adds a neuron without causing the network's behaviour/outputs to change.
  **/
  void NeuralNetwork::split_neuron_non_destructive(size_t layerIndex, size_t neuronIndex) {
    this->network->split_neuron_non_destructive(layerIndex, neuronIndex);
  }

  /**
  * Adds a new neuron that is not affected by any neurons from the previous layer and outputs 0 (i.e. always outputs 0).
  * This effectively adds a neuron without causing the network's behaviour/outputs to change.
  */
  void NeuralNetwork::add_neuron_non_destructive(size_t layerIndex) {
    this->network->add_neuron_non_destructive(layerIndex);
  }

  NeuralNetwork^ NeuralNetwork::produce_new_neural_network() {
    return gcnew NeuralNetwork(this);
  }
}