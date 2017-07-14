#include "NeuralNetworkCLI.h"
namespace NeuralCLI {
  NeuralNetwork::~NeuralNetwork() {
    this->!NeuralNetwork();
  }

  NeuralNetwork::!NeuralNetwork() {
    delete this->network;
  }
  NeuralNetwork::NeuralNetwork(size_t inputCount, array<int>^ neuralCounts) {
    this->network = new neural::NeuralNetwork(inputCount, );
  }
  size_t NeuralNetwork::layer_size() {
    return this->network->layer_size();
  }
  array<Neuron^>^ NeuralNetwork::layer(size_t layerIndex) {
    std::vector<std::shared_ptr<neural::Neuron>>& layer = this->network->layer(layerIndex);
    array<Neuron^>^ layer = gcnew array<Neuron^>^()
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
  array<double>^ NeuralNetwork::create_inactive_neuron_weights(size_t weightCount) {
    return this->network->create_inactive_neuron_weights(weightCount);
  }
  array<double>^ NeuralNetwork::raw_outputs(array<double>^ inputs) {
    vector
    return this->network->raw_outputs(inputs);
  }
  array<array<double>^>^ NeuralNetwork::all_outputs(array<double>^ inputs) {
    return this->network->all_outputs(inputs);
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
    this->network->insert_layer(layerIndex, layer);
  }
  void NeuralNetwork::add_output_neuron(Neuron^ neuron) {
    this->network->add_output_neuron(neuron);
  }
  void NeuralNetwork::add_non_output_neuron(size_t layerIndex, Neuron^ neuron, array<double>^ outputWeights) {
    this->network->add_non_output_neuron(layerIndex, neuron, outputWeights);
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