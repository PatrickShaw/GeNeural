#include "NeuronCLI.h"
#include "Conversion.h"
#include <vector>
#include <memory>
namespace NeuralCLI {
	Neuron::Neuron(cli::array<double>^ weights) {
		this->neuron = new neural::Neuron(Conversion::array_to_vector(weights));
	}

	Neuron::~Neuron() { 
		this->!Neuron(); 
	}

	Neuron::!Neuron() { 
		delete neuron; 
	}

	double Neuron::GetWeight(size_t weightIndex) {
		return this->neuron->weight(weightIndex);
	}

	void Neuron::AddWeight(double weight) {
		this->neuron->push_weight(weight);
	}

	void Neuron::RemoveNeuronWeight(size_t neuronIndex) {
		this->neuron->remove_neuron_weight(neuronIndex);
	}

	void Neuron::SetWeights(cli::array<double>^ weights) {
		this->neuron->set_weights(Conversion::array_to_vector(weights));
	}

	double Neuron::GetNeuronWeight(size_t neuronIndex) {
		return this->neuron->neuron_weight(neuronIndex);
	}

	double Neuron::GetThreshold() {
		return this->neuron->threshold();
	}

	void Neuron::SetWeight(size_t weightIndex, double weight) {
		this->neuron->set_weight(weightIndex, weight);
	}

	void Neuron::SetThresholdWeight(double weight) {
		this->neuron->set_threshold(weight);
	}

	void Neuron::SetNeuronWeight(size_t neuronIndex, double weight) {
		this->neuron->set_neuron_weight(neuronIndex, weight);
	}

	double Neuron::GetOutput(cli::array<double>^ inputs) {
		return this->neuron->output(*Conversion::array_to_vector(inputs));
	}	

	size_t Neuron::GetWeightSize() {
		return this->neuron->weight_size();
	}

	cli::array<double>^ Neuron::CloneWeights() {
		size_t weightLength = this->neuron->weight_size();
		cli::array<double>^ cliWeights = gcnew cli::array<double>(weightLength);
		for (size_t w = 0; w < weightLength; w++) {
			cliWeights[w] = this->neuron->weight(w);
		}
		return cliWeights;
	}
}