#include "NeuronCLI.h"
#include <memory>
namespace NeuralCLI {
	Neuron::Neuron(array<double>^ weights) {
		System::Diagnostics::Debug::WriteLine("Creating neuron...");
		std::vector<double> nativeWeights(weights->Length);
		for each (double weight in weights) {
			nativeWeights.push_back(weight);
		}
		this->neuron = new Neural::NeuronC(nativeWeights);
	}

	Neuron::~Neuron() { 
		this->!Neuron(); 
	}

	Neuron::!Neuron() { 
		delete neuron; 
	}

	double Neuron::GetWeight(size_t weightIndex) {
		return this->neuron->GetWeight(weightIndex);
	}

	void Neuron::AddWeight(double weight) {
		this->neuron->AddWeight(weight);
	}

	void Neuron::RemoveNeuronWeight(int neuronIndex) {
		this->neuron->RemoveNeuronWeight(neuronIndex);
	}

	void Neuron::SetWeights(array<double>^ weights) {
		std::vector<double> stdWeights(weights->Length);
		for each (double weight in weights) {
			stdWeights.push_back(weight);
		}
		this->neuron->SetWeights(stdWeights);
	}

	double Neuron::GetNeuronWeight(size_t neuronIndex) {
		return this->neuron->GetNeuronWeight(neuronIndex);
	}

	double Neuron::GetThreshold() {
		return this->neuron->GetThreshold();
	}

	void Neuron::SetWeight(size_t weightIndex, double weight) {
		this->neuron->SetWeight(weightIndex, weight);
	}

	void Neuron::SetThresholdWeight(double weight) {
		this->neuron->SetThresholdWeight(weight);
	}

	void Neuron::SetNeuronWeight(size_t neuronIndex, double weight) {
		this->neuron->SetNeuronWeight(neuronIndex, weight);
	}

	double Neuron::GetOutput(array<double>^ inputs) {
		std::vector<double> stdInputs(inputs->Length);
		for each (double input in inputs) {
			stdInputs.push_back(input);
		}
		return this->neuron->GetOutput(stdInputs);
	}	

	size_t Neuron::GetWeightSize() {
		return this->neuron->GetWeightSize();
	}

	array<double>^ Neuron::CloneWeights() {
		size_t weightLength = this->neuron->GetWeightSize();
		array<double>^ cliWeights = gcnew array<double>(weightLength);
		for (int w = 0; w < weightLength; w++) {
			cliWeights[w] = neuron->GetWeight(w);
		}
		return cliWeights;
	}
}