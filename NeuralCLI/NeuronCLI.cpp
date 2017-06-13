#include "stdafx.h"
#include "NeuronCLI.h";
namespace NeuralCLI {
	Neuron::Neuron(array<double>^ weights) {
		std::vector<double> nativeWeights(weights->Length);
		for each (double weight in weights) {
			nativeWeights.push_back(weight);
		}
		this->neuron = new Neural::NeuronC(nativeWeights);
	}

	array<double>^ Neuron::GetWeights() {
		std::vector<double> stdWeights = this->neuron->GetWeights();
		array<double>^ cliWeights = gcnew array<double>(stdWeights.size());
		for each (double weight in stdWeights) {
			stdWeights.push_back(weight);
		}
		return cliWeights;
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

	double Neuron::GetNeuronWeight(int neuronIndex) {
		return this->neuron->GetNeuronWeight(neuronIndex);
	}

	double Neuron::GetThreshold() {
		return this->neuron->GetThreshold();
	}

	void Neuron::SetWeight(int weightIndex, double weight) {
		this->neuron->SetWeight(weightIndex, weight);
	}

	void Neuron::SetThresholdWeight(double weight) {
		this->neuron->SetThresholdWeight(weight);
	}

	void Neuron::SetNeuronWeight(int neuronIndex, double weight) {
		this->neuron->SetNeuronWeight(neuronIndex, weight);
	}

	double Neuron::GetOutput(array<double>^ inputs) {
		std::vector<double> stdInputs(inputs->Length);
		for each (double input in inputs) {
			stdInputs.push_back(input);
		}
		return this->neuron->GetOutput(stdInputs);
	}	
}