#include "NeuronCLI.h"
#include <memory>
namespace NeuralCLI {
	Neuron::Neuron(array<double>^ weights) {
		// System::Diagnostics::Debug::WriteLine("Creating neuron...");
		std::shared_ptr<std::vector<double>> stdWeights = std::make_shared<std::vector<double>>(weights->Length);
		for (size_t w = 0; w < stdWeights->size(); w++) {
			stdWeights->at(w) = weights[w];
		}
		// System::Diagnostics::Debug::WriteLine("Weight count: {0}", stdWeights->size());
		this->neuron = new Neural::NeuronC(stdWeights);
		// System::Diagnostics::Debug::WriteLine("Actual Weight count: {0}", neuron->GetWeightSize());
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
		std::shared_ptr<std::vector<double>> stdWeights = std::make_shared<std::vector<double>>(weights->Length);
		for (size_t w = 0; w < stdWeights->size(); w++) {
			stdWeights->at(w) = weights[w];
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
		// System::Diagnostics::Debug::WriteLine("Original weight: {0}", this->neuron->GetWeight(weightIndex));
		// System::Diagnostics::Debug::WriteLine("New weight: {0}", weight);
		this->neuron->SetWeight(weightIndex, weight);
		// System::Diagnostics::Debug::WriteLine("Actual weight: {0}", this->neuron->GetWeight(weightIndex));
	}

	void Neuron::SetThresholdWeight(double weight) {
		this->neuron->SetThresholdWeight(weight);
	}

	void Neuron::SetNeuronWeight(size_t neuronIndex, double weight) {
		this->neuron->SetNeuronWeight(neuronIndex, weight);
	}

	double Neuron::GetOutput(array<double>^ inputs) {
		std::vector<double> stdInputs(inputs->Length);
		for (int i = 0; i < stdInputs.size(); i++) {
			stdInputs.at(i) = inputs[i];
		}
		double output = this->neuron->GetOutput(stdInputs);
		/*for (size_t w = 0; w < this->neuron->GetWeightSize(); w++) {
			System::Diagnostics::Debug::WriteLine("Actual neuron weight {0}: {1}", w, this->neuron->GetWeight(w));
		}
		System::Diagnostics::Debug::WriteLine("Received output: {0}", output);*/
		return output;
	}	

	size_t Neuron::GetWeightSize() {
		return this->neuron->GetWeightSize();
	}

	array<double>^ Neuron::CloneWeights() {
		size_t weightLength = this->neuron->GetWeightSize();
		array<double>^ cliWeights = gcnew array<double>(weightLength);
		for (size_t w = 0; w < weightLength; w++) {
			cliWeights[w] = this->neuron->GetWeight(w);
			// System::Diagnostics::Debug::WriteLine("Cloned weight {0}: {1}", w, cliWeights[w]);
		}
		return cliWeights;
	}
}