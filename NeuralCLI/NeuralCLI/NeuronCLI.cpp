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
		this->neuron = new neural::Neuron(stdWeights);
		// System::Diagnostics::Debug::WriteLine("Actual Weight count: {0}", neuron->GetWeightSize());
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

	void Neuron::RemoveNeuronWeight(int neuronIndex) {
		this->neuron->remove_weight(neuronIndex);
	}

	void Neuron::SetWeights(array<double>^ weights) {
		std::shared_ptr<std::vector<double>> stdWeights = std::make_shared<std::vector<double>>(weights->Length);
		for (size_t w = 0; w < stdWeights->size(); w++) {
			stdWeights->at(w) = weights[w];
		}
		this->neuron->set_weights(stdWeights);
	}

	double Neuron::GetNeuronWeight(size_t neuronIndex) {
		return this->neuron->neuron_weight(neuronIndex);
	}

	double Neuron::GetThreshold() {
		return this->neuron->threshold();
	}

	void Neuron::SetWeight(size_t weightIndex, double weight) {
		// System::Diagnostics::Debug::WriteLine("Original weight: {0}", this->neuron->GetWeight(weightIndex));
		// System::Diagnostics::Debug::WriteLine("New weight: {0}", weight);
		this->neuron->set_weight(weightIndex, weight);
		// System::Diagnostics::Debug::WriteLine("Actual weight: {0}", this->neuron->GetWeight(weightIndex));
	}

	void Neuron::SetThresholdWeight(double weight) {
		this->neuron->set_threshold(weight);
	}

	void Neuron::SetNeuronWeight(size_t neuronIndex, double weight) {
		this->neuron->set_neuron_weight(neuronIndex, weight);
	}

	double Neuron::GetOutput(array<double>^ inputs) {
		std::vector<double> stdInputs(inputs->Length);
		for (int i = 0; i < stdInputs.size(); i++) {
			stdInputs.at(i) = inputs[i];
		}
		double output = this->neuron->output(stdInputs);
		/*for (size_t w = 0; w < this->neuron->GetWeightSize(); w++) {
			System::Diagnostics::Debug::WriteLine("Actual neuron weight {0}: {1}", w, this->neuron->GetWeight(w));
		}
		System::Diagnostics::Debug::WriteLine("Received output: {0}", output);*/
		return output;
	}	

	size_t Neuron::GetWeightSize() {
		return this->neuron->weight_size();
	}

	array<double>^ Neuron::CloneWeights() {
		size_t weightLength = this->neuron->weight_size();
		array<double>^ cliWeights = gcnew array<double>(weightLength);
		for (size_t w = 0; w < weightLength; w++) {
			cliWeights[w] = this->neuron->weight(w);
			// System::Diagnostics::Debug::WriteLine("Cloned weight {0}: {1}", w, cliWeights[w]);
		}
		return cliWeights;
	}
}