#include "stdafx.h";
#include "Neuron.h";
#include "MathHelper.h"
namespace Neural {
	NeuronC::NeuronC(vector<double> weights) {
		this->weights = weights;
	}

	vector<double> NeuronC::GetWeights() {
		return this->weights;
	}

	void NeuronC::AddWeight(double weight = 0) {
		vector<double> newWeights(this->weights.size() + 1);
		for (int i = 0; i < this->weights.size(); i++) {
			newWeights[i] = this->weights[i];
		}
		newWeights[newWeights.size() - 1] = weight;
		this->SetWeights(newWeights);
	}

	void NeuronC::RemoveNeuronWeight(int neuronIndex) {
		int weightIndex = neuronIndex + 1;
		vector<double> newWeights(this->weights.size() - 1);
		for (int w = 0; w < weightIndex; w++) {
			newWeights[w] = this->weights[w];
		}
		for (int w = weightIndex + 1; w < this->weights.size(); w++) {
			newWeights[w - 1] = this->weights[w];
		}
		this->SetWeights(newWeights);
	}

	void NeuronC::SetWeights(vector<double> weights) {
		this->weights = weights;
	}

	double NeuronC::GetNeuronWeight(int neuronIndex) {
		return this->weights[neuronIndex + 1];
	}

	double NeuronC::GetThreshold() {
		return this->weights[0];
	}

	void NeuronC::SetWeight(int weightIndex, double weight) {
		this->weights[weightIndex] = weight;
	}

	void NeuronC::SetThresholdWeight(double weight) {
		this->weights[0] = weight;
	}

	void NeuronC::SetNeuronWeight(int neuronIndex, double weight) {
		this->weights[neuronIndex + 1] = weight;
	}

	double NeuronC::GetOutput(vector<double> inputs) {
		double output = -weights[0];
		for (int i = 0; i < inputs.size(); i++) {
			output += weights[i + 1] * inputs[i];
		}
		return MathHelper::Sigmoid(output);
	}
}