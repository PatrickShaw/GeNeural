#pragma once
#include "stdafx.h"
#include "neural/neural/Neuron.h"
namespace NeuralCLI {
	using namespace System;
	public ref class Neuron {
	private:
		neural::Neuron* neuron;
	public:
		Neuron(array<double>^ weights);	
		~Neuron();
		!Neuron();
		double GetWeight(size_t weight);
		void AddWeight(double weight);
		void RemoveNeuronWeight(int neuronIndex);
		void SetWeights(array<double>^ weights);
		double GetNeuronWeight(size_t neuronIndex);
		double GetThreshold();
		void SetWeight(size_t weightIndex, double weight);
		void SetThresholdWeight(double weight);
		void SetNeuronWeight(size_t neuronIndex, double weight);
		double GetOutput(array<double>^ inputs);
		size_t GetWeightSize();
		array<double>^ CloneWeights();
	};
}