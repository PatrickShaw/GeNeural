#pragma once
#include "stdafx.h";
#include "Neuron.h"
#include <cliext/vector>
namespace NeuralCLI {
	using namespace System;
	public ref class Neuron {
	private:
		Neural::NeuronC* neuron;
	public:
		Neuron(array<double>^ weights);
		array<double>^ GetWeights();
		void AddWeight(double weight);
		void RemoveNeuronWeight(int neuronIndex);
		void SetWeights(array<double>^ weights);
		double GetNeuronWeight(int neuronIndex);
		double GetThreshold();
		void SetWeight(int weightIndex, double weight);
		void SetThresholdWeight(double weight);
		void SetNeuronWeight(int neuronIndex, double weight);
		double GetOutput(array<double>^ inputs);
	};
}