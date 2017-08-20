#pragma once
#include "stdafx.h"
#include "neural/neural/Neuron.h"
using namespace System;
namespace NeuralCLI {
	public ref class Neuron {
	public:
		neural::Neuron* neuron;
		Neuron(cli::array<double>^ weights);
		~Neuron();
		!Neuron();
		double GetWeight(size_t weight);
		void AddWeight(double weight);
		void RemoveNeuronWeight(size_t neuronIndex);
		void SetWeights(cli::array<double>^ weights);
		double GetNeuronWeight(size_t neuronIndex);
		double GetThreshold();
		void SetWeight(size_t weightIndex, double weight);
		void SetThresholdWeight(double weight);
		void SetNeuronWeight(size_t neuronIndex, double weight);
		double GetOutput(cli::array<double>^ inputs);
		size_t GetWeightSize();
		cli::array<double>^ CloneWeights();
	};
}