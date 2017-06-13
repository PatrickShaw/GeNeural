#pragma once
#include <vector>
namespace Neural {
	using namespace std;
	public class NeuronC {
	private:
		vector<double> weights;
	public:
		NeuronC(vector<double> weights);
		vector<double> GetWeights();
		void AddWeight(double weight);
		void RemoveNeuronWeight(int neuronIndex);
		void SetWeights(vector<double> weights);
		double GetNeuronWeight(int neuronIndex);
		double GetThreshold();
		void SetWeight(int weightIndex, double weight);
		void SetThresholdWeight(double weight);
		void SetNeuronWeight(int neuronIndex, double weight);
		double GetOutput(vector<double> inputs);
	};
}