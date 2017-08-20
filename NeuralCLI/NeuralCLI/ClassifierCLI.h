#pragma once
#include <vector>
#include <memory>
namespace NeuralCLI {
	public interface class Classifier {
		/**
		* Outputs a set of values for a given set of inputs.
		*/
		cli::array<double>^ classify(array<double>^ inputs);
	};
}
