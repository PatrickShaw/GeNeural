#pragma once
#include <memory>
#include <vector>
#include "neural/neural/training/SupervisedTrainer.h"
namespace NeuralCLI {
	namespace Training {
		/**
		* A machine learning supervised trainer. Trains an object to output a set of value for a given set of inputs.
		*/
		template<typename T>
		public interface class SupervisedTrainer {
		public:
			/**
			* Trains a given trainable entity to output a certain set of outputs, given a certain set of inputs.
			* @param trainable
			* The object being trained.
			* @param trainingInputs
			* The inputs that the trainable will use to train.
			* @param trainingOutputs
			* The outputs that the trainable needs to reproduce given the training inputs.
			*/
			virtual void train(
				T^ trainable,
				cli::array<double>^ trainingInputs,
				cli::array<double>^ trainingOutputs
			);
		};
	}
}
