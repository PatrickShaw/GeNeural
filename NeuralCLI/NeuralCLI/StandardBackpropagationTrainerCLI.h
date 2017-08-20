#pragma once
#include <memory>
#include <vector>
#include "SupervisedTrainerCLI.h"
#include "NeuralNetworkCLI.h"
#include "neural/neural/training/backpropagation/StandardBackpropagationTrainer.h"
using namespace std;
namespace NeuralCLI {
	namespace Training {
		namespace Backpropagation {
			/**
			 * A supervised trainer that uses standard backpropgation to train a NeuralNetwork
			 */
			public ref class StandardBackpropagationTrainer : SupervisedTrainer<NeuralNetwork> {
			private:
				StandardBackpropagationTrainer::~StandardBackpropagationTrainer();
				StandardBackpropagationTrainer::!StandardBackpropagationTrainer();
			protected:
				neural::training::backpropagation::StandardBackpropagationTrainer* trainer
					= new neural::training::backpropagation::StandardBackpropagationTrainer();
			public:

				/**
				 * Performs backpropagation supervised training on a NeuralNetwork.
				 * Note that this method does exactly the same thing as train but allows you to specify 
				 * the learning rate.
				 * TODO: Return partial differentials and changes in weightings.
				 */
				void backpropagation(
					NeuralNetwork^ neuralNetwork, 
					cli::array<double>^ inputs, 
					cli::array<double>^ desiredOutputs, 
					double learningRateFactor
				);
				/**
				 * Note: It is recommended that the backpropagation method is used instead of this one.
				 */
				virtual void train(
					NeuralNetwork^ trainable, 
					cli::array<double>^ trainingInputs,
					cli::array<double>^ trainingOutputs
				);
			};
		}
	}
}