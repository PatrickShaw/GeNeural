#include "StandardBackpropagationTrainerCLI.h"
#include "Conversion.h"
namespace NeuralCLI {
	namespace Training {
		namespace Backpropagation {
			StandardBackpropagationTrainer::~StandardBackpropagationTrainer() {
				this->!StandardBackpropagationTrainer();
			}

			StandardBackpropagationTrainer::!StandardBackpropagationTrainer() {
				delete this->trainer;
			}

			void StandardBackpropagationTrainer::backpropagation(
				NeuralNetwork^ neuralNetwork,
				cli::array<double>^ inputs,
				cli::array<double>^ desiredOutputs,
				double learningRateFactor
			) {
				this->trainer->backpropagation(
					*neuralNetwork->network,
					*Conversion::array_to_vector(inputs), 
					*Conversion::array_to_vector(desiredOutputs)
				);
			}
			void StandardBackpropagationTrainer::train(
				NeuralNetwork^ trainable,
				cli::array<double>^ trainingInputs,
				cli::array<double>^ trainingOutputs
			) {
				this->trainer->train(
					*trainable->network,
					*Conversion::array_to_vector(trainingInputs),
					*Conversion::array_to_vector(trainingOutputs)
				);
			}
		}
	}
}