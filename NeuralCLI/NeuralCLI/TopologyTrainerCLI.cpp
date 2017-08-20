#pragma once
#include <memory>
#include <vector>
#include "TopologyTrainerCLI.h"
using namespace std;
namespace NeuralCLI {
	namespace Training {
		namespace Topology {
			NeuralNetwork^ TopologyTrainer::configure_topology(
				SupervisedTrainer<NeuralNetwork^>^ trainer,
				const cli::array<cli::array<double>^>^ trainingInputs,
				const cli::array<cli::array<double>^>^ trainingDesiredOutputs,
				const cli::array<cli::array<double>^>^ testInputs,
				const cli::array<cli::array<double>^>^ testDesiredOutputs
			);
		}
	}
}
