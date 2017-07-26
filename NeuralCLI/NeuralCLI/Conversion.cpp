#include "Conversion.h"
#include <vector>
#include <memory>
namespace NeuralCLI {
	template <typename T> std::shared_ptr<std::vector<T>> Conversion::array_to_vector(array<T>^ cliArray) {
		std::shared_ptr<std::vector<T>> stdVector = std::make_shared<std::vector<T>>(cliArray->Length);
		{
			pin_ptr<T> pin(&cliArray[0]);
			T *first(pin),	 *last(pin + cliArray->Length);
			std::copy(first, last, stdVector->begin());
		}
		return stdVector;
	}

	template <typename T> array<T>^ Conversion::vector_to_array(const std::vector<T>& vector) {
		array<T>^ cliArray = gcnew array<T> ^ (vector.size());
		for (int i = 0; i < vector.size(); i++) {
			cliArray[i] = vector.at(i);
		}
		return cliArray;
	}
}