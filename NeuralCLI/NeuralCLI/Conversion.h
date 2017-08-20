#pragma once
#include "stdafx.h"
#include <memory>
#include <vector>
using namespace System;
namespace NeuralCLI {
	public ref class Conversion abstract sealed {
		public:
		template<typename T>
		static std::shared_ptr<std::vector<T>> array_to_vector(cli::array<T>^ array) {
			std::shared_ptr<std::vector<T>> stdVector = std::make_shared<std::vector<T>>(array->Length);
			{
				pin_ptr<T> pin(&array[0]);
				T *first(pin), *last(pin + array->Length);
				std::copy(first, last, stdVector->begin());
			}
			return stdVector;
		}
		template<typename T>
		static cli::array<T>^ vector_to_array(const std::vector<T>& vector) {
			cli::array<T>^ cliArray = gcnew cli::array<T>(vector.size());
			for (size_t i = 0; i < vector.size(); i++) {
				cliArray[i] = vector.at(i);
			}
			return cliArray;
		}
	};
}
