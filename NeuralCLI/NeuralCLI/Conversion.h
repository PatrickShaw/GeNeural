#pragma once
#include "stdafx.h"
#include <memory>
#include <vector>
namespace NeuralCLI {
	using namespace System;
	class Conversion{
	  public:
		template<typename T>
		static std::shared_ptr<std::vector<T>> array_to_vector(array<T>^ array);
		template<typename T>
		static array<T>^ vector_to_array(const std::vector<T>& vector);
	  };
}
