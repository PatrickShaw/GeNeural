#pragma once
#include <memory>
#include <vector>
using namespace System;
namespace NeuralCLI {
  class Conversion{
    generic<typename T>
    static std::shared_ptr<std::vector<T>> array_to_vector(array<T>^ array);
    generic<typename T>
    static array<T>^ vector_to_array(const std::vector<T>& vector);
  };
}
