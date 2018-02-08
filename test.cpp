#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <tensorflow/c/c_api.h>
#include <iostream>
#include "tensorflow_cpp_wrapper.hpp"

int main() {
    printf("Hello from TensorFlow C library version %s\n", TF_Version());

    TensorflowGraphWrapper tf("models/graph.pb");
    tf.add_output("output/BiasAdd");
    tf.add_input("inputs", {4, 2});

    float input_vals[] = {1, 0,
                          0, 1,
                          1, 1,
                          0, 0};
    std::vector<TF_Tensor*> outputs = tf.run({"output/BiasAdd"}, {{"inputs", input_vals}});

    for (TF_Tensor* output : outputs) {
        auto tensor_data = TensorflowGraphWrapper::get_tensor_data(output);
        TensorflowGraphWrapper::print_tensor(output);
    }
    TensorflowGraphWrapper::delete_tensors(outputs);
    return 0;
}

