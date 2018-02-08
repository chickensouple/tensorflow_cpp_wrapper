A C++ wrapper for the Tensorflow C API

This will allow you to take a network trained in python and run it in a c++ program. This wraps the Tensorflow C API in a class (which is supposedly more stable?).

To use this, there are two steps: 
1) Build a shared tensorflow library 
2) Build static wrapper library that links to shared tensorflow library

To build tensorflow shared library: 
1) clone tensorflow: https://github.com/tensorflow/tensorflow 
2) checkout the version you want 
3) cd into tensorflow/ root directory 
4) run "./configure" 
5) run "bazel build tensorflow:libtensorflow.so"

To build static wrapper library: 
1) cd into tensorflow_cpp_wrapper/ root folder 
2) set the "tensorflow_path" variable in the CMakeLists.txt to point to the root directory of tensorflow 
2) mkdir build 
3) cd build 
4) cmake.. 
5) make

================================================

Now you can link against the static wrapper library to create c++ code that can load saved binary protobuf tensorflow models. (you may also have to link your c++ code with the shared tensorflow library)

This wrapper only works with tensorflow models that have constant variables (it won't restore a checkpoint). Use tf.graph_util.convert_variables_to_constants() and tf.train.write_graph() to save tensorflow models in python.

See test.cpp for example usage.

You can test if the wrapper is working by going to the main tensorflow_cpp_wrapper/ folder and 
1) run xor_network.py to train a simple MLP that learns the xor function
2) run "build/test_exe"

================================================

some of this code is based off of: https://stackoverflow.com/questions/44305647/segmentation-fault-when-using-tf-sessionrun-to-run-tensorflow-graph-in-c-not-c


