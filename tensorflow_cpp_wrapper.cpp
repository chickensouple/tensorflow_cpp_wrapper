#include "tensorflow_cpp_wrapper.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <numeric>

static void free_buffer(void* data, size_t length) {
    free(data);
}

static void deallocator(void* data, size_t length, void* arg) {
    free(data);
}


TensorflowGraphWrapper::TensorflowGraphWrapper(const std::string& filename) {
    // load _graph_def
    TF_Buffer* graph_def = load_pb_file(filename);

    // allocating space for private member variables
    _status = TF_NewStatus();
    _graph = TF_NewGraph();


    // Import graph_def into graph
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(_graph, graph_def ,opts, _status);
    TF_DeleteImportGraphDefOptions(opts);
    if (TF_GetCode(_status) != TF_OK) {
        std::cerr << "ERROR: Unable to import graph " << TF_Message(_status) << "\n";
        return;
    }
    std::cout << "Successfully imported graph\n";
    TF_DeleteBuffer(graph_def);

    // creating new session variables
    _sess_opts = TF_NewSessionOptions();
    _session = TF_NewSession(_graph, _sess_opts, _status);
    if (TF_GetCode(_status) != TF_OK) {
        throw std::runtime_error(TF_Message(_status));
    }
}

TensorflowGraphWrapper::~TensorflowGraphWrapper() {
    TF_CloseSession(_session, _status);
    TF_DeleteSession(_session, _status);
    TF_DeleteSessionOptions(_sess_opts);
    TF_DeleteGraph(_graph);
    TF_DeleteStatus(_status);
}

std::vector<std::string> TensorflowGraphWrapper::get_op_names() {
    size_t pos = 0;
    TF_Operation* oper;
    std::vector<std::string> op_names;
    while ((oper = TF_GraphNextOperation(_graph, &pos)) != nullptr) {
        op_names.push_back(TF_OperationName(oper));
    }
    return op_names;
}

void TensorflowGraphWrapper::add_input(const std::string& opname, const std::vector<int64_t>& shape) {
    std::pair<std::string, int> op = parse_opstring(opname);
    std::string name = op.first;
    int idx = op.second;

    TF_Operation* input_op = TF_GraphOperationByName(_graph, name.c_str());
    if (input_op == nullptr) {
        throw std::runtime_error("Operation not found.");
    }
    TF_Output input = {input_op, idx};

    // check shape is valid
    int num_dims = TF_GraphGetTensorNumDims(_graph, input, _status);
    if (TF_GetCode(_status) != TF_OK) {
        throw std::runtime_error(TF_Message(_status));
    }
    if (num_dims != shape.size()) {
        throw std::runtime_error("Number of dimensions is wrong.");
    }

    std::vector<int64_t> tensor_shape(num_dims);
    TF_GraphGetTensorShape(_graph, input, tensor_shape.data(), num_dims, _status);
    if (TF_GetCode(_status) != TF_OK) {
        throw std::runtime_error(TF_Message(_status));
    }
    for (int i = 0; i < num_dims; i++) {
        if (tensor_shape[i] == -1) {
            continue;
        }
        if (tensor_shape[i] != shape[i]) {
            throw std::runtime_error("Shape of input is incorrect.");
        }
    }

    TF_DataType data_type = TF_OperationOutputType(input);
    size_t type_size = TF_DataTypeSize(data_type);

    // computing number of bytes tensor will occupy
    size_t num_bytes = type_size * std::accumulate(std::begin(tensor_shape),
        std::end(tensor_shape), 1, std::multiplies<size_t>());

    InputInfo data{input, data_type, shape, num_bytes};
    _input_map.insert({opname, data});
}

void TensorflowGraphWrapper::add_output(const std::string& opname) {
    std::pair<std::string, int> op = parse_opstring(opname);
    std::string name = op.first;
    int idx = op.second;

    TF_Operation* output_op = TF_GraphOperationByName(_graph, name.c_str());
    if (output_op == nullptr) {
        throw std::runtime_error("Operation not found.");
    }
    TF_Output output = {output_op, idx};

    _output_map.insert({opname, output});
}

std::vector<TF_Tensor*> TensorflowGraphWrapper::run(const std::vector<std::string>& outputs,
        const std::unordered_map<std::string, void*>& inputs) {

    // create list of outputs
    std::vector<TF_Output> output_vec;
    for (const std::string& name : outputs) {
        if (_output_map.find(name) != _output_map.end()) {
            output_vec.push_back(_output_map[name]);
        } else {
            throw std::runtime_error("You have not added this output yet.");
        }
    }
    std::vector<TF_Tensor*> output_tensors(output_vec.size());

    // create list of inputs
    std::vector<TF_Output> input_vec;
    std::vector<TF_Tensor*> input_tensors;
    for (const auto& input : inputs) {
        std::string name = input.first;
        void* data = input.second;

        if (_input_map.find(name) != _input_map.end()) {
            InputInfo& info = _input_map[name];
            input_vec.push_back(info.input);

            TF_Tensor* tensor = TF_NewTensor(info.data_type,
                info.shape.data(),
                info.shape.size(),
                data,
                info.num_bytes,
                &deallocator, 0);

            input_tensors.push_back(tensor);
        } else {
            throw std::runtime_error("You have not added this output yet.");
        }
    }

    TF_SessionRun(_session, nullptr,
        input_vec.data(), input_tensors.data(), input_vec.size(),
        output_vec.data(), output_tensors.data(), output_vec.size(),
        nullptr, 0, nullptr, _status);
    if (TF_GetCode(_status) != TF_OK) {
        throw std::runtime_error(TF_Message(_status));
    }

    return output_tensors;
}


static std::vector<int64_t> clean_dim_vec(const std::vector<int64_t>& dims) {
    std::vector<int64_t> vec;
    for (int i = 0; i < dims.size(); i++) {
        if (dims[i] == 0) {
            continue;
        }
        vec.push_back(dims[i]);
    }
    return vec;
}

std::tuple<void*, TF_DataType, std::vector<int64_t>> TensorflowGraphWrapper::get_tensor_data(TF_Tensor* tensor) {
    void* data = TF_TensorData(tensor);
    TF_DataType data_type = TF_TensorType(tensor);
    int num_dims = TF_NumDims(tensor);
    std::vector<int64_t> dims(num_dims);
    for (int i = 0; i < num_dims; i++) {
        dims.push_back(TF_Dim(tensor, i));
    }
    dims = clean_dim_vec(dims);
    auto tuple = std::make_tuple(data, data_type, dims);
    return tuple;
}

void TensorflowGraphWrapper::delete_tensors(std::vector<TF_Tensor*>& tensors) {
    for (auto tensor : tensors) {
        TF_DeleteTensor(tensor);
    }
}

template <class T>
void print_helper(T* data, int nrows, int ncols) {
    size_t count = 0;
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            std::cout << data[count] << "\t";
            count += 1;
        }
        std::cout << "\n";
    }
}

void TensorflowGraphWrapper::print_tensor(TF_Tensor* tensor) {
    auto tuple = TensorflowGraphWrapper::get_tensor_data(tensor);
    std::vector<int64_t>& dims = std::get<2>(tuple);

    if (dims.size() > 2) {
        throw std::runtime_error("Can only print 1d or 2d tensor.");
    }
    int ncols;
    int nrows;
    if (dims.size() > 1) {
        nrows = dims[0];
        ncols = dims[1];
    } else {
        nrows = 1;
        ncols = dims[0];
    }

    void* data = std::get<0>(tuple);
    TF_DataType type = std::get<1>(tuple);
    if (type == TF_FLOAT) {
        print_helper<float>((float*)data, nrows, ncols);
    } else {
        throw std::runtime_error("DataType not supported.");
    }

}

TF_Buffer* TensorflowGraphWrapper::load_pb_file(const std::string& filename) {
    std::streampos fsize = 0;
    std::ifstream file(filename, std::ios::binary);

    // get data size
    fsize = file.tellg();
    file.seekg(0, std::ios::end);
    fsize = file.tellg() - fsize;

    // reset stream
    file.seekg(0, std::ios::beg);

    char* data = new char[fsize];
    file.read(data, fsize);

    file.close();


    TF_Buffer* graph_def = TF_NewBuffer();
    graph_def->data = data;
    graph_def->length = fsize;
    graph_def->data_deallocator = free_buffer;
    return graph_def;
}

std::pair<std::string, int> TensorflowGraphWrapper::parse_opstring(std::string opstring) {
    int idx = 0;
    std::string name = opstring;
    size_t loc = opstring.find(':');

    // if user specifies a specific index, extract it
    if (loc != std::string::npos) {
        std::string num_str = opstring.substr(loc+1);

        idx = std::stoi(num_str);
        name = opstring.substr(0, loc);
    }

    return {name, idx};
}



