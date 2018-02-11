#ifndef TENSORFLOW_GRAPH_WRAPPER_HPP_
#define TENSORFLOW_GRAPH_WRAPPER_HPP_

#include <tensorflow/c/c_api.h>
#include <cstdint>

#include <string>
#include <vector>
#include <unordered_map>


class TensorflowGraphWrapper {
public:
    /**
     * @brief Loads a frozen protobuf graph definition
     * @details protobuf file should have all variables
     * as constants. This constructor will not restore any
     * checkpoints. The protobuf file must also be a binary file and not text
     * 
     * @param filename name of protobuf file
     */
    TensorflowGraphWrapper(const std::string& filename);
    ~TensorflowGraphWrapper();

    /**
     * @brief get all the op names in the graph
     */
    std::vector<std::string> get_op_names();

    /**
     * @brief Sets up data about a possible input to the network
     * @details opname can be entered with or without the tensor index
     * Example: "layer1/weights" or "layer1/weights:0"
     * The opname you use here will be the same you use later in run()
     * 
     * This wrapper can not currently handle variable size inputs.
     * 
     * @param opname
     * @param shape shape of the input
     */
    void add_input(const std::string& opname, const std::vector<int64_t>& shape);
    /**
     * @brief Sets up data about a possible output from the network
     * @details opname can be entered with or without the tensor index
     * Example: "layer1/weights" or "layer1/weights:0"
     * The opname you use here will be the same you use later in run()
     * 
     * @param opname
     */
    void add_output(const std::string& opname);

    /**
     * @brief Equivalent of sess.run()
     * @details outputs is the vector of fetched variables
     * Together (inputs, input_data) is like a feed_dict
     * NOTE: You will need to delete the output tensors data
     * using delete_tensors() or else you will have memory leaks
     * 
     * @param input_data vector of outputs you want to compute from network
     * @param inputs dict of inputs you want to feed
     * @returns vector of output tensors.
     */
    std::vector<TF_Tensor*> run(const std::vector<std::string>& outputs,
        const std::unordered_map<std::string, void*>& inputs);

    /**
     * @brief Extracts data from a tensor
     * @details will return a tuple of <pointer to data, data_type, vector of length of each dim>
     * the pointer will be need to be cast from void*
     * the data will in an order such that the last dimension will be contiguous
     * and then will work its way to the first dimension
     * For example, for an n by m matrix, the first n elements of the data
     * will be the first row, the next n elements will be the second row, and so on
     * Note: this data will be invalidated after running delete_tensors()
     *
     * @param tensor tensor to extract data from
     * @return tuple
     */
    static std::tuple<void*, TF_DataType, std::vector<int64_t>> get_tensor_data(TF_Tensor* tensor);

    /**
     * @brief Clean up memory of a vector of tensors
     * @details this will invalidate any data returned from get_tensor_data()
     *
     * @param tensors vector of tensors to delete
     */
    static void delete_tensors(std::vector<TF_Tensor*>& tensors);

    /**
     * @brief Prints a 1d or 2d vector
     * @details currently only prints floats
     * 
     * @param tensor tensor to print
     */
    static void print_tensor(TF_Tensor* tensor);

private:
    struct InputInfo {
        TF_Output input;
        TF_DataType data_type;
        std::vector<int64_t> shape;
        size_t num_bytes;
    };

    static TF_Buffer* load_pb_file(const std::string& filename);

    std::pair<std::string, int> parse_opstring(std::string opstring);

    TF_Graph* _graph;
    TF_Status* _status; // used to hold all statuses about TF operations
    TF_SessionOptions* _sess_opts;
    TF_Session * _session;


    std::unordered_map<std::string, TF_Output> _output_map;
    std::unordered_map<std::string, InputInfo> _input_map;
};


#endif /* #define TENSORFLOW_GRAPH_WRAPPER_HPP_ */
