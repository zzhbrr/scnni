#include "scnni/graph.hpp"

auto main() -> int {
  std::unique_ptr<scnni::Graph> g = std::make_unique<scnni::Graph>();
  g->LoadModel("../demo_net/demo_net.pnnx.param",
              "../demo_net/demo_net.pnnx.bin");
  scnni::Excecutor exe = scnni::Excecutor(std::move(g));

  scnni::Tensor<float> input_tensor;
  input_tensor.FromImage("../demo_net/examples/abstract_face.jpg", true);
  std::vector<scnni::Tensor<float>> input_batch;
  input_batch.push_back(input_tensor);

  exe.Input("0", input_batch);
  exe.Forward();
  std::vector<scnni::Tensor<float>> output_batch = exe.Output(); 

  output_batch.at(0).Show();
}