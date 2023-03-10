/*
 * @Author: zzh
 * @Date: 2023-03-04 
 * @LastEditTime: 2023-03-10 06:51:36
 * @Description: 
 * @FilePath: /SCNNI/test/test_layer.cpp
 */
#include "scnni/graph.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
using std::cout;
using std::endl;

TEST(layer_test, relu_only_1batch_test) {
  std::cout << "In graph_test load params" << std::endl;
  std::unique_ptr<scnni::Graph> g = std::make_unique<scnni::Graph>();
  g->LoadModel("/ws/CourseProject/SCNNI/python_scripts/relu_only_net/relu_only_net.pnnx.param",
              "/ws/CourseProject/SCNNI/python_scripts/relu_only_net/relu_only_net.pnnx.bin");
  EXPECT_EQ(g->blobs_.size(), 2);
  EXPECT_EQ(g->operators_.size(), 3);
  scnni::Excecutor exe = scnni::Excecutor(std::move(g));

  scnni::Tensor<float> input_tensor(3, 2, 2);
  Eigen::Tensor<float, 3> input_data(2, 2, 3);
  for (int i = 0; i < 3; i ++) {
    for (int j = 0; j < 2; j ++) {
      for (int k = 0; k < 2; k ++) {
        input_data(j, k, i) = ((j + k) % 2) != 0 ? 12.324 : -12121.123;
      }
    }
  }
  // for (int i = 0; i < 3; i ++) {
  //   for (int j = 0; j < 2; j ++) {
  //     for (int k = 0; k < 2; k ++) {
  //       cout << input_data(j, k, i) << " ";
  //     }
  //     cout << endl;
  //   }
  //   cout << endl;
  // }
  // cout << endl;

  input_tensor.SetData(input_data);
  std::vector<scnni::Tensor<float>> input_batch;
  input_batch.push_back(input_tensor);

  exe.Input("0", input_batch);
  exe.Forward();
  std::vector<scnni::Tensor<float>> output_batch = exe.Output(); 

  Eigen::Tensor<float, 3> output_data(2, 2, 3);
  output_data = output_batch[0].GetData();
  EXPECT_EQ(output_batch.size(), 1);
  EXPECT_EQ(output_data.size(), 12);
  for (int i = 0; i < 3; i ++) {
    for (int j = 0; j < 2; j ++) {
      for (int k = 0; k < 2; k ++) {
        EXPECT_EQ(output_data(j, k, i), input_data(j, k, i) > 0 ? input_data(j, k, i) : 0);
      }
    }
  }
}
