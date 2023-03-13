/*
 * @Author: zzh
 * @Date: 2023-03-04 
 * @LastEditTime: 2023-03-13 11:59:47
 * @Description: 
 * @FilePath: /SCNNI/test/test_layer.cpp
 */
#include "scnni/graph.hpp"
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <random>
using std::cout;
using std::endl;

TEST(relu_test, DISABLED_relu_only_1batch_test) {
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
  // for (int i = 0; i < 3; i++) {
  //   for (int j = 0; j < 2; j ++) {
  //     for (int k = 0; k < 2; k ++) {
  //       cout << output_data(j, k, i) << " ";
  //     }
  //     cout << endl;
  //   }
  //   cout << endl;
  // }
  // cout << endl;
}
TEST(flatten_test, relu_flatten_1batch_test) {
  srand(time(nullptr));
  std::cout << "In graph_test load params" << std::endl;
  std::unique_ptr<scnni::Graph> g = std::make_unique<scnni::Graph>();
  g->LoadModel("/ws/CourseProject/SCNNI/python_scripts/relu_flatten_net/relu_flatten_net.pnnx.param",
              "/ws/CourseProject/SCNNI/python_scripts/relu_flatten_net/relu_flatten_net.pnnx.bin");
  EXPECT_EQ(g->blobs_.size(), 3);
  EXPECT_EQ(g->operators_.size(), 4);
  scnni::Excecutor exe = scnni::Excecutor(std::move(g));

  scnni::Tensor<float> input_tensor(3, 2, 2);
  Eigen::Tensor<float, 3> input_data(2, 2, 3);
  for (int i = 0; i < 3; i ++) {
    for (int j = 0; j < 2; j ++) {
      for (int k = 0; k < 2; k ++) {
        input_data(j, k, i) = (rand() % 10) * (rand() % 2 == 0 ? -1 : 1);
      }
    }
  }
  for (int i = 0; i < 3; i ++) {
    for (int j = 0; j < 2; j ++) {
      for (int k = 0; k < 2; k ++) {
        cout << input_data(j, k, i) << " ";
      }
      cout << endl;
    }
    cout << endl;
  }
  cout << endl;

  input_tensor.SetData(input_data);
  std::vector<scnni::Tensor<float>> input_batch;
  input_batch.push_back(input_tensor);

  exe.Input("0", input_batch);
  exe.Forward();
  std::vector<scnni::Tensor<float>> output_batch = exe.Output(); 

  Eigen::Tensor<float, 3> output_data(12, 1, 1);
  output_data = output_batch[0].GetData();
  
  EXPECT_EQ(output_batch.size(), 1);
  EXPECT_EQ(output_data.size(), 12);
  for (int i = 0; i < 12; i++) {
    cout << output_data(i, 0, 0) << " ";
  }
  cout << endl;
  for (int i = 0; i < 3; i ++) {
    for (int j = 0; j < 2; j ++) {
      for (int k = 0; k < 2; k ++) {
        EXPECT_EQ(output_data(k+j*2+i*4, 0, 0), input_data(j, k, i) > 0 ? input_data(j, k, i) : 0);
      }
    }
  }
}
TEST(maxpool2d_test, kernel2_padding0_stride2_1batch_test) {
  srand(time(nullptr));
  std::cout << "In graph_test load params" << std::endl;
  std::unique_ptr<scnni::Graph> g = std::make_unique<scnni::Graph>();
  g->LoadModel("/ws/CourseProject/SCNNI/python_scripts/relu_maxpool_flatten_net/relu_maxpool_flatten_net.pnnx.param",
              "/ws/CourseProject/SCNNI/python_scripts/relu_maxpool_flatten_net/relu_maxpool_flatten_net.pnnx.bin");
  EXPECT_EQ(g->blobs_.size(), 4);
  EXPECT_EQ(g->operators_.size(), 5);
  scnni::Excecutor exe = scnni::Excecutor(std::move(g));

  scnni::Tensor<float> input_tensor(3, 4, 4);
  Eigen::Tensor<float, 3> input_data(4, 4, 3);
  for (int i = 0; i < 3; i ++) {
    for (int j = 0; j < 4; j ++) {
      for (int k = 0; k < 4; k ++) {
        input_data(j, k, i) = static_cast<float>(rand() % 10);
      }
    }
  }
  // cout << input_data << endl << endl;

  for (int i = 0; i < 3; i ++) {
    for (int j = 0; j < 4; j ++) {
      for (int k = 0; k < 4; k ++) {
        cout << input_data(j, k, i) << " ";
      }
      cout << endl;
    }
    cout << endl;
  }
  cout << endl;

  input_tensor.SetData(input_data);
  std::vector<scnni::Tensor<float>> input_batch;
  input_batch.push_back(input_tensor);

  exe.Input("0", input_batch);
  exe.Forward();
  std::vector<scnni::Tensor<float>> output_batch = exe.Output(); 

  Eigen::Tensor<float, 3> output_data(12, 1, 1);
  output_data = output_batch[0].GetData();

  for (int i = 0; i < 12; i ++) {
    cout << output_data(i) << " ";
  }
  cout << endl;
  
  for (int i = 0; i < 3; i ++) {
    for (int j = 0; j < 2; j ++) {
      for (int k = 0; k < 2; k ++) {
        EXPECT_EQ(output_data(k+2*j+4*i),
                  std::max(std::max(std::max(input_data(j * 2, k * 2, i),
                                             input_data(j * 2 + 1, k * 2, i)),
                                    input_data(j * 2, k * 2 + 1, i)),
                           input_data(j * 2 + 1, k * 2 + 1, i)));

        // cout << std::max(std::max(std::max(input_data(j * 2, k * 2, i),
        //                                      input_data(j * 2 + 1, k * 2, i)),
        //                             input_data(j * 2, k * 2 + 1, i)),
        //                    input_data(j * 2 + 1, k * 2 + 1, i)) << " ";
      }
      cout << endl;
    }
    cout << endl;
  }
}