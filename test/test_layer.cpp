/*
 * @Author: zzh
 * @Description: 
 * @FilePath: /scnni/test/test_layer.cpp
 */
#include "scnni/graph.hpp"
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <random>
using std::cout;
using std::endl;

// xzj_path = ../...
// zzh_path = ../...




TEST(relu_test, DISABLED_relu_only_1batch_test) {
  std::cout << "In graph_test load params" << std::endl;
  std::unique_ptr<scnni::Graph> g = std::make_unique<scnni::Graph>();
  g->LoadModel("../python_scripts/relu_only_net/relu_only_net.pnnx.param",
              "../python_scripts/relu_only_net/relu_only_net.pnnx.bin");
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

TEST(flatten_test, DISABLED_relu_flatten_1batch_test) {
  srand(time(nullptr));
  std::cout << "In graph_test load params" << std::endl;
  std::unique_ptr<scnni::Graph> g = std::make_unique<scnni::Graph>();
  g->LoadModel("../python_scripts/relu_flatten_net/relu_flatten_net.pnnx.param",
              "../python_scripts/relu_flatten_net/relu_flatten_net.pnnx.bin");
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

TEST(maxpool2d_test, DISABLED_kernel2_padding0_stride2_1batch_test) {
  srand(time(nullptr));
  std::cout << "In graph_test load params" << std::endl;
  std::unique_ptr<scnni::Graph> g = std::make_unique<scnni::Graph>();
  g->LoadModel("../python_scripts/relu_maxpool_flatten_net/relu_maxpool_flatten_net.pnnx.param",
              "../python_scripts/relu_maxpool_flatten_net/relu_maxpool_flatten_net.pnnx.bin");
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

TEST(softmax_test, DISABLED_softmax_only_1batch_test) {
    srand(time(nullptr));
    std::cout << "In graph_test load params" << std::endl;
    std::unique_ptr<scnni::Graph> g = std::make_unique<scnni::Graph>();
    g->LoadModel("../python_scripts/softmax_only_net/softmax_net.pnnx.param",
                "../python_scripts/softmax_only_net/softmax_net.pnnx.bin");
    EXPECT_EQ(g->blobs_.size(), 2);
    EXPECT_EQ(g->operators_.size(), 3);
    scnni::Excecutor exe = scnni::Excecutor(std::move(g));

    scnni::Tensor<float> input_tensor(1, 5, 1);
    Eigen::Tensor<float, 3> input_data(5, 1, 1);
    for (int i = 0; i < 5; i ++) {
        input_data(i, 0, 0) = static_cast<float>(rand() % 10);
    }
    // cout << input_data << endl << endl;

    for (int i = 0; i < 5; i ++) {
        cout << input_data(i, 0, 0) << " ";
    }
    cout << endl;

    input_tensor.SetData(input_data);
    std::vector<scnni::Tensor<float>> input_batch;
    input_batch.push_back(input_tensor);

    exe.Input("0", input_batch);
    exe.Forward();
    std::vector<scnni::Tensor<float>> output_batch = exe.Output(); 

    Eigen::Tensor<float, 3> output_data(5, 1, 1);
    output_data = output_batch[0].GetData();

    for (int i = 0; i < 5; i ++) {
        cout << output_data(i, 0, 0) << " ";
    }
    cout << endl;

}

TEST(linear_test, DISABLED_infeat5_outfeat3_input1x5_1batch_test) {
  srand(time(nullptr));
  std::cout << "In graph_test load params" << std::endl;
  std::unique_ptr<scnni::Graph> g = std::make_unique<scnni::Graph>();
  g->LoadModel("../python_scripts/linear_net/linear_net.pnnx.param",
              "../python_scripts/linear_net/linear_net.pnnx.bin");
  EXPECT_EQ(g->blobs_.size(), 2);
  EXPECT_EQ(g->operators_.size(), 3);
  scnni::Excecutor exe = scnni::Excecutor(std::move(g));

  scnni::Tensor<float> input_tensor(1, 5, 1);
  Eigen::Tensor<float, 3> input_data(5, 1, 1);
  for (int i = 0; i < 5; i ++) {
    // input_data(i, 0, 0) = static_cast<float>(rand() % 10);
    input_data(i, 0, 0) = i;
  }

  input_tensor.SetData(input_data);
  std::vector<scnni::Tensor<float>> input_batch;
  input_batch.push_back(input_tensor);

  exe.Input("0", input_batch);
  exe.Forward();
  std::vector<scnni::Tensor<float>> output_batch = exe.Output(); 

  Eigen::Tensor<float, 3> output_data(3, 1, 1);
  output_data = output_batch[0].GetData();

  for (int i = 0; i < 3; i ++) {
    cout << output_data(i, 0, 0) << " ";
  }
  cout << endl;
 
}

TEST(combine_test, DISABLED_input3x4x4_output12x1_test) {
    srand(time(nullptr));
    std::cout << "In graph_test load params" << std::endl;
    std::unique_ptr<scnni::Graph> g = std::make_unique<scnni::Graph>();
    g->LoadModel("../python_scripts/relu_maxpool_flatten_linear_softmax_net/relu_maxpool_flatten_linear_softmax_net.pnnx.param",
                "../python_scripts/relu_maxpool_flatten_linear_softmax_net/relu_maxpool_flatten_linear_softmax_net.pnnx.bin");
    EXPECT_EQ(g->blobs_.size(), 6);
    EXPECT_EQ(g->operators_.size(), 7);
    scnni::Excecutor exe = scnni::Excecutor(std::move(g));

    scnni::Tensor<float> input_tensor(3, 4, 4);
    Eigen::Tensor<float, 3> input_data(4, 4, 3);

    float cnt = 0.0;
    for (int i = 0; i < 3; i ++) {
        for (int j = 0; j < 4; j ++) {
            for (int k = 0; k < 4; k ++) {
                input_data(j, k, i) = cnt;
                cnt = cnt + 1.0;
            }
        }
    }
    
    // for (int i = 0; i < 3; i ++) {
    //     for (int j = 0; j < 4; j ++) {
    //         for (int k = 0; k < 4; k ++) {
    //             cout << input_data(j, k, i) << " ";
    //         }
    //         cout << endl;
    //     }
    //     cout << endl;
    // }
    // cout << endl;

    input_tensor.SetData(input_data);
    std::vector<scnni::Tensor<float>> input_batch;
    input_batch.push_back(input_tensor);

    exe.Input("0", input_batch);
    exe.Forward();
    std::vector<scnni::Tensor<float>> output_batch = exe.Output(); 

    Eigen::Tensor<float, 3> output_data(3, 1, 1);
    output_data = output_batch[0].GetData();

    for (int i = 0; i < 3; i ++) {
        cout << output_data(i, 0, 0) << " ";
    }
    cout << endl;

    EXPECT_NEAR(output_data(0, 0, 0), 0.988494, 1e-6);
    EXPECT_NEAR(output_data(1, 0, 0), 0.00873744, 1e-6);
    EXPECT_NEAR(output_data(2, 0, 0), 0.00276862, 1e-6);


}

TEST(conv2d_test, DISABLED_conv2d_test1) {
  srand(time(nullptr));
  std::cout << "In graph_test load params" << std::endl;
  std::unique_ptr<scnni::Graph> g = std::make_unique<scnni::Graph>();
  g->LoadModel("../python_scripts/covn2d_net/conv2d_test1/conv2d_test1.pnnx.param",
              "../python_scripts/covn2d_net/conv2d_test1/conv2d_test1.pnnx.bin");
  EXPECT_EQ(g->blobs_.size(), 2);
  EXPECT_EQ(g->operators_.size(), 3);
  scnni::Excecutor exe = scnni::Excecutor(std::move(g));

  scnni::Tensor<float> input_tensor(2, 5, 5);
  Eigen::Tensor<float, 3> input_data(5, 5, 2);
  for (int i = 0; i < 2; i ++) {
    for (int j = 0; j < 5; j ++) {
      for (int k = 0; k < 5; k ++) {
        input_data(j, k, i) = k + j * 5 + i * 25;
      }
    }
  }

  for (int c = 0; c < 2; c ++) {
    for (int h = 0; h < 5; h ++) {
      for (int w = 0; w < 5; w ++) {
        cout << input_data(h, w, c) << " ";
      }
      cout << endl;
    }
    cout << endl;
  }
  input_tensor.SetData(input_data);
  std::vector<scnni::Tensor<float>> input_batch;
  input_batch.push_back(input_tensor);

  exe.Input("0", input_batch);
  exe.Forward();
  std::vector<scnni::Tensor<float>> output_batch = exe.Output(); 

  Eigen::Tensor<float, 3> output_data(3, 3, 4);
  output_data = output_batch[0].GetData();

  for (int c = 0; c < 4; c ++) {
    for (int h = 0; h < 3; h ++) {
      for (int w = 0; w < 3; w ++) {
        cout << output_data(h, w, c) << " ";
      }
      cout << endl;
    }
    cout << endl;
  }
  EXPECT_NEAR(output_data(0, 0, 0), -6.6528, 1e-4);
  EXPECT_NEAR(output_data(2, 1, 0), -14.9152, 1e-4);
  EXPECT_NEAR(output_data(1, 1, 1), 3.8489, 1e-4);
  EXPECT_NEAR(output_data(0, 2, 2), 10.7468, 1e-4);
  EXPECT_NEAR(output_data(1, 0, 2), 20.7828, 1e-4);
  EXPECT_NEAR(output_data(2, 2, 3), 19.9446, 1e-4);

  
}
TEST(demo_test, DISABLED_demo_test_1) {
    srand(time(nullptr));
    std::cout << "In graph_test load params" << std::endl;
    std::unique_ptr<scnni::Graph> g = std::make_unique<scnni::Graph>();
    g->LoadModel("../demo_net/demo_net.pnnx.param",
                "../demo_net/demo_net.pnnx.bin");
    EXPECT_EQ(g->blobs_.size(), 12);
    EXPECT_EQ(g->operators_.size(), 13);
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