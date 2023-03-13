/*
 * @Author: zzh
 * @Date: 2023-03-04 06:41:48
 * @LastEditTime: 2023-03-13 10:17:31
 * @Description: 
 * @FilePath: /SCNNI/test/test_tensor.cpp
 */

// #include "sccni/tensor.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/util/Constants.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <gtest/gtest.h>
#include <iostream>
using std::cout;
using std::endl;

TEST(test_tensor, simple_tensor)
{
  float data[18];
  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> a(data, 3, 2, 3);
  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::ColMajor>> b(data, 3, 2, 3);
  for (int i = 0; i < 18; i++) {
    data[i] = i;
  }
  cout << "a:" << endl;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 3; k++) {
        cout << a(i, j, k) << " ";
      }
      cout << endl;
    }
    cout << endl << endl;;
  }

  cout << "b:" << endl;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j ++) {
      for (int k = 0; k < 3; k++) {
        cout << b(i, j, k) << " ";
      }
      cout << endl;
    }
    cout << endl << endl;
  }

  cout << "a.size(): ";
  cout << a.size() << endl;
}

TEST(test_tensor, simple_tensor2)
{
    Eigen::Tensor<float, 3> a(3, 2, 3);
    int cnt = 0;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 3; k++) {
          a(i, j, k) = ++cnt;
        }
      }
    }
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 3; k++) {
          cout << a(i, j, k) << " ";
        }
        cout << endl;
      }
      cout << endl;
    }
    cout << endl;
    Eigen::Tensor<float, 2, Eigen::ColMajor> b(3, 6);
    Eigen::array<Eigen::DenseIndex, 2> new_dim({3, 6});
    // b = a.reshape(new_dim);
    b = a.swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0}).reshape(new_dim).swap_layout().shuffle(Eigen::array<int, 2>{1, 0});
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 6; j++) {
          cout << b(i, j) << " ";
      }
      cout << endl;
    }
    cout << endl;

    Eigen::Tensor<float, 3, Eigen::ColMajor> c(3, 3, 2);
    Eigen::array<Eigen::DenseIndex, 3> new_dim_2({3, 3, 2});
    // c = b.reshape(new_dim_2);
    c = a.swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0}).reshape(new_dim_2).swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0});
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 2; k++) {
          cout << c(i, j, k) << " ";
        }
        cout << endl;
      }
      cout << endl << endl;
    }
    cout << endl;

    Eigen::Tensor<float, 3, Eigen::ColMajor> d(18, 1, 1);
    Eigen::array<Eigen::DenseIndex, 3> new_dim_3({18, 1, 1});
    // d = b.reshape(new_dim_3);
    d = a.swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0}).reshape(new_dim_3).swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0});
    for (int i = 0; i < 18; i ++) {
      cout << d(i, 0, 0) << " ";
    }
    cout << endl;
}

TEST(test_tensor, simple_tensor3)
{
    Eigen::Tensor<float, 3> a(2, 4, 3);
    int cnt = 0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 3; k++) {
          a(i, j, k) = ++cnt;
        }
      }
    }
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 4; k++) {
          cout << a(j, k, i) << " ";
        }
        cout << endl;
      }
      cout << endl;
    }
    cout << endl;
  Eigen::Tensor<float, 3, Eigen::RowMajor> b = a.swap_layout();
  for (int c = 0; c < b.dimension(2); c ++) {
    for (int h = 0; h < b.dimension(0); h ++) {
      for (int w = 0 ; w < b.dimension(1); w ++) {
        cout << b(h, w, c) << " ";
      }
      cout << endl;
    }
    cout << endl;
  }
  cout << endl;

  Eigen::Tensor<float, 3, Eigen::RowMajor> c = b.shuffle(Eigen::array<int, 3>{1, 2, 0});
  for (int ch = 0; ch < c.dimension(2); ch ++) {
    for (int h = 0; h < c.dimension(0); h ++) {
      for (int w = 0 ; w < c.dimension(1); w ++) {
        cout << c(h, w, ch) << " ";
      }
      cout << endl;
    }
    cout << endl;
  }
  cout << endl;

  float* array = c.data();
  for (int i = 0; i < 24; i ++) {
    cout << array[i] << " ";
  }
  cout << endl;

  cout << "d: " << endl;
  Eigen::Tensor<float, 3, Eigen::RowMajor> d = c.reshape(Eigen::array<Eigen::DenseIndex, 3> {2, 12, 1});
  for (int ch = 0; ch < d.dimension(2); ch ++) {
    for (int h = 0; h < d.dimension(0); h ++) {
      for (int w = 0 ; w < d.dimension(1); w ++) {
        cout << d(h, w, ch) << " ";
      }
      cout << endl;
    }
    cout << endl;
  }
  cout << endl;

  Eigen::Tensor<float, 3, Eigen::ColMajor> e = d.swap_layout();
  for (int ch = 0; ch < e.dimension(2); ch ++) {
    for (int h = 0; h < e.dimension(0); h ++) {
      for (int w = 0 ; w < e.dimension(1); w ++) {
        cout << e(h, w, ch) << " ";
      }
      cout << endl;
    }
    cout << endl;
  }
  cout << endl;

  Eigen::Tensor<float, 3, Eigen::ColMajor> f = e.shuffle(Eigen::array<int, 3>{2, 1, 0});
  for (int ch = 0; ch < f.dimension(2); ch ++) {
    for (int h = 0; h < f.dimension(0); h ++) {
      for (int w = 0 ; w < f.dimension(1); w ++) {
        cout << f(h, w, ch) << " ";
      }
      cout << endl;
    }
    cout << endl;
  }
  cout << endl;


}

//>>> a = torch.Tensor([[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], [[[12, 13], [14, 15], [16, 17]], [[18, 19], [20, 21], [22, 23]]], [[[24, 25], [26, 27], []], [[], [], []]], [[[], [], []], [[], [], []]]])