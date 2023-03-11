/*
 * @Author: zzh
 * @Date: 2023-03-04 06:41:48
 * @LastEditTime: 2023-03-10 15:32:40
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
    Eigen::Tensor<float, 3, Eigen::RowMajor> a(3, 2, 3);
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
      cout << endl << endl;
    }

    Eigen::Tensor<float, 2, Eigen::RowMajor> b(3, 6);
    Eigen::array<Eigen::DenseIndex, 2> new_dim({3, 6});
    b = a.reshape(new_dim);
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 6; j++) {
          cout << b(i, j) << " ";
      }
      cout << endl;
    }

    Eigen::Tensor<float, 3, Eigen::RowMajor> c(3, 3, 2);
    Eigen::array<Eigen::DenseIndex, 3> new_dim_2({3, 3, 2});
    c = b.reshape(new_dim_2);
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 2; k++) {
          cout << c(i, j, k) << " ";
        }
        cout << endl;
      }
      cout << endl << endl;
    }



}