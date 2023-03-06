/*
 * @Author: zzh
 * @Date: 2023-03-04 06:41:48
 * @LastEditTime: 2023-03-06 16:44:09
 * @Description: 
 * @FilePath: /scnni/test/test_tensor.cpp
 */

// #include "sccni/tensor.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <gtest/gtest.h>
#include <iostream>

TEST(test_tensor, test_tensor_1)
{
Eigen::Tensor<float, 3> a(1, 2, 3);
Eigen::Tensor<float, 3> b(1, 2, 3);
a.setConstant(1);
b.setConstant(1);
using std::cout;
using std::endl;
cout << a.size() << endl;
cout << (a == b) << endl;
cout << (a == b).sum() << endl;
// std::cout << ((a == b).sum() == a.size()) << std::endl;
// EXPECT_EQ(a == b, 1);
// if(a == b)
//     std::cout << "afdadfsfdsf ";
// else
// a.e
    // std::cout << (a == b);
}