/*
 * @Author: zzh
 * @Date: 2023-03-04 06:41:48
 * @LastEditTime: 2023-03-06 17:06:32
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
for(int i = 0; i < 2; i++){
    for(int j = 0; j < 3; j++){
        a(0, i, j) = i;
        b(0, i, j) = j;
    }
}
using std::cout;
using std::endl;
cout << a << endl;
cout << b << endl;
Eigen::Tensor<int , 3> c = (a==b);
cout << c << endl;
cout << a.size() << endl;
cout << (a == b) << endl;
cout << a.sum() << endl;
cout << b.sum() << endl;
cout << (a == b).sum() << endl;
// std::cout << ((a == b).sum() == a.size()) << std::endl;
// EXPECT_EQ(a == b, 1);
// if(a == b)
//     std::cout << "afdadfsfdsf ";
// else
// a.e
    // std::cout << (a == b);
}