/*
 * @Author: xzj
 * @Date: 2023-03-06 07:01:23
 * @LastEditTime: 2023-03-06 10:13:43
 * @Description: 
 * @FilePath: /scnni/src/tmp.cpp
 */
//
// Created by fss on 22-11-12.
//

// #include "scnni/tensor.hpp"
#include <bits/stdint-uintn.h>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <glog/logging.h>
#include <memory>
#include <iostream>

Eigen::Tensor<int, 3> a(2, 4, 3);
// int main() {
//     a.setValues({{ {0, 100, 200}, {300, 400, 500}, {600, 700, 800}, {900, 1000, 1100} },
//                     { {0, 100, 200}, {300, 400, 500}, {600, 700, 800}, {900, 1000, 1100} }});
//     std::cout << a << std::endl;
// //     // Eigen::TensorMap<Eigen::Tensor<float, 3, >> b(a.data(), 4, 2, 3);
//     Eigen::Tensor<int, 3> b(2, , 3);
//     b = a;
//     // a.Reshape(4, 2, 3);

//     std::cout << b << std::endl;
    
// }

int main(){
    Eigen::MatrixXf a(2, 3), b(2, 3), c(2, 3);
    for(int i=0;i<2;++i){
        // 对矩阵进行初始化
        a(i,0)=i;
        a(i,1)=i;
        a(i,2)=i;
        b(i,0)=i;
        b(i,1)=i;
        b(i,2)=i;
        c(i,0)=2*i;
        c(i,1)=2*i;
        c(i,2)=2*i;
    }
    bool samea = (a == b);
    bool samec = (c == b);

    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    std::cout << samea << std::endl;
    std::cout << samec << std::endl;
}