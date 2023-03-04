/*
 * @Author: zzh
 * @Date: 2023-03-04 10:27:16
 * @LastEditTime: 2023-03-04 12:19:37
 * @Description: 
 * @FilePath: /SCNNI/test/test_layer.cpp
 */
#include <gtest/gtest.h>
#include <iostream>

TEST(layer_test, init_layer) {

}

auto main(int argc, char *argv[]) -> int {
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    if (ret == 0) {
        std::cout << "<<<SUCCESS>>>" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
    }
    return 0;
}