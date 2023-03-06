/*
 * @Author: zzh
 * @Date: 2023-03-04 
 * @LastEditTime: 2023-03-06 11:15:56
 * @Description: 
 * @FilePath: /scnni/include/scnni/tensor.hpp
 */

#ifndef SCNNI_TENSOR_HPP_
#define SCNNI_TENSOR_HPP_
#include "macros.h"
#include <bits/stdint-uintn.h>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace scnni {

template <typename T>
class Tensor {};


template <>
class Tensor<float> {
  public:
    explicit Tensor() = default;

    /**
     * @description: 声明构造函数(创建张量)
     * @param channels
     * @param rows
     * @param cols
     * @return {*}
     */
    explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
    
    /**
     * @description: 初始化shapes
     */
    explicit Tensor(const std::vector<uint32_t>& shapes);

    /**
     * @description: 构造拷贝函数
     * @param tensor
     * @return {*}
     */
    Tensor(const Tensor& tensor);
    
    /**
     * @description: 重载=，赋值拷贝函数
     */    
    auto operator=(const Tensor& tensor) -> Tensor<float>&;

    /**
     * @description: 移动构造函数
     * @param {Tensor&&} tensor
     * @return {*}
     */   
    Tensor(Tensor&& tensor) noexcept ;
    
    /**
     * @description: 重载=，赋值移动
     */    
    auto operator=(Tensor&& tenson) noexcept -> Tensor<float>&;

    auto Channels() const -> uint32_t;  //返回张量的通道数（第一维度）
    auto Rows() const -> uint32_t;  //返回张量的行数（第二维度）
    auto Cols() const -> uint32_t;  //返回张量的列数（第三维度）
    auto Size() const -> uint32_t;  //返回张量中元素的数量（放了多少个float元素）
    void Set_data(const Eigen::Tensor<float, 3, Eigen::RowMajor>& data);  //赋值，设置张量中的具体数据
    bool Empty() const;  //判空，返回张量是否为空

    /**
     * @description: 返回张量展平后，顺序位置（offset位置）的元素中。index(c*Rows*Cols + row*Cols + col) = At(c, row, col)
     * @param offset 需要访问的位置
     * @return offset位置的元素
     */
    float Index(uint32_t offset) const;

    /**
     * @description: 返回张量中offset位置的元素
     * @param offset 需要访问的位置
     * @return offset位置的元素
     */
    float& Index(uint32_t offset);

    std::vector<uint32_t> Shapes() const;  //张量的尺寸大小
    const std::vector<uint32_t>& Raw_shapes() const;  //张量的实际尺寸大小

    /**
     * @description: 返回张量中的裸数据
     * @return 张量中的数据
     */
    Eigen::Tensor<float, 3, Eigen::RowMajor>& data();

    /**
     * @description: 返回张量中的数据
     * @return 张量中的数据
     */
    const Eigen::Tensor<float, 3, Eigen::RowMajor>& data() const;

    /**
     * @description: 返回张量第channel通道中的数据
     * @param channel 需要返回的通道
     * @return 返回的通道
     */
    Eigen::MatrixXf& Slice(uint32_t channel);

    /**
     * 返回张量第channel通道中的数据
     * @param channel 需要返回的通道
     * @return 返回的通道
     */
    const Eigen::MatrixXf& Slice(uint32_t channel) const;

    /**
     * @description: 返回特定位置的元素
     * @param channel 通道
     * @param row 行数
     * @param col 列数
     * @return 特定位置的元素
     */
    float At(uint32_t channel, uint32_t row, uint32_t col) const;

    /**
     * 返回特定位置的元素
     * @param channel 通道
     * @param row 行数
     * @param col 列数
     * @return 特定位置的元素
     */
    float& At(uint32_t channel, uint32_t row, uint32_t col);

    /**
     * @description: 填充张量
     * @param pads 填充张量的尺寸
     * @param padding_value 填充张量
     */
    void Padding(const std::vector<uint32_t>& pads, float padding_value);

    /**
     * @description: 使用value值去初始化向量
     * @param value
     */
    void Fill(float value);

    /**
     * @description: 使用values中的数据初始化张量
     * @param values 用来初始化张量的数据
     */
    void Fill(const std::vector<float>& values);

    /**
     * @description: 以常量1初始化张量
     */
    void Ones();

    /**
     * @description: 以随机值初始化张量
     */
    void Rand();

    /**
     * @description: 打印张量
     */
    void Show();

    /// reshape和review的区别
    /// reshape是满足列优先的
    /// review是满足行优先的
    /**
     * 1 3 4 7
     * review(2,2)  reshape(2,2)
     * 1 3          1 4
     * 4 7          3 7
     */
    
    /**
     * @description: 张量的实际尺寸大小的Reshape
     * @param shapes 张量的实际尺寸大小
     */
    void ReRawshape(const std::vector<uint32_t>& shapes);

    /**
     * @description: 张量的实际尺寸大小的Reshape pytorch兼容
     * @param shapes 张量的实际尺寸大小
     */
    void ReRawView(const std::vector<uint32_t>& shapes);

    /**
     * @description: 展开张量
     */    
    void Flatten();

    /**
     * @description: 对张量中的元素进行过滤
     * @param filter 过滤函数
     */
    void Transform(const std::function<float(float)>& filter);

    /**
     * @description: 返回一个深拷贝后的张量
     * @return 新的张量
     */
    std::shared_ptr<Tensor> Clone();

    /**
     * @description: 返回数据的原始指针
     */
    const float* raw_ptr() const;

  private:
    void Review(const std::vector<uint32_t>& shapes);
    std::vector<uint32_t> raw_shapes_;  // 张量数据的实际尺寸大小
    Eigen::Tensor<float, 3, Eigen::RowMajor> data_;  // 张量数据
};


using ftensor = Tensor<float>;
using std::shared_ptr<Tensor<float>> = std::shared_ptr<Tensor<float>>;

/**
 * 创建一个张量
 * @param channels 通道数量
 * @param rows 行数
 * @param cols 列数
 * @return 创建后的张量
 */
std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows, uint32_t cols);

/**
 * 创建一个张量
 * @param shapes 张量的形状
 * @return 创建后的张量
 */
std::shared_ptr<Tensor<float>> TensorCreate(const std::vector<uint32_t>& shapes);

std::tuple<std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<float>>> TensorBroadcast(const std::shared_ptr<Tensor<float>> &s1, const std::shared_ptr<Tensor<float>> &s2);

std::shared_ptr<Tensor<float>> TensorPadding(
    const std::shared_ptr<Tensor<float>>& tensor,
    const std::vector<uint32_t>& pads, float padding_value);

/**
 * 比较tensor的值是否相同
 * @param a 输入张量1
 * @param b 输入张量2
 * @return 比较结果
 */
bool TensorIsSame(const std::shared_ptr<Tensor<float>>& a,
                  const std::shared_ptr<Tensor<float>>& b);

/**
 * 张量相加
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @param output_tensor 输出张量
 */
void TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
                      const std::shared_ptr<Tensor<float>>& tensor2,
                      const std::shared_ptr<Tensor<float>>& output_tensor);

/**
 * 张量相加
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @return 张量相加的结果
 */
std::shared_ptr<Tensor<float>> TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
																								const std::shared_ptr<Tensor<float>>& tensor2);

/**
 * @description: 矩阵点乘
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @param output_tensor 输出张量
 */
void TensorElementMultiply(const std::shared_ptr<Tensor<float>>& tensor1,
														const std::shared_ptr<Tensor<float>>& tensor2,
														const std::shared_ptr<Tensor<float>>& output_tensor);

/**
 * 张量相乘
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @return 张量相乘的结果
 */
std::shared_ptr<Tensor<float>> TensorElementMultiply(const std::shared_ptr<Tensor<float>>& tensor1,
																											const std::shared_ptr<Tensor<float>>& tensor2);



} // namespace scnni

#endif // SCNNI_TENSOR_HPP_