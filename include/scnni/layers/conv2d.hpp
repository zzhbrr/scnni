/*
 * @Author: zzh
 * @Date: 2023-03-14 
 * @LastEditTime: 2023-03-15 07:10:56
 * @Description: 
 * @FilePath: /scnni/include/scnni/layers/conv2d.hpp
 */
#ifndef SCNNI_CONV2D_HPP_
#define SCNNI_CONV2D_HPP_

#include "scnni/layer.hpp"
#include "scnni/operator.hpp"

namespace scnni {
class Con2dLayer: public Layer {
  public:
     auto Forward(const std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &output_blobs) const -> int override;
    ~Con2dLayer() override = default;

    void SetBias(bool bias);
    void SetDilation(const std::vector<int>& dilation);
    void SetInChannels(int in_channels);
    void SetOutChannels(int out_channels);
    void SetGroups(int groups);
    void SetKernelSize(const std::vector<int>& kernel_size);
    void SetPadding(const std::vector<int>& padding);
    void SetPaddingmode(const std::string& padding_mode);
    void SetStiride(const std::vector<int>& stride);
    void SetWeights(const Attribute& att);
    void SetBiasValue(const Attribute& att);

  private:
    bool bias_;
    std::vector<int> dilation_;
    int groups_;
    int in_channels_;
    int out_channels_;
    std::vector<int> kernel_size_;
    std::vector<int> padding_;
    std::string padding_mode_;
    std::vector<int> stride_;

    std::vector<std::shared_ptr<Tensor<float>>> weights_; // 多个输出通道，多组卷积核
    std::shared_ptr<Tensor<float>> bias_v_; // bias一维的
};
} // namespace scnni

#endif

