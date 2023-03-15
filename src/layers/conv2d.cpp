#include "scnni/layers/conv2d.hpp"
#include "scnni/layer_factory.hpp"
#include "scnni/operator.hpp"
#include "scnni/macros.h"
#include "scnni/logger.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <bits/stdint-uintn.h>
#include <cfloat>
#include <iostream>
#include <memory>
#include <sys/types.h>
#include <vector>

namespace scnni {
auto Con2dLayer::Forward(const std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>>
                &output_blobs) const -> int {
    SCNNI_ASSERT(!input_blobs.empty(), "Conv2dLayer's input blobs empty");
    SCNNI_ASSERT(input_blobs.size() == 1, "Conv2dLayer has multiple inputs");
    SCNNI_ASSERT(!output_blobs.empty(), "Conv2dLayer's output blobs empty");
    if (groups_ != 1) {
        LOG_ERROR("Conv2dLayer: groups conv not implemented");
    }
    if (padding_mode_ != "zeros") {
        LOG_ERROR("Conv2dLayer: only padding_mode=zeros supported yet");
    }
    for (size_t batch = 0; batch < input_blobs[0].size(); batch++) {
        const auto input_tensor_shptr = input_blobs[0][batch];
        const std::shared_ptr<Tensor<float>> feat = output_blobs[0].at(0);
        auto in_shape = input_tensor_shptr->Shapes();

        Tensor<float> after_padding = *input_tensor_shptr;
        after_padding.Padding({static_cast<unsigned int>(padding_[0]),
                               static_cast<unsigned int>(padding_[0]),
                               static_cast<unsigned int>(padding_[1]),
                               static_cast<unsigned int>(padding_[1])},
                               0);
        SCNNI_ASSERT((int)feat->Shapes()[0] == out_channels_, "Conv2d: output channels not match output blob's shape");
        int out_h = (input_tensor_shptr->Rows() + 2 * padding_[0] -
                         dilation_[0] * (kernel_size_[0] - 1)) / stride_[0] + 1;
        int out_w = (input_tensor_shptr->Cols() + 2 * padding_[1] -
                         dilation_[1] * (kernel_size_[1] - 1)) / stride_[1] + 1;
        // LOG_DEBUG("Conv2dLayer: out_h:%d, out_w:%d", out_h, out_w);
        SCNNI_ASSERT(out_h == (int)feat->Rows(), "Conv2dLayer: shape not right");
        SCNNI_ASSERT(out_w == (int)feat->Cols(), "Conv2dLayer: shape not right");
        Eigen::MatrixXf kernel_mat(out_channels_, kernel_size_[0]*kernel_size_[1]*in_channels_);
        Eigen::MatrixXf img_mat(kernel_size_[0]*kernel_size_[1]*in_channels_, out_h * out_w);
        Eigen::MatrixXf mulres;

        for (int outc = 0; outc < out_channels_; outc ++) {
            int cnt = 0;
            for (int c = 0; c < in_channels_; c ++) {
                for (int i = 0; i < kernel_size_[0]; i ++) {
                    for (int j = 0; j < kernel_size_[1]; j ++) {
                        kernel_mat(outc, cnt) = weights_.at(outc)->At(c, i, j);
                        cnt++;
                    }
                }
            }
        }
        int colcnt = 0;
        int step_h = 0;
        for (uint32_t h = 0; h < after_padding.Rows(); h += stride_[0]) {
            if (step_h == out_h) {
                break;
            }
            step_h++;
            int step_w = 0;
            for (uint32_t w = 0; w < after_padding.Cols(); w += stride_[1]) {
                if (step_w == out_w) {
                    break;
                }
                step_w++;
                int rowcnt = 0;
                for (int c = 0; c < in_channels_; c++) {
                    for (int i = 0; i < kernel_size_[0]; i++) {
                        for (int j = 0; j < kernel_size_[1]; j++) {
                            img_mat(rowcnt, colcnt) = after_padding.At(c, h + i * dilation_[0], w + j * dilation_[1]); 
                            rowcnt++;
                        }
                    }
                }
                colcnt++;
            }
        }
        after_padding.Show();
        for (int i = 0; i < kernel_size_[0]*kernel_size_[1]*in_channels_; i ++) {
            for (int j = 0; j < out_h*out_w; j ++) {
                std::cout << img_mat(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "ready to matmul" << std::endl;
        mulres = kernel_mat * img_mat; // out_channels x (out_w * out_h)
        for (int c = 0; c < out_channels_; c ++) {
            for (int i = 0; i < out_h; i ++) {
                for (int j = 0; j < out_w; j ++) {
                    feat->At(c, i, j) = mulres(c, j+i*out_w);
                    if (bias_) {
                        feat->At(c, i, j) += bias_v_->At(0, c, 0);
                    }
                }
            }
        }
    }
    
    return 0;
}
void Con2dLayer::SetBias(bool bias) {
    bias_ = bias;
}
void Con2dLayer::SetDilation(const std::vector<int> &dilation) {
    dilation_ = dilation;
}
void Con2dLayer::SetInChannels(int in_channels) {
    in_channels_ = in_channels;
}
void Con2dLayer::SetOutChannels(int out_channels) {
    out_channels_ = out_channels;
}
void Con2dLayer::SetGroups(int groups) {
    groups_ = groups;
}
void Con2dLayer::SetKernelSize(const std::vector<int> &kernel_size) {
    kernel_size_ = kernel_size;
}
void Con2dLayer::SetPadding(const std::vector<int> &padding) {
    padding_ = padding;
}
void Con2dLayer::SetPaddingmode(const std::string &padding_mode) {
    padding_mode_ = padding_mode;
}
void Con2dLayer::SetStiride(const std::vector<int> &stride) {
    stride_ = stride;
}

void Con2dLayer::SetWeights(const Attribute &att) {
    SCNNI_ASSERT(att.shape_.size() == 4, "Conv2d: weight att shape != 4");
    // LOG_DEBUG("Conv2d weight shape: [%d %d %d %d]", att.shape_[0], att.shape_[1], att.shape_[2], att.shape_[3]);
    std::vector<float> weights = att.Get();
    this->weights_.resize(att.shape_[0]);
    std::cout << "Linear weight:" << std::endl;
    for (int i = 0; i < att.shape_[0]; i ++) {
        this->weights_.at(i) = std::make_shared<Tensor<float>>(att.shape_[1], att.shape_[2], att.shape_[3]);
        std::cout << i << "'th" << std::endl;
        for (int c = 0; c < att.shape_[1]; c ++) {
            for (int h = 0; h < att.shape_[2]; h ++) {
                for (int w = 0; w < att.shape_[3]; w ++) {
                    this->weights_.at(i)->At(c, h, w) = weights.at(w + h * att.shape_[3] + c * att.shape_[2] * att.shape_[3] + i * att.shape_[1]*att.shape_[2]*att.shape_[3]);
                    std::cout << this->weights_.at(i)->At(c, h, w) << " ";
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }
}
void Con2dLayer::SetBiasValue(const Attribute &att) {
    SCNNI_ASSERT(att.shape_.size() == 1, "Conv2d: bias att shape != 1");
    std::vector<float> bias_value = att.Get();
    this->bias_v_ = std::make_shared<Tensor<float>>(1, att.shape_[0], 1);
    std::cout << "Conv2d bias:" << std::endl;
    for (int i = 0; i < att.shape_[0]; i ++) {
        this->bias_v_->At(0, i, 0) = bias_value.at(i);
        std::cout << this->bias_v_->At(0, i, 0) << " ";
    }
    std::cout << std::endl;
}
auto GetConv2dLayer(const std::shared_ptr<Operator> &op) -> Layer* {
    auto* layer = new Con2dLayer();
    Parameter p = op->GetParam("bias");
    bool use_bias = p.GetValueBool();
    layer->SetBias(p.GetValueBool());
    p = op->GetParam("dilation");
    layer->SetDilation(p.GetValueIntArray());
    p = op->GetParam("groups");
    layer->SetGroups(p.GetValueInt());
    p = op->GetParam("in_channels");
    layer->SetInChannels(p.GetValueInt());
    p = op->GetParam("out_channels");
    layer->SetOutChannels(p.GetValueInt());
    p = op->GetParam("kernel_size");
    layer->SetKernelSize(p.GetValueIntArray());
    p = op->GetParam("padding");
    layer->SetPadding(p.GetValueIntArray());
    p = op->GetParam("padding_mode");
    layer->SetPaddingmode(p.GetValueString());
    p = op->GetParam("stride");
    layer->SetStiride(p.GetValueIntArray());
    Attribute att = op->GetAttr("weight");
    layer->SetWeights(att);
    if (use_bias) {
      att = op->GetAttr("bias");
      layer->SetBiasValue(att);
    }
    return layer;
} 

LayerRegistelrWrapper conv2d_layer_registe("nn.Conv2d", LayerRegister::layer_creator_function(GetConv2dLayer));

}  // namespace scnni