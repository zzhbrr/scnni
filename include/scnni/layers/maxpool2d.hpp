/*
 * @Author: zzh
 * @Description: 
 * @FilePath: /scnni/include/scnni/layers/maxpool2d.hpp
 */

#ifndef SCNNI_MAXPOOL_HPP_
#define SCNNI_MAXPOOL_HPP_

#include "scnni/layer.hpp"

namespace scnni {
class Maxpool2dLayer: public Layer {
    public:
        auto Forward(
            const std::vector<std::vector<std::shared_ptr<Tensor<float>>>> &input_blobs,
            std::vector<std::vector<std::shared_ptr<Tensor<float>>>> &output_blobs) const -> int override;
        ~Maxpool2dLayer() override = default;

        void SetCeilMode(bool ceil_mode);  //
        void SetDilation(const std::vector<int>& dilation);
        void SetKernelSize(const std::vector<int>& kernel_size);
        void SetPadding(const std::vector<int>& padding);
        // if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later
        void SetReturnIndices(bool return_indices);
        void SetStride(const std::vector<int>& stride);

    private:
        bool ceil_mode_;
        std::vector<int> dilation_;
        std::vector<int> kernel_size_;
        std::vector<int> padding_;
        bool return_indices_;
        std::vector<int> stride_;
};
} // namespace scnni


#endif
