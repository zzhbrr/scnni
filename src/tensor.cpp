//
// Created by fss on 22-11-12.
//

#include "scnni/tensor.hpp"
#include <glog/logging.h>
#include <memory>

namespace scnni {

//!
Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  data_ = Eigen::Tensor<float, 3, Eigen::RowMajor>(rows, cols, channels);
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

//!
Tensor<float>::Tensor(const std::vector<uint32_t>& shapes) {
  SCNNI_ASSERT(shapes.size() == 3, "Find shapes.size() != 3");
  uint32_t channels = shapes.At(0);
  uint32_t rows = shapes.At(1);
  uint32_t cols = shapes.At(2);

  data_ = Eigen::Tensor<float, 3, Eigen::RowMajor>(channels, rows, cols);
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

/**
 * @description: 复制拷贝
 * @param {Tensor&} tensor
 * @return {*}
 */
Tensor<float>::Tensor(const Tensor& tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
}

Tensor<float>::Tensor(Tensor<float>&& tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shapes_ = tensor.raw_shapes_;
  }
}

Tensor<float>& Tensor<float>::operator=(Tensor<float>&& tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

Tensor<float>& Tensor<float>::operator=(const Tensor& tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

//!
uint32_t Tensor<float>::Rows() const {
	SCNNI_ASSERT(!this->data_.empty(), "data_ is empty"); 
  return this->data_.dimensions()[1];
}

//!
uint32_t Tensor<float>::Cols() const {
  SCNNI_ASSERT(!this->data_.empty(), "data_ is empty"); 
  return this->data_.dimensions()[2];
}

//!
uint32_t Tensor<float>::Channels() const {
  SCNNI_ASSERT(!this->data_.empty(), "data_ is empty");
  return this->data_.dimensions()[0];
}

//!
uint32_t Tensor<float>::Size() const {
  SCNNI_ASSERT(!this->data_.empty(), "data_ is empty");
  return this->data_.size();
}

//!
void Tensor<float>::Set_data(const Eigen::Tensor<float, 3, Eigen::RowMajor>& data) {
  CHECK(data.dimensions()[0] == this->data_.dimensions()[0])
    << data.dimensions()[0] << " != " << this->data_.dimensions()[0];
	CHECK(data.dimensions()[1] == this->data_.dimensions()[1])
		<< data.dimensions()[1] << " != " << this->data_.dimensions()[1];
  CHECK(data.dimensions()[2] == this->data_.dimensions()[2])
		<< data.dimensions()[2] << " != " << this->data_.dimensions()[2];
  this->data_ = data;
}

//!
bool Tensor<float>::Empty() const {
	return this->data_.empty();
}

//!
float Tensor<float>::Index(uint32_t offset) const {
  CHECK(offset < this->data_.size()) << "Tensor capacity is not enough!";
  return this->data_.at(offset);
}

//!
float & Tensor<float>::Index(uint32_t offset) {
  SCNNI_ASSERT(offset < this->data_.size()) << "Tensor capacity is not enough!";
  return this->data_.at(offset);
}

//!
std::vector<uint32_t> Tensor<float>::Shapes() const {
  SCNNI_ASSERT(!this->data_.empty(), "data_ is empty");
  return {this->Channels(), this->Rows(), this->Cols()};
}

//!
const std::vector<uint32_t>& Tensor<float>::Raw_shapes() const {
  SCNNI_ASSERT(!this->raw_shapes_.empty(),  "raw_shapes_ is empty");
  return this->raw_shapes_;
}

//!
Eigen::Tensor<float, 3, Eigen::RowMajor>& Tensor<float>::data() {
	return this->data_;
}

//!
const Eigen::Tensor<float, 3, Eigen::RowMajor>& Tensor<float>::data() const {
	return this->data_;
}

//!
/**
 * @description: 返回张量第channel通道中的数据
 * @param {uint32_t} channel
 * @return {*}
 */
Eigen::MatrixXf& Tensor<float>::Slice(uint32_t channel) {
  CHECK_LT(channel, this->Channels());
	Eigen::array<Eigen::DenseIndex, 3> offsets = { channel, 0, 0 }  //起点
	Eigen::array<Eigen::DenseIndex, 3> extends = { 1, this->Rows(), this->Cols() }  //扩充
	return this->data_.slice(offsets, extents)
}

//!
const Eigen::MatrixXf& Tensor<float>::Slice(uint32_t channel) const {
  CHECK_LT(channel, this->Channels());
  Eigen::array<Eigen::DenseIndex, 3> offsets = { channel, 0, 0 }
	Eigen::array<Eigen::DenseIndex, 3> extends = { 1, this->Rows(), this->Cols() }
	return this->data_.slice(offsets, extents)
}

//!
float Tensor<float>::At(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->Rows());
  CHECK_LT(col, this->Cols());
  CHECK_LT(channel, this->Channels());
  return this->data_(row, col, channel);
}

//!
float& Tensor<float>::At(uint32_t channel, uint32_t row, uint32_t col) {
  CHECK_LT(row, this->Rows());
  CHECK_LT(col, this->Cols());
  CHECK_LT(channel, this->Channels());
  return this->data_(row, col, channel);
}

void Tensor<float>::Padding(const std::vector<uint32_t>& pads, float padding_value) {
  SCNNI_ASSERT(!this->data_.empty(), "data_ is empty");
  CHECK_EQ(pads.size(), 4);
  uint32_t pad_rows1 = pads.At(0);  // up
  uint32_t pad_rows2 = pads.At(1);  // bottom
  uint32_t pad_cols1 = pads.At(2);  // left
  uint32_t pad_cols2 = pads.At(3);  // right

  Eigen::Tensor<float, 3, Eigen::RowMajor> new_data(this->data_.dimensions()[1] + pad_rows1 + pad_rows2,
                      this->data_.dimensions()[2] + pad_cols1 + pad_cols2,
                      this->data_.dimensions()[0]);
  new_data.fill(padding_value);

  new_data.subcube(pad_rows1, pad_cols1, 0, new_data.dimensions()[1] - pad_rows2 - 1,
                    new_data.dimensions()[2] - pad_cols2 - 1, new_data.dimensions()[0] - 1) =
      this->data_;
  this->data_ = std::move(new_data);
}

//!
void Tensor<float>::Fill(float value) {
  SCNNI_ASSERT(!this->data_.empty(), "data_ is empty");
  this->data_.setConstant(value);
}

//!
void Tensor<float>::Fill(const std::vector<float>& values) {
  SCNNI_ASSERT(!this->data_.empty(), "data_ is empty");
  const uint32_t total_elems = this->data_.size();
  CHECK_EQ(values.size(), total_elems);  //检查size相同
	//基本属性
  const uint32_t rows = this->Rows();
  const uint32_t cols = this->Cols();
  const uint32_t channels = this->Channels();
  const uint32_t planes = rows * cols;

  for(uint32_t i = 0; i < channels; ++i){
    auto& channel_data = this->data_.Slice(i);
		//指针起点为values.data() + i * planes, 大小为(this->Cols(), this->Rows())的vector片段
    const Eigen::MatrixXf& channel_data_t = Eigen::MatrixXf(values.data() + i * planes, this->Cols(), this->Rows());
    channel_data = channel_data_t.transpose();
  }
}

//!
void Tensor<float>::Ones() {
  SCNNI_ASSERT(!this->data_.empty(), "data_ is empty");
  this->Fill(1.f);
}

//!
void Tensor<float>::Rand() {
  SCNNI_ASSERT(!this->data_.empty(), "data_ is empty");
  this->data_.setRandom();
}

//!
void Tensor<float>::Show() {
  for(uint32_t i = 0; i < this->Channels(); ++i) {
    LOG(INFO) << "Channel: " << i;
    LOG(INFO) << "\n" << this->data_.Slice(i);
  }
}

//!
void Tensor<float>::ReRawshape(const std::vector<uint32_t>& shapes) {
  SCNNI_ASSERT(!this->data_.empty(), "data_ is empty");
  SCNNI_ASSERT(!shapes.empty(), "shapes is empty");
  
	const uint32_t origin_size = this->size();  // 原始大小
  uint32_t current_size = 1;  //reshape后大小
  for (uint32_t s : shapes) {
    current_size *= s;
  }
  SCNNI_ASSERT(shapes.size() <= 3, "shapes.size() > 3");
  SCNNI_ASSERT(current_size == origin_size, "Find current_size != origin_size");

  if (shapes.size() == 3) {
		this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
  } else if (shapes.size() == 2) {  //为保证data_为三维且列优先, 在0维度设1
    this->raw_shapes_ = {1, shapes.At(0), shapes.At(1)};
  } else {  //为保证data_为三维且列优先, 在0维度和1维度设1
    this->raw_shapes_ = {1, 1, shapes.At(0)};
  }
	Eigen::array<Eigen::DenseIndex, 3> dim = {this->raw_shapes_};
	this->data_ = data().reshape(dim);
}

void Tensor<float>::ReRawView(const std::vector<uint32_t>& shapes) {
  SCNNI_ASSERT(!this->data_.empty(), "data_ is empty");
  SCNNI_ASSERT(!shapes.empty(), "shapes is empty");
  
	const uint32_t origin_size = this->size();
  uint32_t current_size = 1;
  for (uint32_t s : shapes) {
    current_size *= s;
  }
  SCNNI_ASSERT(shapes.size() <= 3, "shapes.size() > 3");
  SCNNI_ASSERT(current_size == origin_size, "Find current_size != origin_size");
	
  std::vector<uint32_t> target_shapes;  // channel row col
  if (shapes.size() == 3) {
    target_shapes = {shapes.At(0), shapes.At(1), shapes.At(2)};
    this->raw_shapes_ = {shapes.At(0), shapes.At(1), shapes.At(2)};
  } else if (shapes.size() == 2) {
    target_shapes = {1, shapes.At(0), shapes.At(1)};
    this->raw_shapes_ = {shapes.At(0), shapes.At(1)};
  } else {
    target_shapes = {1, shapes.At(0), 1};
    this->raw_shapes_ = {shapes.At(0)};
  }
  this->ReView(target_shapes);
}

//!
void Tensor<float>::Flatten() {
  SCNNI_ASSERT(!this->data_.empty(), "data_ is empty");
  const uint32_t size = this->data_.size();
  this->ReRawshape({size});
}

void Tensor<float>::Transform(const std::function<float(float)>& filter) {
  SCNNI_ASSERT(!this->data_.empty(), "data_ is empty");
  this->data_.transform(filter);
}

std::shared_ptr<Tensor<float>> Tensor<float>::Clone() {
  return std::make_shared<Tensor>(*this);
}





void Tensor<float>::Review(const std::vector<uint32_t>& shapes) {
  SCNNI_ASSERT(!this->data_.empty(), "data_ is empty");
  const uint32_t target_channels = shapes.At(0);
  const uint32_t target_rows = shapes.At(1);
  const uint32_t target_cols = shapes.At(2);
  CHECK_EQ(this->data_.size(), target_channels * target_cols * target_rows);
  Eigen::Tensor<float, 3, Eigen::RowMajor> new_data(target_channels, target_rows, target_cols);

  const uint32_t plane_size = target_rows * target_cols;
  for (uint32_t c = 0; c < this->data_.dimensions()[0]; c++) {
    const Eigen::MatrixXf& channel = this->data_.Slice(c);
    for (uint32_t col_ = 0; col_ < this->data_.dimensions()[2]; ++col_) {
      const float* col_ptr = channel.colptr(col_);
      for (uint32_t r = 0; r < this->data_.dimensions()[1]; r++) {
        const uint32_t pos_index = c * data_.dimensions()[1] * data_.dimensions()[2] + r * data_.dimensions()[2] + col_;
        const uint32_t ch = pos_index / plane_size;
        const uint32_t row = (pos_index - ch * plane_size) / target_cols;
        const uint32_t col = (pos_index - ch * plane_size - row * target_cols);
        SCNNI_ASSERT(ch < new_data.dimensions()[0] && col < new_data.dimensions()[2] && row < new_data.dimensions()[1]);
        new_data.At(row, col, ch) = *(col_ptr + r);
      }
    }
  }
  this->data_ = std::move(new_data);
}

const float* Tensor<float>::raw_ptr() const {
  SCNNI_ASSERT(!this->data_.empty(), "data_ is empty");
  return this->data_.memptr();
}

std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows, uint32_t cols) {
  return std::make_shared<Tensor<float>>(channels, rows, cols);
}

std::shared_ptr<Tensor<float>> TensorCreate(const std::vector<uint32_t>& shapes) {
  SCNNI_ASSERT(shapes.size() == 3, "Find shapes.size() != 3");
  return TensorCreate(shapes.At(0), shapes.At(1), shapes.At(2));
}

bool TensorIsSame(const std::shared_ptr<Tensor<float>>& a,
                  const std::shared_ptr<Tensor<float>>& b) {
  SCNNI_ASSERT(a != nullptr, "a_ptr is nullptr");
  SCNNI_ASSERT(b != nullptr, "b_ptr is nullptr");
  if(a->Shapes() != b->Shapes()) {
    return false;
  }
  bool is_same = (a->data() == b->data());  //针对整数，不针对float
  return is_same;
}

//!
void TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
                      const std::shared_ptr<Tensor<float>>& tensor2,
                      const std::shared_ptr<Tensor<float>>& output_tensor) {
  SCNNI_ASSERT(tensor1 != nullptr, "When tensorElementAdd, tensor1 is nullptr");
	SCNNI_ASSERT(tensor2 != nullptr, "When tensorElementAdd, tensor2 is nullptr");
	SCNNI_ASSERT(output_tensor != nullptr, "When tensorElementAdd, output_tensor is nullptr");
  if(tensor1->Shapes() == tensor2->Shapes()){
    SCNNI_ASSERT(tensor1->Shapes() == output_tensor->Shapes());
    output_tensor->data_ = tensor1->data_ + tensor2->data_;
  }
	else{
		// broadcast
    SCNNI_ASSERT(tensor1->Channels() == tensor2->Channels(), "When tensorElementAdd, tensors shape are not adapting");
    const auto& [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
    SCNNI_ASSERT(output_tensor->Shapes() == input_tensor1->Shapes() &&
									output_tensor->Shapes() == input_tensor2->Shapes());
    output_tensor->data_ = input_tensor1->data_ + input_tensor2->data_;
  }
}

//！
std::shared_ptr<Tensor<float>> TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
																								const std::shared_ptr<Tensor<float>>& tensor2) {
  SCNNI_ASSERT(tensor1 != nullptr, "When tensorElementAdd, tensor1 is nullptr");
	SCNNI_ASSERT(tensor2 != nullptr, "When tensorElementAdd, tensor2 is nullptr");
  if (tensor1->Shapes() == tensor2->Shapes()) {
    std::shared_ptr<Tensor<float>> output_tensor = TensorCreate(tensor1->Shapes());
		output_tensor->data_ = tensor1->data_ + tensor2->data_;
    return output_tensor;
  }
	else {
    // broadcast
    SCNNI_ASSERT(tensor1->Channels() == tensor2->Channels(), "When tensorElementAdd, tensors shape are not adapting");
    const auto& [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
    SCNNI_ASSERT(input_tensor1->Shapes() == input_tensor2->Shapes());
    std::shared_ptr<Tensor<float>> output_tensor = TensorCreate(input_tensor1->Shapes());
		output_tensor->data_ = input_tensor1->data_ + input_tensor2->data_;
    return output_tensor;
  }
}

//!
/**
 * @description: 矩阵点乘
 */
void TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2,
    const std::shared_ptr<Tensor<float>>& output_tensor) {
	SCNNI_ASSERT(tensor1 != nullptr, "When tensorElementMultiply, tensor1 is nullptr");
	SCNNI_ASSERT(tensor2 != nullptr, "When tensorElementMultiply, tensor2 is nullptr");
	SCNNI_ASSERT(output_tensor != nullptr, "When tensorElementMultiply, output_tensor is nullptr");
  if(tensor1->Shapes() == tensor2->Shapes()){
    SCNNI_ASSERT(tensor1->Shapes() == output_tensor->Shapes());
    output_tensor->data_ = tensor1->data_ * tensor2->data_;
  }
	else{
		// broadcast
    SCNNI_ASSERT(tensor1->Channels() == tensor2->Channels(), "When tensorElementMultiply, tensors shape are not adapting");
    const auto& [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
    SCNNI_ASSERT(output_tensor->Shapes() == input_tensor1->Shapes() && output_tensor->Shapes() == input_tensor2->Shapes());
		output_tensor->data_ = input_tensor1->data_ * input_tensor2->data_;
  }
}

//!
std::shared_ptr<Tensor<float>> TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2) {
  SCNNI_ASSERT(tensor1 != nullptr, "When tensorElementMultiply, tensor1 is nullptr");
	SCNNI_ASSERT(tensor2 != nullptr, "When tensorElementMultiply, tensor2 is nullptr");
  if (tensor1->Shapes() == tensor2->Shapes()) {
    std::shared_ptr<Tensor<float>> output_tensor = TensorCreate(tensor1->Shapes());
    output_tensor->data_ = tensor1->data_ * tensor2->data_;
    return output_tensor;
  }
	else {
    // broadcast
    SCNNI_ASSERT(tensor1->Channels() == tensor2->Channels(), "When tensorElementMultiply, tensors shape are not adapting");
    const auto& [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
    SCNNI_ASSERT(input_tensor1->Shapes() == input_tensor2->Shapes());
    std::shared_ptr<Tensor<float>> output_tensor = TensorCreate(input_tensor1->Shapes());
    output_tensor->data_ = input_tensor1->data_ * input_tensor2->data_;
    return output_tensor;
  }
}

//!
std::tuple<std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<float>>> TensorBroadcast(const std::shared_ptr<Tensor<float>>& s1,
																																													const std::shared_ptr<Tensor<float>>& s2) {
  SCNNI_ASSERT(s1 != nullptr && s2 != nullptr);
	SCNNI_ASSERT(tensor1 != nullptr, "tensor1 is nullptr");
	SCNNI_ASSERT(tensor2 != nullptr, "tensor2 is nullptr");
  if(s1->Shapes() == s2->Shapes()){
    return {s1, s2};
  }
	else{
    SCNNI_ASSERT(s1->Channels() == s2->Channels());
    if (s2->Rows() == 1 && s2->Cols() == 1) {  //s2为Flatten
      std::shared_ptr<Tensor<float>> s2_ = TensorCreate(s2->Channels(), s1->Rows(), s1->Cols());
      SCNNI_ASSERT(s2->size() == s2->Channels());
      for (uint32_t c = 0; c < s2->Channels(); c++) {
        s2_->Slice(c).Fill(s2->Index(c));
      }
      return {s1, s2_};
    }
		else if (s1->Rows() == 1 && s1->Cols() == 1) {  //s1为Flatten
      std::shared_ptr<Tensor<float>> s1_ = TensorCreate(s1->Channels(), s2->Rows(), s2->Cols());
      SCNNI_ASSERT(s1->size() == s1->Channels());
      for (uint32_t c = 0; c < s1->Channels(); c++) {
        s1_->Slice(c).Fill(s1->Index(c));
      }
      return {s1_, s2};
    }
		else {
      LOG(FATAL) << "Broadcast shape is not adapting!";
      return {s1, s2};
    }
  }
}

std::shared_ptr<Tensor<float>> TensorPadding(
    const std::shared_ptr<Tensor<float>>& tensor,
    const std::vector<uint32_t>& pads, float padding_value) {
  SCNNI_ASSERT(tensor != nullptr && !tensor->empty());
  SCNNI_ASSERT(pads.size() == 4);
  uint32_t pad_rows1 = pads.At(0);  // up
  uint32_t pad_rows2 = pads.At(1);  // bottom
  uint32_t pad_cols1 = pads.At(2);  // left
  uint32_t pad_cols2 = pads.At(3);  // right

  std::shared_ptr<ftensor> output = std::make_shared<ftensor>(tensor->Channels(),
																															tensor->Rows() + pad_rows1 + pad_rows2,
																															tensor->Cols() + pad_cols1 + pad_cols2);
  if (padding_value != 0.f) output->Fill(padding_value);
  const uint32_t channels = tensor->Channels();
  for (uint32_t channel = 0; channel < channels; ++channel) {
    const Eigen::MatrixXf& in_channel = tensor->Slice(channel);
    Eigen::MatrixXf& output_channel = output->Slice(channel);
    const uint32_t in_channel_width = in_channel.dimensions()[2];
    const uint32_t in_channel_height = in_channel.dimensions()[1];

    for (uint32_t w = 0; w < in_channel_width; ++w) {
      float* output_channel_ptr = const_cast<float*>(output_channel.colptr(w + pad_cols1));
      const float* in_channel_ptr = in_channel.colptr(w);
      for (uint32_t h = 0; h < in_channel_height; ++h) {
        const float value = *(in_channel_ptr + h);
        *(output_channel_ptr + h + pad_rows1) = value;
      }
    }
  }
  return output;
}

}  // namespace scnni
