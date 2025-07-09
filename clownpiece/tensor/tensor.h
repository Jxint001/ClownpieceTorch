#ifndef TENSOR_H
#define TENSOR_H 1

#include "meta.h"
#include <algorithm>
#include <iostream>
#include <functional>

namespace at {
class Storage {

 public:
  sptr<dtype[]> data;
  size_t size;

  Storage() : size(0) {
    data = nullptr;
  }
  Storage(size_t size) : size(size) {
    data = sptr<dtype []>(new dtype[size]);
  }
  Storage(size_t size, dtype value) : size(size) {
    data = sptr<dtype []>(new dtype[size]);
    std::fill(data.get(), data.get() + size, value);
  }
  Storage(size_t size, std::function<dtype()> generator) : size(size) {
    data = sptr<dtype []>(new dtype[size]);
    for (size_t i = 0; i < size; ++i) {
      data[i] = generator();
    }
  }
  Storage(const vec<dtype>& data) : size(data.size()) {
    this->data = sptr<dtype []>(new dtype[size]);
    std::copy(data.begin(), data.end(), this->data.get());
  }
  
  ~Storage() = default;

  dtype& operator[](size_t index) const {
    return data[index];
  }

  // note that storage is essentally a wrapper of some meta data, so move assignment is unnecessary
  // same argument applies for tensors.
  Storage(const Storage& other) = default;
  Storage& operator=(const Storage& other) = default;

  Storage clone() const {
    Storage cloned = Storage(this->size);
    std::copy(data.get(), data.get() + size, cloned.data.get());
    return cloned;
  }
};



class Tensor {

 protected:
  shape_t shape_;
  stride_t stride_;
  int offset_ = 0;
  Storage storage_;

 private:
  /*
    other helper member variables/functions
  */
  int numel_;
  friend int print_tensor_data_recursive(std::ostream& os, const Tensor& tensor, int dim_index, int flat_data_index, std::string prefix);
  friend inline int logidx_to_phyidx(const Tensor& tensor, int logidx);
  friend inline void calc_numel_(const shape_t& shape, int& numel_);
  friend inline void fill_in_stride(const shape_t& shape, stride_t& stride);
  friend inline int get_phyidx_in_scatter_(int dim /* 0 based */, const Tensor& index, const Tensor& self, int org_idx /* no need range check */);
  Tensor apply_unary_op(std::function<dtype(dtype)> op) const;
  Tensor apply_binary_op(std::function<dtype(dtype, dtype)> op, const Tensor& rhs, const shape_t& sp) const;
  friend inline Tensor matrix_mult(const Tensor& a, const Tensor& b);
  friend inline Tensor strict_2d_matmul(const Tensor& l, const Tensor& r);

 public:
  /*
    constructors and assignments
  */
  dtype& data_at(int index) const;
  Tensor();
  Tensor(dtype value);
  explicit Tensor(const shape_t& shape);
  explicit Tensor(const shape_t& shape, dtype value);
  explicit Tensor(const shape_t& shape, std::function<dtype()> generator);
  explicit Tensor(const shape_t& shape, const vec<dtype>& data);
  explicit Tensor(const shape_t& shape, const stride_t& stride, int offset, Storage storage);
  Tensor(const Tensor& other);

  Tensor& operator=(const Tensor& other);
  // only valid for singleton tensor
  Tensor& operator=(dtype value);
  /*
    destructor
  */
  ~Tensor();

  /*
    convert to dtype value
  */
  // only valid for singleton tensor
  dtype item() const;

  /*
    utils
  */
  int numel() const;
  int dim() const;
  veci size() const;
  int size(int dim) const;
  bool is_contiguous() const;
  const shape_t& get_shape() const { return shape_; }               // for graderlib
  dtype& get_data_at(int index) const { return data_at(index); }    // for graderlib
  const stride_t& get_stride() const { return stride_; }            // for graderlib
  int get_offset() const { return offset_; }                        // for graderlib
  const Storage& get_storage() const { return storage_; }                        // for graderlib
  /*
    clone, make contiguous and copy_from
  */
  Tensor clone() const;
  Tensor contiguous() const;
  Tensor copy_(const Tensor& other) const;
  Tensor scatter_(int dim, const Tensor& index, const Tensor& src) const;

  /*
    subscriptor
  */
  Tensor operator[](const vec<slice_t>& slice) const;

  Tensor operator[](slice_t slice) const;

  Tensor operator[](const veci& index) const;

  Tensor operator[](int index) const;


  /*
    operators  
  */
  Tensor operator-() const;

  friend Tensor operator+(const Tensor& lhs, const Tensor& rhs);
  // how is tensor +-*/ scalar handled?
  //  scalar is convert to singletion tensor automatically,
  //  then the operator applies broadcast rule.
  //  so it's not necessary to overload scalar + Tensor, etc.

  friend Tensor operator-(const Tensor& lhs, const Tensor& rhs);

  friend Tensor operator*(const Tensor& lhs, const Tensor& rhs);

  friend Tensor operator/(const Tensor& lhs, const Tensor& rhs);

  // Comparison
  friend Tensor operator==(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator!=(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator<(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator<=(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator>=(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator>(const Tensor& lhs, const Tensor& rhs);

  /*
    matrix multiplication
  */
  friend Tensor matmul(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator^(const Tensor& lhs, const Tensor& rhs);
  
  /*
    other mathematical operations
  */
  Tensor sign() const;
  
  Tensor abs() const;

  Tensor sin() const;

  Tensor cos() const;

  Tensor tanh() const;

  Tensor clamp(dtype min, dtype max) const;

  Tensor log() const;

  Tensor exp() const;
  Tensor pow(dtype exponent) const;

  Tensor sqrt() const;

  Tensor sum(int dim, bool keepdims=false) const;

  std::pair<Tensor, Tensor> max(int dim, bool keepdims=false) const;

  Tensor softmax(int dim) const;

  /*
    helper constructor
  */

  Tensor ones_like() const;
  Tensor zeros_like() const;
  Tensor randn_like() const;
  Tensor empty_like() const;


  /*
    printing
  */
 friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

  /*
    indexing, slicing, joining, mutating ops

    const means the metadata of the tensor is not changed, but the data may be changed (by shared underyling storage)
  */
  Tensor permute(veci p) const;
  Tensor transpose(int dim1, int dim2) const;

  Tensor reshape(const shape_t& purposed_shape, bool copy=false) const;
  Tensor view(const shape_t& purposed_shape) const;

  Tensor narrow(int dim, int start, int length, bool copy=false) const;
  vec<Tensor> chunk(int chunks, int dim) const;

  vec<Tensor> split(int dim, int split_size) const;
  vec<Tensor> split(int dim, veci split_sections) const;

  static Tensor stack(const vec<Tensor>& inputs, int dim=0);
  static Tensor cat(const vec<Tensor>& inputs, int dim=0);

  Tensor squeeze(int dim) const;
  Tensor unsqueeze(int dim) const;

  Tensor broadcast_to(const shape_t& shape) const;
  static std::pair<Tensor, Tensor> broadcast(const Tensor& lhs, const Tensor& rhs);
  static vec<Tensor> broadcast(const vec<Tensor>& tensors);


  /*
    Week3 add-ons
  */
  Tensor mean(int dim, bool keepdims=false) const;
  Tensor var(int dim, bool keepdims=false, bool unbiased=true) const;
};


/*
  non-member version operations on Tensor.
*/

Tensor abs(const Tensor& tensor);

Tensor sin(const Tensor& tensor);

Tensor cos(const Tensor& tensor);

Tensor tanh(const Tensor& tensor);

Tensor clamp(const Tensor& tensor, dtype min, dtype max);

Tensor log(const Tensor& tensor);

Tensor exp(const Tensor& tensor);

Tensor pow(const Tensor& tensor, dtype exponent);

Tensor sqrt(const Tensor& tensor);

Tensor sum(const Tensor& tensor, int dim, bool keepdims=false);

std::pair<Tensor, Tensor> max(const Tensor& tensor, int dim, bool keepdims=false);

Tensor softmax(const Tensor& tensor, int dim);

/*
  helper constructors
*/
Tensor to_singleton_tensor(dtype value, int dim);

Tensor ones(const shape_t& shape);
Tensor ones_like(const Tensor& ref);

Tensor zeros(const shape_t& shape);
Tensor zeros_like(const Tensor& ref);

Tensor randn(const shape_t& shape);
Tensor randn_like(const Tensor& ref);

Tensor empty(const shape_t& shape);
Tensor empty_like(const Tensor& ref);

Tensor arange(dtype start, dtype end, dtype step);

Tensor range(dtype start, dtype end, dtype step);

Tensor linspace(dtype start, dtype end, int num_steps);

/*
  formal declaration of friend function
*/

Tensor matmul(const Tensor& lhs, const Tensor& rhs);

};
#endif
