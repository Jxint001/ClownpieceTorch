// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// namespace py = pybind11;
#include "tensor.h"
#include "meta.h"
#include <cassert>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>
#include <iostream>



namespace at {

  /*
    utils for printing
  */
  // print vector int
  std::ostream& operator<<(std::ostream& os, const shape_t& shape) {
    os << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
      os << shape[i];
      if (i < shape.size() - 1)
        os << ", ";
    }
    os << ")";
    return os;
  }

  int print_tensor_data_recursive(std::ostream& os, const Tensor& tensor, int dim_index, int data_index, std::string prefix) {
    if (tensor.dim() == 0) {
      if (tensor.numel() == 0)
        os << "[]";
      else
        os << tensor.data_at(0);
      return 0;
    }
    os << "[";
    if (dim_index == tensor.dim() - 1 || tensor.dim() == 0) {
      for (int i = 0; i < tensor.size(dim_index); ++i) {
        os << tensor.data_at(data_index++);
        if (i < tensor.size(dim_index) - 1)
            os << ", ";
      }
    } else {

      for (int i = 0; i < tensor.size(dim_index); ++i) {
        if (i > 0)
          os << "\n" << prefix;
        data_index = print_tensor_data_recursive(os, tensor, dim_index + 1, data_index, prefix + " ");
        if (i < tensor.size(dim_index) - 1)
          os << ",";
      }
    }
    os << "]";
    return data_index;
  }

  std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(\n  shape=" << tensor.shape_ << ", strides=" << tensor.stride_ << "\n  data={\n";
    std::string prefix = "    ";
    os << prefix;
    print_tensor_data_recursive(os, tensor, 0, 0, prefix + " ");
    os << "\n  }\n)\n";
    return os;
  }


  /*
    Begin your implement here !
  */

  inline int logidx_to_phyidx(const Tensor& tensor, int logidx) {
    /* 可以优化 */
    int n = tensor.numel();
    if (logidx < 0)  logidx += n;

    if (logidx < 0 || logidx >= n) {
      throw std::runtime_error("in Tensor::data_at, index out of range");
    }

    int phyidx = tensor.offset_;
    for (int d = tensor.shape_.size() - 1; d >= 0; --d) {
      int dim = tensor.shape_[d];
      int coord = logidx % dim;

      phyidx += coord * tensor.stride_[d];
      logidx /= dim;
    }
    return phyidx;
  }

  inline void calc_numel_(const shape_t& shape, int& numel_) {
    numel_ = 1;
    for (const auto& d : shape)  numel_ *= d;
  }

  inline void fill_in_stride(const shape_t& shape, stride_t& stride) {
    /* 可以优化 */
    int n = shape.size();
    stride.resize(n);
    int s = 1;
    for (int i = n - 1; i >= 0; --i) {
      stride[i] = s;
      s *= shape[i];
    }
  }

  dtype& Tensor::data_at(int index) const {
    int phyidx = logidx_to_phyidx(*this, index);
   return storage_[phyidx];
  }

  int check_index_range(int idx, int siz, const std::string& name) {
    if (idx < 0)  idx += siz;
    if (idx < 0 || idx >= siz) {
      throw std::runtime_error(name + " index out of range");
      std::terminate();
    }
    return idx;
  }

  Tensor Tensor::apply_unary_op(std::function<dtype(dtype)> op) const {
    Tensor result = this->clone();
    int n = numel();
    for (int i = 0; i < n; ++i) {
        int idx = logidx_to_phyidx(*this, i);
        result.storage_[idx] = op(storage_[idx]);
    }
    return result;
  }

  Tensor Tensor::apply_binary_op(std::function<dtype(dtype, dtype)> op, const Tensor& rhs, const shape_t& sp) const {
    if (this->shape_ != rhs.shape_) {
        throw std::runtime_error("Tensors must have the same shape for binary operation");
    }

    Tensor result(sp);
    int n = numel();
    for (int i = 0; i < n; ++i) {
        int idx = logidx_to_phyidx(*this, i);
        result.data_at(i) = op(storage_[idx], rhs.data_at(i));
    }
    return result;
  }

  inline Tensor strict_2d_matmul(const Tensor& lhs, const Tensor& rhs) {
    /* Already checked dimension (2D) and alignment (m * n, n * k) */
    int m = lhs.size(0), n = lhs.size(1), k = rhs.size(1);
    Tensor result({m, k});
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < k; ++j) {
        dtype sum = 0;
        for (int p = 0; p < n; ++p) {
          sum += lhs.data_at(i * n + p) * rhs.data_at(p * k + j);
        }
        result.data_at(i * k + j) = sum;
      }
    }
    return result;
  }

  inline Tensor matrix_mult(const Tensor& a, const Tensor& b) {
    if (a.dim() < 2 || b.dim() < 2) { throw std::runtime_error("Invalid dim in matrix_mult"); }
    int dima = a.dim(), dimb = b.dim(), dimn = std::max(dima, dimb);
    int m = a.size(dima - 2), n = a.size(dima - 1), k = b.size(dimb - 1);
    if (n != b.size(dimb - 2)) { throw std::runtime_error("not aligned in matrix_mult"); }

    /* Broadcast leading dimensions */
    shape_t na_shape = a.shape_, nb_shape = b.shape_;
    while (na_shape.size() < nb_shape.size()) {
      na_shape.insert(na_shape.begin(), 1);
    }
    while (nb_shape.size() < na_shape.size()) {
      nb_shape.insert(nb_shape.begin(), 1);
    }
    for (int i = na_shape.size() - 3, j = nb_shape.size() - 3; i >= 0 && j >= 0; --i, --j) {
      if (nb_shape.size() < dima - 2 - i) {
        nb_shape.insert(nb_shape.begin(), na_shape[i]);
        ++j;
      } else {
        int ms = nb_shape[j], ts = na_shape[i];
        if (ms > ts && ts > 1) {
          throw std::runtime_error("Invalid broadcast in matrix_mult");
        }
        nb_shape[j] = std::max(ms, ts);
      }
    }

    na_shape = nb_shape;  na_shape[dimn - 1] = n, na_shape[dimn - 2] = m;
    shape_t res_shape = na_shape;  res_shape[dimn - 1] = k;
    Tensor result(res_shape, 0);

    int lead_dims = dimn - 2;
    int lead_numel = 1;
    for (int i = 0; i < lead_dims; ++i) {
      lead_numel *= res_shape[i];
    }

    for (int idx = 0; idx < lead_numel; ++idx) {
      /* Calculate coordinate */
      veci coords(lead_dims, 0);
      int temp = idx;
      for (int d = lead_dims - 1; d >= 0; --d) {
        coords[d] = temp % res_shape[d];
        temp /= res_shape[d];
      }

      int offset_a = a.offset_;
      for (int d = 0; d < lead_dims; ++d) {
        int coord = (d < dima - 2) ? 
                  (coords[d] < a.shape_[d] ? coords[d] : 0) : 0;
        offset_a += coord * a.stride_[d];
      }

      Tensor a_2d({m, n}, 
                  {a.stride_[dima - 2], a.stride_[dima - 1]},
                  offset_a, a.storage_);

      int offset_b = b.offset_;
      for (int d = 0; d < lead_dims; ++d) {
        int coord = (d < dimb - 2) ? 
                  (coords[d] < b.shape_[d] ? coords[d] : 0) : 0;
        offset_b += coord * b.stride_[d];
      }

      Tensor b_2d({n, k}, 
                  {b.stride_[dimb - 2], b.stride_[dimb - 1]},
                  offset_b, b.storage_);

      Tensor res_block = strict_2d_matmul(a_2d, b_2d);

      int offset_res = result.offset_;
      for (int d = 0; d < lead_dims; ++d) {
        offset_res += coords[d] * result.stride_[d];
      }      

      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
          result.data_at((idx * m * k) + i * k + j) = res_block.data_at(i * k + j);
        }
      }
    }
    
    return result;
  }

  /*
    constructors and assignments
  */
  Tensor::Tensor() :storage_(0) {}
  Tensor::Tensor(dtype value) :storage_(1, value){}
  Tensor::Tensor(const shape_t& shape) {
    shape_ = shape;
    fill_in_stride(shape_, stride_);
    calc_numel_(shape_, numel_);
    Storage temp(numel_);
    storage_ = temp;
  }
  Tensor::Tensor(const shape_t& shape, dtype value) {
    shape_ = shape;
    fill_in_stride(shape_, stride_);
    calc_numel_(shape_, numel_);
    Storage temp(numel_, value);
    storage_ = temp;
  }
  Tensor::Tensor(const shape_t& shape, std::function<dtype()> generator) {
    shape_ = shape;
    fill_in_stride(shape_, stride_);
    calc_numel_(shape_, numel_);
    Storage temp(numel_, generator);
    storage_ = temp;
  }
  Tensor::Tensor(const shape_t& shape, const vec<dtype>& data) {
    shape_ = shape;
    fill_in_stride(shape_, stride_);
    numel_ = data.size();
    Storage temp(data);
    storage_ = temp;
  }
  Tensor::Tensor(const shape_t& shape, const stride_t& stride, int offset, Storage storage) :shape_(shape), stride_(stride), storage_(storage), offset_(offset){
    calc_numel_(shape_, numel_);
  }

  Tensor::Tensor(const Tensor& other) :shape_(other.shape_), stride_(other.stride_), storage_(other.storage_), offset_(other.offset_), numel_(other.numel_) {}

  Tensor& Tensor::operator=(const Tensor& other) {
    shape_ = other.shape_;  stride_ = other.stride_;
    storage_ = other.storage_;
    offset_ = other.offset_;
    numel_ = other.numel_;
    return *this;
  }

  Tensor& Tensor::operator=(dtype value) {
    assert(numel() == 1);
    storage_[offset_] = value;
    return *this;
  }

  /* 
    destructor
  */
  Tensor::~Tensor() {}


  /*
    convert to dtype value
    only valid for singleton tensor
  */
  dtype Tensor::item() const {
    assert(numel() == 1);

    /* Physical index is offset + 0 here. */
    return storage_[offset_];
  }

  /*
    utils
  */

  int Tensor::numel() const {
    /* 可优化的 */
    if (dim() == 0) {
      if (storage_.size == 0) {
        return 0;
      }
      return 1;
    }
    int ret = 1;
    for (const auto& d : shape_)  ret *= d;
    return ret;
  }

  int Tensor::dim() const { return shape_.size(); }

  veci Tensor::size() const { return shape_; }

  int Tensor::size(int dim) const {
    int size = (int)shape_.size();
    int didx = check_index_range(dim, size, "Tensor::size(int dim)");
    return shape_[didx];
  }

  bool Tensor::is_contiguous() const {
    /* 可优化的, 考虑加 is_contiguous */
    int expected_s = 1, n = shape_.size();
    if (n > 0 && stride_[n - 1] != 1) {
      return false;
    }
    for (int i = n - 1; i > 0; --i) {
      expected_s *= shape_[i];
      if (stride_[i - 1] != expected_s) {
        return false;
      }
    }
    return true;
  }


  /*
    clone, make contiguous, copy_ and scatter
  */
  Tensor Tensor::clone() const {
    /* 可以优化 */
    Tensor temp(shape_);
    int n = numel();
    temp.storage_ = Storage(n);
    for (int i = 0; i < n; ++i) {
      temp.storage_[i] = data_at(i);
    }
    return temp;
  }

  Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
      return Tensor(shape_, stride_, offset_, storage_);
    } else {
      return clone();
    }
  }

  Tensor Tensor::copy_(const Tensor& other) const {
    if (shape_ != other.shape_) {
      throw std::runtime_error("shape mismatch in Tensor::copy(other)");
    }

    int n = numel();
    for (int i = 0; i < n; ++i) {
      data_at(i) = other.data_at(i);
    }
    return *this;
  }

  /* 正确性需要检查，先往下写 */
  Tensor Tensor::scatter_(int dim, const Tensor& index, const Tensor& src) const {
    dim = check_index_range(dim, shape_.size(), "Tensor::scatter_");
    
    int ndim = shape_.size();
    shape_t shape = shape_;
    shape.erase(shape.begin() + dim);
    
    for (int i = 0; i < ndim; i++) {
      if (i == dim) continue;
      if (shape_[i] != index.shape_[i] || shape_[i] != src.shape_[i]) {
        throw std::runtime_error("shape mismatch in Tensor::scatter_");
      }
    }

    Tensor transposed = transpose(dim, -1);
    int size_along_dim = transposed.size(-1);
    int n = index.numel();
    for (int i = 0; i < n; ++i) {
      int pos = index.data_at(i);
      dtype src_val = src.data_at(i);
      transposed.data_at(i * size_along_dim + pos) = src_val;
    }

    return *this;
  }


  /*
    subscriptor
  */
  Tensor Tensor::operator[](const vec<slice_t>& slices) const {
    Tensor temp(*this);
    int n = slices.size();
    for (int d = 0; d < n; ++d) {
      int begin = slices[d].first, end = slices[d].second, siz = shape_[d];
      begin = begin < 0 ? begin + siz : begin;
      end = end < 0 ? end + siz : end;

      int nbegin = std::max(0, begin), nend = std::min(siz, end);

      temp.offset_ += nbegin * stride_[d];
      temp.shape_[d] = nbegin >= nend ? 0 : nend - nbegin;
    }
    return temp;
  }

  Tensor Tensor::operator[](slice_t slice) const {
    Tensor temp(*this);
    int begin = slice.first, end = slice.second, siz = shape_[0];
    begin = begin < 0 ? begin + siz : begin;
    end = end < 0 ? end + siz : end;

    int nbegin = std::max(0, begin), nend = std::min(siz, end);
    temp.offset_ += nbegin * stride_[0];
    temp.shape_[0] = nbegin >= nend ? 0 : nend - nbegin;
    return temp;
  }

  Tensor Tensor::operator[](const veci& index) const {
    Tensor temp(*this);
    int n = dim();
    for (int d = 0; d < n; ++d) {
      int idx = check_index_range(index[d], shape_.empty() ? 0 : shape_[d], "Tensor::operator[](const veci& index)");
      temp.offset_ += idx * stride_[d];
    }
    for (int d = index.size() - 1; d >= 0; --d) {
      temp.shape_.erase(temp.shape_.begin() + d);
      temp.stride_.erase(temp.stride_.begin() + d);
    }
    return temp;
  }

  Tensor Tensor::operator[](int index) const {
    Tensor temp(*this);
    int idx = check_index_range(index, shape_.empty() ? 0 : shape_[0], "Tensor::operator[](int index)");
    temp.offset_ += idx * stride_[0];
    temp.shape_.erase(temp.shape_.begin());
    temp.stride_.erase(temp.stride_.begin());
    return temp;
  }

  /*
    operators
  */
  Tensor Tensor::operator-() const {
    return apply_unary_op([](dtype a) -> dtype {
      return -a;
    });
  }

  Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    auto [l, r] = lhs.broadcast(lhs, rhs);
    return l.apply_binary_op([](dtype a, dtype b) -> dtype {
      return a + b;
    }, r, l.shape_);
  }
  
  Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
    auto [l, r] = lhs.broadcast(lhs, rhs);
    return l.apply_binary_op([](dtype a, dtype b) -> dtype {
      return a - b;
    }, r, l.shape_);
  }

  Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    auto [l, r] = lhs.broadcast(lhs, rhs);
    return l.apply_binary_op([](dtype a, dtype b) -> dtype {
      return a * b;
    }, r, l.shape_);
  }

  Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
    auto [l, r] = lhs.broadcast(lhs, rhs);
    return l.apply_binary_op([](dtype a, dtype b) -> dtype {
      return a / b;
    }, r, l.shape_);
  }
  
  Tensor operator==(const Tensor& lhs, const Tensor& rhs) {
    auto [l, r] = lhs.broadcast(lhs, rhs);
    return l.apply_binary_op([](dtype a, dtype b) -> dtype {
      return a == b ? 1 : 0;
    }, r, l.shape_);
  }

  Tensor operator!=(const Tensor& lhs, const Tensor& rhs) {
    auto [l, r] = lhs.broadcast(lhs, rhs);
    return l.apply_binary_op([](dtype a, dtype b) -> dtype {
      return a != b ? 1 : 0;
    }, r, l.shape_);
  }
  Tensor operator<(const Tensor& lhs, const Tensor& rhs) {
    auto [l, r] = lhs.broadcast(lhs, rhs);
    return l.apply_binary_op([](dtype a, dtype b) -> dtype {
      return a < b ? 1 : 0;
    }, r, l.shape_);
  }

  Tensor operator<=(const Tensor& lhs, const Tensor& rhs) {
    auto [l, r] = lhs.broadcast(lhs, rhs);
    return l.apply_binary_op([](dtype a, dtype b) -> dtype {
      return a <= b ? 1 : 0;
    }, r, l.shape_);
  }

  Tensor operator>=(const Tensor& lhs, const Tensor& rhs) {
    auto [l, r] = lhs.broadcast(lhs, rhs);
    return l.apply_binary_op([](dtype a, dtype b) -> dtype {
      return a >= b ? 1 : 0;
    }, r, l.shape_);
  }

  Tensor operator>(const Tensor& lhs, const Tensor& rhs) {
    auto [l, r] = lhs.broadcast(lhs, rhs);
    return l.apply_binary_op([](dtype a, dtype b) -> dtype {
      return a > b ? 1 : 0;
    }, r, l.shape_);
  }

  /*
    matrix multiplication
  */
  Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    int dl = lhs.dim(), dr = rhs.dim();

    if (dl == 1 && dr == 1) {
        return sum(lhs * rhs, 0, false);
    }
    else if (dl == 1 && dr >= 2) {
      Tensor lhs_2d = lhs.unsqueeze(0);  // Change shape from (n,) to (1, n)
      Tensor temp = matrix_mult(lhs_2d, rhs);
      return temp.squeeze(0);
    }
    else if (dr == 1 && dl >= 2) {
      Tensor rhs_2d = (rhs.unsqueeze(0)).transpose(1, 0);
      Tensor temp = matrix_mult(lhs, rhs_2d);
      return temp.squeeze(1);
    } 
    else { /* dl >= 2 && dr >= 2*/
      return matrix_mult(lhs, rhs);
    }
  }

  Tensor operator^(const Tensor& lhs, const Tensor& rhs) { return matmul(lhs, rhs); }

  /*
    other mathematical operations
  */
  Tensor Tensor::sign() const {
    return apply_unary_op([](dtype a) -> dtype {
      if (a == 0)  return 0;
      if (a < 0) return -1;
      return 1;
    });
  }

  Tensor Tensor::abs() const {
    return apply_unary_op([](dtype a) -> dtype {
      return a < 0 ? -a : a;
    });
  }
  Tensor abs(const Tensor& tensor) {
    return tensor.abs();
  }

  Tensor Tensor::sin() const {
    return apply_unary_op([](dtype a) -> dtype {
      return std::sin(a);
    });
  }
  Tensor sin(const Tensor& tensor) {
    return tensor.sin();
  }

  Tensor Tensor::cos() const {
    return apply_unary_op([](dtype a) -> dtype {
      return std::cos(a);
    });
  }
  Tensor cos(const Tensor& tensor) {
    return tensor.cos();
  }
  Tensor Tensor::tanh() const {
    return apply_unary_op([](dtype a) -> dtype {
      return std::tanh(a);
    });
  }
  Tensor tanh(const Tensor& tensor) {
    return tensor.tanh();
  }

  Tensor Tensor::clamp(dtype min, dtype max) const {
    return apply_unary_op([min, max](dtype a) -> dtype {
      return std::min(max, std::max(min, a));
    });
  }

  Tensor clamp(const Tensor& tensor, dtype min, dtype max) {
    return tensor.clamp(min, max);
  }

  Tensor Tensor::log() const {
    return apply_unary_op([](dtype a) -> dtype {
      return std::log(a);
    });
  }

  Tensor log(const Tensor& tensor) {
    return tensor.log();
  }

  Tensor Tensor::exp() const {
    return apply_unary_op([](dtype a) -> dtype {
      return std::exp(a);
    });
  }

  Tensor exp(const Tensor& tensor) {
    return tensor.exp();
  }

  Tensor Tensor::pow(dtype exponent) const {
    return apply_unary_op([exponent](dtype a) -> dtype {
      return std::pow(a, exponent);
    });
  }

  Tensor pow(const Tensor& tensor, dtype exponent) {
    return tensor.pow(exponent);
  }

  Tensor Tensor::sqrt() const {
    return apply_unary_op([](dtype a) -> dtype {
      return std::sqrt(a);
    });
  }

  Tensor sqrt(const Tensor& tensor) {
    return tensor.sqrt();
  }

  Tensor Tensor::sum(int dim, bool keepdims) const {
    dim = check_index_range(dim, shape_.size(), "sum");
    int ndim = shape_.size();

    shape_t new_shape = shape_;
    if (keepdims) {
        new_shape[dim] = 1;
    } else {
        new_shape.erase(new_shape.begin() + dim);
    }
    
    Tensor result(new_shape, 0);
    int n = numel();
    
    std::vector<int> indices(ndim, 0);
    
    for (int i = 0; i < n; ++i) {
      int result_index = 0;
      int stride = 1;
      for (int d = ndim - 1; d >= 0; --d) {
          if (d == dim) {
            continue;
          }
          int coord = indices[d];
          result_index += coord * stride;
          
          /* stride in new shape */
          int dim_in_new_shape = (d > dim && !keepdims) ? d - 1 : d;
          if (dim_in_new_shape > 0) {
              stride *= new_shape[dim_in_new_shape];
          }
      }
      
      result.storage_[result_index] += data_at(i);
      
      /* Update indices */
      for (int d = shape_.size() - 1; d >= 0; --d) {
          if (++indices[d] < shape_[d]) {
              break;
          } else {
              indices[d] = 0;
          }
      }
    }
    
    return result;
  }

  Tensor sum(const Tensor& tensor, int dim, bool keepdims) {
    return tensor.sum(dim, keepdims);
  }

  std::pair<Tensor, Tensor> Tensor::max(int dim, bool keepdims) const {
    dim = check_index_range(dim, shape_.size(), "max");
    
    shape_t new_shape = shape_;
    if (keepdims) {
        new_shape[dim] = 1;
    } else {
        new_shape.erase(new_shape.begin() + dim);
    }
    
    Tensor max_values(new_shape);
    Tensor max_indices(new_shape, static_cast<dtype>(-1));
    
    int n = numel();
    int size_along_dim = shape_[dim];
    
    std::vector<int> indices(shape_.size(), 0);
    
    for (int i = 0; i < max_values.numel(); ++i) {
        max_values.storage_[i] = std::numeric_limits<dtype>::lowest();
    }
    
    for (int i = 0; i < n; ++i) {
        int result_index = 0;
        int stride = 1;
        for (int d = shape_.size() - 1; d >= 0; --d) {
            if (d == dim) continue; // skip reduced dimension
            
            if (keepdims) {
                int coord = (d < dim) ? indices[d] : indices[d+1];
                result_index += coord * stride;
            } else {
                int coord = (d < dim) ? indices[d] : indices[d];
                result_index += coord * stride;
            }
            stride *= (keepdims && d == dim+1) ? 1 : new_shape[d];
        }
        
        dtype value = data_at(i);
        if (value > max_values.storage_[result_index]) {
            max_values.storage_[result_index] = value;
            max_indices.storage_[result_index] = indices[dim];
        }
        
        for (int d = shape_.size() - 1; d >= 0; --d) {
            if (++indices[d] < shape_[d]) {
                break;
            } else {
                indices[d] = 0;
            }
        }
    }
    
    return {max_values, max_indices};
  }

  std::pair<Tensor, Tensor> max(const Tensor& tensor, int dim, bool keepdims) {
    return tensor.max(dim, keepdims);
  }

  Tensor Tensor::softmax(int dim) const {
    dim = check_index_range(dim, shape_.size(), "softmax");
    
    Tensor exp_tensor = this->exp();
  
    Tensor sum_exp = exp_tensor.sum(dim, true);
    
    return exp_tensor / sum_exp;
  }
  Tensor softmax(const Tensor& tensor, int dim) {
     return tensor.softmax(dim);
  }

  /*
    helper constructor
  */

  Tensor Tensor::ones_like() const {
    return Tensor(shape_, 1);
  }
  Tensor Tensor::zeros_like() const {
    return Tensor(shape_, 0);
  }
  Tensor Tensor::randn_like() const {
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    Tensor tensor(shape_);
    int n = tensor.numel();
    for (int i = 0; i < n; ++i) {
      tensor.data_at(i) = dist(gen);
    }

    return tensor;
  }
  Tensor Tensor::empty_like() const {
    return Tensor(shape_);
  }

  /*
    shape manipulation
  */

  Tensor Tensor::permute(veci p) const {
    int ndim = shape_.size();
    for (int& idx : p) {
      idx = check_index_range(idx, ndim, "permute");
    }

    shape_t new_shape(ndim);
    vec<bool> included(ndim, false);
    for (int i = 0; i < p.size(); ++i) {
      new_shape[i] = shape_[p[i]];
      included[p[i]] = true;
    }

    int j = 0;
    for (int i = p.size(); i < ndim; ++i) {
      while (included[j]) { ++j; }
      new_shape[i] = shape_[j];
      included[j] = true;
      p.push_back(j);
    }

    std::vector<int> new_strides(ndim);
  
    for (int i = 0; i < ndim; ++i) {
      int dim_idx = p[i];
      new_strides[i] = stride_[dim_idx];
      // if (!cont) {
      //   new_offset += stride_[dim_idx] * (shape_[dim_idx] - 1);
      // }
    }
    
    return Tensor(new_shape, new_strides, offset_, storage_);
  }

  Tensor Tensor::transpose(int dim1, int dim2) const {
    int ndim = shape_.size();
    dim1 = check_index_range(dim1, ndim, "Tensor::transpose");
    dim2 = check_index_range(dim2, ndim, "Tensor::transpose");
    veci p;
    for (int i = 0; i < ndim; ++i) { p.push_back(i); }
    std::swap(p[dim1], p[dim2]);
    return permute(p);
  }

  Tensor Tensor::reshape(const shape_t& purposed_shape, bool copy) const {
    bool cont = is_contiguous();
    if (cont && !copy) {
      return view(purposed_shape); }

    shape_t new_shape = purposed_shape;
    int n = numel();
    
    int missing_dim = -1;
    int known_size = 1;
    for (int i = 0; i < purposed_shape.size(); ++i) {
        if (purposed_shape[i] == -1) {
            missing_dim = i;
        } else {
            known_size *= purposed_shape[i];
        }
    }
    if (missing_dim != -1) {
        new_shape[missing_dim] = n / known_size;
    }

    int new_num_elements = 1;
    for (int dim : new_shape) {
        new_num_elements *= dim;
    }

    if (new_num_elements != n) {
        throw std::runtime_error("Incompatible shape for reshape");
    }

    Tensor new_tensor(new_shape);
    new_tensor.storage_ = Storage(n);
    fill_in_stride(new_shape, new_tensor.stride_);
    for (int i = 0; i < n; ++i) {
      new_tensor.storage_[i] = data_at(i);
    }
    return new_tensor;
  }

  Tensor Tensor::view(const shape_t& purposed_shape) const {
    if (!is_contiguous()) {
      throw std::runtime_error("non-contiguous in Tensor::view(const shape_t& purposed_shape)");
    }

    Tensor temp(*this);
    temp.shape_ = purposed_shape;

    /* 这一段可以封装 */
    int n = numel();
    int missing_dim = -1;
    int known_size = 1;
    for (int i = 0; i < purposed_shape.size(); ++i) {
      if (purposed_shape[i] == -1) {
        missing_dim = i;
      } else {
        known_size *= purposed_shape[i];
      }
    }
    if (missing_dim != -1) {
      temp.shape_[missing_dim] = n / known_size;
    }
    
    int new_num_elements = 1;
    for (int dim : temp.shape_) {
        new_num_elements *= dim;
    }
    if (new_num_elements != n) {
        throw std::runtime_error("Incompatible shape for reshape");
    }
    /**/
    fill_in_stride(temp.shape_, temp.stride_);
    return temp;
  }

  Tensor Tensor::narrow(int dim, int start, int length, bool copy) const {
    dim = check_index_range(dim, shape_.size(), "narrow");
    start = check_index_range(start, shape_[dim], "narrow start check");
    length = check_index_range(length, shape_[dim] - start + 1, "narrow length");
    if (!copy) {
      Tensor temp(*this);
      temp.shape_[dim] = length;
      temp.offset_ += start * stride_[dim];
      return temp;
    }

    shape_t new_shape = shape_;
    new_shape[dim] = length;
    Tensor result(new_shape);
    
    int slice_size = 1;
    for (int i = 0; i < shape_.size(); ++i) {
      if (i != dim) { slice_size *= shape_[i]; }
    }
    
    int total_elements = result.numel(), ndim = this->dim();
    std::vector<int> indices(ndim, 0);
    
    for (int i = 0; i < total_elements; ++i) {
        int src_pos = offset_;
        for (int d = 0; d < ndim; ++d) {
            int coord = (d == dim) ? (indices[d] + start) : indices[d];
            src_pos += coord * stride_[d];
        }
        
        result.storage_[i] = storage_[src_pos];
        
        /* Update indices. */
        for (int d = ndim - 1; d >= 0; --d) {
            if (d == dim && indices[d] + 1 >= length) continue;
            
            if (++indices[d] < new_shape[d]) break;
            indices[d] = 0;
        }
    }
    
    return result;
  }

  vec<Tensor> Tensor::chunk(int chunks, int dim) const {
    dim = check_index_range(dim, shape_.size(), "chunk");
    int size_along_dim = shape_[dim];
    if (chunks <= 0) {
        throw std::runtime_error("chunk: number of chunks must be positive");
    }
    if (size_along_dim == 0) {
        throw std::runtime_error("chunk: cannot split a dimension of size 0");
    }
    
    int chunk_size = (size_along_dim + chunks - 1) / chunks;
    vec<Tensor> result;
    int start = 0;
    
    for (int i = 0; i < chunks; ++i) {
        int end = std::min(start + chunk_size, size_along_dim);
        if (start >= size_along_dim) {
            break;
        }
        int length = end - start;
        result.push_back(narrow(dim, start, length, false));
        start = end;
    }
    
    return result;
  }

  vec<Tensor> Tensor::split(int dim, int split_size) const {
    dim = check_index_range(dim, shape_.size(), "split overload 1");
    int size_along_dim = shape_[dim];
    if (split_size <= 0) {
        throw std::runtime_error("split: split_size must be positive");
    }
    if (size_along_dim == 0) {
        throw std::runtime_error("split: cannot split a dimension of size 0");
    }
    
    int num_chunks = (size_along_dim + split_size - 1) / split_size;
    vec<Tensor> result;
    int start = 0;
    
    for (int i = 0; i < num_chunks; ++i) {
        int end = std::min(start + split_size, size_along_dim);
        int length = end - start;
        result.push_back(narrow(dim, start, length, false));
        start = end;
    }
    
    return result;
  }
  vec<Tensor> Tensor::split(int dim, veci split_sections) const {
    dim = check_index_range(dim, shape_.size(), "split");
    int size_along_dim = shape_[dim];
    if (size_along_dim == 0) {
        throw std::runtime_error("split: cannot split a dimension of size 0");
    }
    
    int total = 0;
    for (int sec : split_sections) {
        if (sec <= 0) {
            throw std::runtime_error("split: each section must be positive");
        }
        total += sec;
    }
    
    if (total != size_along_dim) {
        throw std::runtime_error("split: sum of split_sections does not match dimension size");
    }
    
    vec<Tensor> result;
    int start = 0;
    
    for (int sec : split_sections) {
        result.push_back(narrow(dim, start, sec, false));
        start += sec;
    }
    
    return result;
  }

  Tensor Tensor::stack(const vec<Tensor>& inputs, int dim) {
    if (inputs.empty()) {
        throw std::runtime_error("stack: input tensors can't be empty");
    }
    /* Every input must have the same shape. */
    shape_t shape0 = inputs[0].shape_;
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i].shape_ != shape0) {
            throw std::runtime_error("stack: all tensors must have the same shape");
        }
    }
    
    int ndim = inputs[0].dim();
    dim = check_index_range(dim, ndim + 1, "stack"); // [0, ndim]
    
    vec<Tensor> result;
    for (const auto& t : inputs) {
      result.push_back(t.unsqueeze(dim));
    }
    
    return cat(result, dim);
  }

  Tensor Tensor::cat(const vec<Tensor>& inputs, int dim) {
    if (inputs.empty()) {
        throw std::runtime_error("cat: input tensors can't be empty");
    }
    /* The size of dimension except dim should be same. */
    shape_t shape0 = inputs[0].shape_;
    int ndim = shape0.size();
    dim = check_index_range(dim, ndim, "cat");  // [0, ndim)
    
    int total_size = shape0[dim];
    for (size_t i = 1; i < inputs.size(); ++i) {
        const shape_t& shape_i = inputs[i].shape_;
        if (shape_i.size() != ndim) {
            throw std::runtime_error("cat: all tensors must have the same number of dimensions");
        }
        for (int d = 0; d < ndim; ++d) {
            if (d != dim && shape_i[d] != shape0[d]) {
                throw std::runtime_error("cat: all tensors must have the same shape in non-cat dimensions");
            }
        }
        total_size += shape_i[dim];
    }
    
    shape_t result_shape = shape0;
    result_shape[dim] = total_size;
    
    Tensor result(result_shape);
    int offset = 0;
    
    for (const auto& t : inputs) {
        int length = t.shape_[dim];
        /* slice and copy. */
        Tensor dest_slice = result.narrow(dim, offset, length, false);
        
        dest_slice.copy_(t);
        offset += length;
    }
    
    return result;
  }

  Tensor Tensor::squeeze(int dim) const {
    int ndim = shape_.size();
    if (ndim == 0) { return *this; }
    
    dim = check_index_range(dim, ndim, "squeeze");
    
    if (shape_[dim] != 1) {
      //std::cout << "dim is " << shape_[dim] << std::endl;
      throw std::runtime_error("squeeze: dimension size must be 1");
    }
    
    shape_t new_shape(shape_);
    stride_t new_stride(stride_);
    new_shape.erase(new_shape.begin() + dim);
    new_stride.erase(new_stride.begin() + dim);

    return Tensor(new_shape, new_stride, offset_, storage_);
  }

  Tensor Tensor::unsqueeze(int dim) const {
    int ndim = shape_.size();
    dim = check_index_range(dim, ndim + 1, "unsqueeze"); // [0, ndim]
    
    shape_t new_shape;
    stride_t new_stride;
    for (int d = 0; d < dim; ++d) {
        new_shape.push_back(shape_[d]);
        new_stride.push_back(stride_[d]);
    }
    
    new_shape.push_back(1);
    new_stride.push_back(dim < ndim ? stride_[dim] : 1);
    
    for (int d = dim; d < ndim; ++d) {
        new_shape.push_back(shape_[d]);
        new_stride.push_back(stride_[d]);
    }
    
    return Tensor(new_shape, new_stride, offset_, storage_);
  }

  Tensor Tensor::broadcast_to(const shape_t& shape) const {
    if (shape_.size() > shape.size()) {
      throw std::runtime_error("not broadcastable in Tensor::broadcast_to(const shape_t& shape)");
    }
    Tensor temp(*this);
    int n = shape.size();
    if (temp.stride_.size() == 0) {
      while (temp.stride_.size() < n) {temp.stride_.push_back(0); }
      temp.shape_ = shape;
      return temp;
    }

    while (temp.shape_.size() < n) {temp.shape_.insert(temp.shape_.cbegin(), 1); temp.stride_.insert(temp.stride_.cbegin(), 0); }
    
    for (int i = n - 1; i >= 0; --i) {
      if (temp.shape_[i] == shape[i]) { continue; }
      if (temp.shape_[i] > shape[i]) {
        throw std::runtime_error("not broadcastable in Tensor::broadcast_to(const shape_t& shape)");
      }
      if (temp.shape_[i] < shape[i]) {
        if (temp.shape_[i] == 1) {
          temp.shape_[i] = shape[i];
          temp.stride_[i] = 0;
          // temp.stride_.insert(temp.stride_.cbegin() + i, shape[i] - 1, 0);
          //std::cout << "inserted" << std::endl;
        } else {
          throw std::runtime_error("not broadcastable in Tensor::broadcast_to(const shape_t& shape)");
        }
      }
    }
    return temp;
  }

  std::pair<Tensor, Tensor> Tensor::broadcast(const Tensor& lhs, const Tensor& rhs) {
    /* Get common max shape */
    shape_t max_shape = lhs.shape_;
    int rsiz = rhs.shape_.size(), lsiz = lhs.shape_.size();
    for (int i = rsiz - 1, j = lsiz - 1; i >= 0; --i, --j) {
      if (max_shape.size() < rsiz - i) {
        max_shape.insert(max_shape.begin(), rhs.shape_[i]);
        ++j;
      } else {
        int ms = max_shape[j], ts = rhs.shape_[i];
        if (ms > ts && ts > 1) {
          throw std::runtime_error("Invald pair broadcast");
        }
        max_shape[j] = std::max(ms, ts);
      }
    }

    /* pair broadcast */
    return {Tensor(lhs.broadcast_to(max_shape)), Tensor(rhs.broadcast_to(max_shape))};
  }
  vec<Tensor> Tensor::broadcast(const vec<Tensor>& tensors) {
    shape_t max_shape = tensors[0].shape_;
    
    for (const auto& tensor : tensors) {
      int rsiz = tensor.shape_.size();
        for (int i = rsiz - 1, j = max_shape.size() - 1; i >= 0; --i, --j) {
            if (max_shape.size() < rsiz - i) {
                max_shape.insert(max_shape.begin(), tensor.shape_[i]);
                ++j;
            } else {
                int ms = max_shape[j], ts = tensor.shape_[i];
                if (ms > ts && ts > 1) {
                  throw std::runtime_error("Invalid vector broadcast");
                }
                max_shape[j] = std::max(ms, ts);
            }
        }
    }

    /* Broadcast every tensor. */
    vec<Tensor> v;
    for (const auto& tensor : tensors) {
      v.push_back(tensor.broadcast_to(max_shape));
    }
    return v;
  }



  /*
    helper constructors
  */
  Tensor to_singleton_tensor(dtype value, int dim) {
    shape_t v(dim, 1);
    return Tensor(v, value);
  }

  Tensor ones(const shape_t& shape) {
    return Tensor(shape, 1);
  }
  Tensor ones_like(const Tensor& ref) {
    return ref.ones_like();
  }

  Tensor zeros(const shape_t& shape) {
    return Tensor(shape, 0);
  }
  Tensor zeros_like(const Tensor& ref) {
    return Tensor(ref.zeros_like());
  }

  Tensor randn(const shape_t& shape) {
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    Tensor tensor(shape);
    int n = tensor.numel();
    for (int i = 0; i < n; ++i) {
      tensor.data_at(i) = dist(gen);
    }

    return tensor;
  }
  Tensor randn_like(const Tensor& ref) {
    return ref.randn_like();
  }

  Tensor empty(const shape_t& shape) {
    return Tensor(shape);
  }
  Tensor empty_like(const Tensor& ref) {
    return ref.empty_like();
  }

  Tensor arange(dtype start, dtype end, dtype step) {
    int siz = (end - start + step - 1) / step;
    shape_t shape(1, siz);
    vec<dtype> data;
    for (dtype i = start; i < end; i += step) {
      data.push_back(i);
    }
    return Tensor(shape, data);
  }

  Tensor range(dtype start, dtype end, dtype step) {
    int siz = (end - start) / step + 1;
     shape_t shape(1, siz);
    vec<dtype> data;
    for (dtype i = start; i <= end; i += step) {
      data.push_back(i);
    }
    return Tensor(shape, data);
  }

  Tensor linspace(dtype start, dtype end, int num_steps) {
    shape_t shape(1, num_steps);
    vec<dtype> data;
    dtype interval = (end - start) / (num_steps - 1), tmp = start;
    data.push_back(tmp);
    for (int i = 1; i < num_steps; ++i) {
      data.push_back(tmp = tmp + interval);
    }
    return Tensor(shape, data);
  }
  
  /*
    Week3 adds-on
  */
  Tensor Tensor::mean(int dim, bool keepdims) const {
    Tensor s = sum(dim, keepdims);
    int size_along_dim = shape_[dim];

    return s / static_cast<dtype>(size_along_dim);
  }

  Tensor Tensor::var(int dim, bool keepdims, bool unbiased) const {
    /* mean */
    Tensor m = mean(dim, true);
    
    Tensor diff = *this - m;
    diff = diff * diff;
    
    Tensor var_tensor = diff.mean(dim, keepdims);
    
    if (unbiased) {
        int n = size(dim);
        if (n <= 1) {
            return var_tensor * 0;
        }
        var_tensor = var_tensor * static_cast<dtype>(n) / static_cast<dtype>(n-1);
    }
    
    return var_tensor;
  }
};
