#pragma once
//#include "Eigen/Dense"
#include "EigenHelper.hpp"
#include <torch/torch.h>
#include <type_traits>
#include <vector>
//#include <variant>
namespace SciLib {
template <typename T> torch::Tensor mat_to_tensor(const T &mat) {
    using eleT = typename T::Scalar;
    if constexpr (std::is_same<eleT, float>::value) {
        return torch::from_blob(const_cast<eleT *>(mat.data()), {mat.rows(), mat.cols()}).clone();
    } else if constexpr (std::is_same<eleT, double>::value) {
        return torch::from_blob(const_cast<eleT *>(mat.data()), {mat.rows(), mat.cols()},
                                torch::kF64)
            .clone();
    } else if constexpr (std::is_same<eleT, int>::value) {
        return torch::from_blob(const_cast<eleT *>(mat.data()), {mat.rows(), mat.cols()},
                                torch::kInt32)
            .clone();
    } else if constexpr (std::is_same<eleT, bool>::value) {
        return torch::from_blob(const_cast<eleT *>(mat.data()), {mat.rows(), mat.cols()},
                                torch::kBool)
            .clone();
    }
    return torch::Tensor();
};

// template <typename T1, typename T2>
// torch::Tensor stdvec_to_tensor(const T1 &vec, std::initializer_list<T2> sizes) {
//     using eleT = typename T1::value_type;
//     if constexpr (std::is_same<eleT, int>::value)
//         return torch::from_blob(const_cast<eleT *>(vec.data()), sizes, torch::kInt).clone();
//     else if constexpr (std::is_same<eleT, float>::value)
//         return torch::from_blob(const_cast<eleT *>(vec.data()), sizes, torch::kF32).clone();
//     else if constexpr (std::is_same<eleT, double>::value)
//         return torch::from_blob(const_cast<eleT *>(vec.data()), sizes, torch::kF64).clone();
//     else if constexpr (std::is_same<eleT, bool>::value)
//         return torch::from_blob(const_cast<eleT *>(vec.data()), sizes, torch::kBool).clone();
//     return torch::Tensor();
// }

template <typename T> auto tensor_to_mat(const torch::Tensor &tensor) {
    if (tensor.dim() == 1) {
        TMat<T> res = TMap<T>(static_cast<T *>(tensor.data_ptr()), tensor.size(0), 1);
        return res;
    } else if (tensor.dim() == 2) {
        TMat<T> res = TMap<T>(static_cast<T *>(tensor.data_ptr()), tensor.size(0), tensor.size(1));
        return res;
    } else {
        return TMat<T>();
    }
};
// dim=0: each row convert to a tensor.
// sequeeze =true: now each tensor is 1d.
template <typename T>
std::vector<torch::Tensor> mat_to_vec_tensor(const T &mat, int dim = 0, bool squeeze = false) {
    using eleT = typename T::Scalar;

    std::vector<torch::Tensor> res;
    Eigen::Index r = mat.rows(), c = mat.cols();
    Eigen::Index n = (dim == 0) ? r : c, l = (dim == 0) ? c : r;
    // std::cout << "Size of mat: " << r << " " << c << std::endl;

    res.resize(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        TVec<eleT> v = (dim == 0) ? TVec<eleT>(mat.row(i).transpose()) : TVec<eleT>(mat.col(i));
        torch::Tensor t;
        if constexpr (std::is_same<eleT, float>::value) {
            // std::cout << v << std::endl;
            if (squeeze)
                t = torch::from_blob(v.data(), {l}).clone();
            else {
                if (dim == 0)
                    t = torch::from_blob(v.data(), {1, l}).clone();
                else
                    t = torch::from_blob(v.data(), {l, 1}).clone();
            }
        } else if constexpr (std::is_same<eleT, double>::value) {
            if (squeeze)
                t = torch::from_blob(v.data(), {c}, torch::kF64).clone();
            else {
                if (dim == 0)
                    t = torch::from_blob(v.data(), {1, l}, torch::kF64).clone();
                else
                    t = torch::from_blob(v.data(), {l, 1}, torch::kF64).clone();
            }
        } else {
            throw std::runtime_error(std::string{"Invalid type: "} +
                                     std::string{typeid(eleT).name()});
        }

        // std::cout << "t: " << t << std::endl;
        res[i] = t;
    }
    std::cout << "res[0]: " << res[0] << std::endl;

    return res;
};

void write_1d2d_tensor(const torch::Tensor &t, const std::string &filename);

// length L double column vector to 3D (1,1,L) or 2D CPU float tensor.
template <typename T>
torch::Tensor coldvec_to_3d_ftensor(const T &x, torch::Device device = torch::kCPU, int dim = 3) {
    long long int n = x.rows();
    return torch::from_blob(TVecf(x.template cast<float>()).data(), {1, 1, n})
        .clone()
        .to(device)
        .detach();
}

template <typename T> int vec_tensor_numel(const T &v) {
    int n = 0;
    for (const auto &e : v) {
        n += e.numel();
    }
    return n;
}
} // namespace SciLib