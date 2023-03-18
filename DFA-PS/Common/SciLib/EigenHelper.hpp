#pragma once

#include <Eigen/Dense>

#include "boost/tokenizer.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <algorithm>
#include <numeric>

// Type definition
template <typename T = double>
using TMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using TMati = TMat<int>;
using TMatb = TMat<bool>;
using TMatd = TMat<double>;
using TMatf = TMat<float>;

template <typename T = double>
using TArr = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T = double> using TVec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T = double> using TVecA = Eigen::Array<T, Eigen::Dynamic, 1>;
template <typename T = double> using TRVec = Eigen::Matrix<T, 1, Eigen::Dynamic>;
template <typename T = double> using TRVecA = Eigen::Array<T, 1, Eigen::Dynamic>;

using TVeci = TVec<int>;
using TVecb = TVec<bool>;
using TVecd = TVec<double>;
using TVecf = TVec<float>;

template <typename T = double> using TMap = Eigen::Map<TMat<T>>;
using TMapd = TMap<double>;
using TMapf = TMap<float>;

namespace SciLib {
template <typename T> inline T inner_product(const TVec<T> &x, const TMat<T> &m) {
    return x.transpose() * m * x;
};

// decide shape of file.
template <typename T = double> auto decide_tab_file_shape(const std::string &filename) {
    std::ifstream file(filename);
    int rows = 0, cols = 0;
    std::string str;
    std::getline(file, str);
    std::stringstream ss(str);
    double tmp;
    while (ss >> tmp) {
        ++cols;
    };
    ++rows;
    while (std::getline(file, str)) {
        if (str.size() > 0)
            ++rows;
    }

    return std::make_pair(rows, cols);
};

// Load tab delimited file. Shape is automatically detected.
template <typename T = double> TMat<T> load_tab_file(const std::string &filename) {
    std::ifstream ifile(filename);
    if (!ifile.is_open()) {
        throw std::range_error("File " + filename + " not exists.");
    };
    auto [rows, cols] = decide_tab_file_shape<T>(filename);
    std::ifstream file(filename);
    TMat<T> res(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            file >> res.coeffRef(i, j);
        }
    }
    return res;
};

// decide shape of csv file.
template <typename T = double> auto decide_csv_file_shape(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.good()) {
        fmt::print(fmt::fg(fmt::color::red), "File {} doesn't exists.\n",filename);
        std::abort();
    }
    size_t rows = 0, cols = 0;
    std::string line;

    std::getline(file, line);

    // const std::string csv_delimiter = ",";
    boost::tokenizer<boost::escaped_list_separator<char>> tok(line);
    std::vector<std::string> line_tokens(tok.begin(), tok.end());

    cols = line_tokens.size();

    ++rows;
    while (std::getline(file, line)) {
        if (line.size() > 0)
            ++rows;
    }

    return std::make_pair(rows, cols);
};

// Load csv file. Shape is automatically detected.
template <typename T = double>
TMat<T> load_csv_file(const std::string &filename, int skip_row = 0, int skip_column = 0) {
    auto [rows, cols] = decide_csv_file_shape<T>(filename);
    std::ifstream file(filename);
    TMat<T> res(rows - skip_row, cols - skip_column);

    const std::string csv_delimiter = ",";

    std::string line;

    int k = 0, km = 0;
    while (std::getline(file, line)) {
        std::vector<std::string> line_tokens;
        ++k;
        if (line.size() > 0) {
            boost::tokenizer<boost::escaped_list_separator<char>> tok(line);
            std::vector<std::string> line_tokens(tok.begin(), tok.end());
            if (line_tokens.size() != cols) {
                throw std::runtime_error("Columns at row " + std::to_string(k) + " is less than " +
                                         std::to_string(cols) + ".");
            };

            if (k <= skip_row) {
                continue;
            } else {
                for (int i = skip_column; i < line_tokens.size(); ++i) {
                    T val;
                    try {
                        val = std::stod(line_tokens[i]);
                    } catch (...) {
                        throw std::runtime_error("Failed to parse number at Row " +
                                                 std::to_string(k) + " Col " + std::to_string(i) +
                                                 ", string is: " + line_tokens[i]);
                    }
                    res.coeffRef(km, i - skip_column) = val;
                };
                ++km;
            };
        };
    }

    return res;
};

template <typename T = double>
TMat<T> readmatrix(const std::string &filename, int skip_row = 0, int skip_column = 0) {
    auto p = std::filesystem::path(filename);
    // std::ifstream ifile(filename);
    if (!std::filesystem::exists(p)) {
        throw std::range_error("File " + filename + " not exists.");
    };
    // Load data.
    if (p.extension() == std::filesystem::path(".txt")) {
        return load_tab_file<T>(filename);
    } else if (p.extension() == std::filesystem::path(".csv")) {
        return load_csv_file<T>(filename, skip_row, skip_column);
    } else {
        throw std::runtime_error("Unknown file type " + p.extension().string());
    }
}

template <typename T>
void write_mat(const T &mat, const std::string &filename, const std::string &delimiter = ",") {
    std::ofstream file(filename);
    auto m = mat.rows(), n = mat.cols();
    for (Eigen::Index i = 0; i < m - 1; ++i) {
        for (Eigen::Index j = 0; j < n - 1; ++j) {
            file << mat.coeff(i, j) << delimiter;
        }
        file << mat.coeff(i, n - 1) << std::endl;
    }
    for (Eigen::Index j = 0; j < n - 1; ++j) {
        file << mat.coeff(m - 1, j) << delimiter;
    }
    file << mat.coeff(m - 1, n - 1);
};

//(rows count that contains nan)
template <typename T> int check_nan(const T &mat) {
    int nan_rows = 0;
    for (auto &row : mat.rowwise()) {
        nan_rows += (row.array().isNaN().sum() > 0);
    }
    return nan_rows;
};

// fill na in matrix.
template <typename T, typename Tv> T fillna(T &&m, Tv v) {
    using eleT = typename std::remove_reference_t<T>::Scalar;
    eleT val = static_cast<eleT>(v);
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            if (std::isnan(m.coeff(i, j))) {
                m.coeffRef(i, j) = val;
            }
        }
    }
    return m;
};

template <typename T> bool contain_nan(const T &m) { return m.array().isNaN().sum(); }

// Missing data between the first non-missing and the last non-missing is filled by previous valid
// rows.
template <typename T> T forward_fillna(T &&mat) {
    using eleT = typename std::remove_reference_t<T>::Scalar;
    // eleT val = static_cast<eleT>(v);
    int m = static_cast<int>(mat.rows()), n = static_cast<int>(mat.cols());
    for (int j = 0; j < n; ++j) {
        // Find first non-missing index.
        int start = 0, end = m - 1;
        for (; start < m; ++start) {
            if (!std::isnan(mat.coeff(start, j))) {
                break;
            }
        }
        // Find the last non-missing index.
        for (; end > 0; --end) {
            if (!std::isnan(mat.coeff(end, j))) {
                break;
            }
        }
        // Fill mask
        if (start < end) {
            for (int i = start; i <= end; ++i) {
                if (std::isnan(mat.coeff(i, j))) {
                    mat.coeffRef(i, j) = mat.coeffRef(i - 1, j);
                }
            }
        }
    }
    return mat;
};

template <typename T> inline void writematrix(T &&m, const std::string &filename) {
    Eigen::IOFormat mat_format(
        std::numeric_limits<typename std::remove_reference_t<T>::Scalar>::max_digits10,
        Eigen::DontAlignCols, ", ");
    std::ofstream(filename) << m.format(mat_format);
};

// Given a vector, make elements less than nth to be zero.
template <typename VT, typename Comp>
static std::pair<VT, std::vector<int>> keep_topk(const VT &_v, int topk, const Comp &comp) {
    using value_type = decltype(*(_v.begin()));
    VT v = _v;

    int N = v.size();
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::nth_element(indices.begin(), indices.begin() + topk, indices.end(),
                     [&](const auto &l, const auto &r) { return comp(v[l], v[r]); });

    std::fill(v.begin(), v.end(), 0);
    for (int i = 0; i < topk; ++i) {
        v[indices[i]] = _v[indices[i]];
    }

    std::vector<int> subindices(indices.begin(), indices.begin() + topk);
    return {v, subindices};
}
template <typename VT> static std::pair<VT, std::vector<int>> keep_topk(const VT &_v, int topk) {
    using value_type = decltype(*(_v.begin()));
    return keep_topk(_v, topk, std::less<value_type>{});
}
} // namespace SciLib