#pragma once

// This file can only be include once for each project. Use it in main.cpp
//#include "boost/json/src.hpp"
#include <boost/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace SciLib {
boost::json::object read_config(const std::string &config_file);

std::string_view json_string_to_string_view(const boost::json::string &s);

std::string json_string_to_string(const boost::json::string &s);

std::vector<int> json_int_array_to_vector(const boost::json::array &arr);

// template <typename T> T value_to(const boost::json::value &v) {
//    if constexpr (std::is_same<T, double>::value)
//        return v.get_double();
//    else if constexpr (std::is_same<T, float>::value)
//        return static_cast<float>(v.get_double());
//    else if constexpr (std::is_same<T, int>::value)
//        return static_cast<int>(v.get_int64());
//    else if constexpr (std::is_same<T, int64_t>::value)
//        return v.get_int64();
//    else
//        return T();
//};
template <typename T> std::vector<T> json_array_to_vector(const boost::json::array &arr) {
    std::vector<T> res;
    for (auto &ele : arr) {
        if constexpr (std::is_same<T, int>::value) {
            res.emplace_back(ele.as_int64());
        } else if constexpr (std::is_same<T, double>::value) {
            res.emplace_back(ele.as_double());
        } else if constexpr (std::is_same<T, std::string>::value) {
            res.emplace_back(json_string_to_string(ele.as_string()));
        } else {
            static_assert(!sizeof(T *), "Invalid type for json_array_to_vector<T>.");
        };
    }
    return res;
};

template <typename T>
std::vector<std::vector<T>> json_2d_array_to_2d_vector(const boost::json::array &arr) {
    std::vector<std::vector<T>> res;
    for (auto &ele : arr) {
        res.emplace_back(json_array_to_vector<T>(ele.as_array()));
    };
    return res;
};

void merge_json_in_place(boost::json::object &base, const boost::json::object &extend);

boost::json::object merge_json(const boost::json::object &base, const boost::json::object &extend);

void pretty_print(std::ostream &os, boost::json::value const &jv, std::string *indent = nullptr);

void write_json(const boost::json::object &obj, const std::string &filename);

} // namespace SciLib