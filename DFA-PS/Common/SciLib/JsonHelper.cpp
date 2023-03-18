#include "stdafx.h"
#include "JsonHelper.hpp"

namespace SciLib {
boost::json::object read_config(const std::string &config_file) {
    boost::json::object res;
    try {
        std::ifstream file(config_file, std::ios::in);
        if (!file.is_open()) {
            fmt::print("Failed to open file {}.\n", config_file);
            throw std::runtime_error("Failed to open config file.");
        };
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string config_str = buffer.str();
        res = boost::json::parse(config_str).as_object();
    } catch (...) {
        fmt::print(fmt::fg(fmt::color::red), "Failed to read configure file {}.\n", config_file);
        std::terminate();
    }
    return res;
};

std::string_view json_string_to_string_view(const boost::json::string &s) {
    return {s.data(), s.size()};
};

std::string json_string_to_string(const boost::json::string &s) {
    std::string_view v{s.data(), s.size()};
    std::string res{v};
    return res;
};

std::vector<int> json_int_array_to_vector(const boost::json::array &arr) {
    std::vector<int> res;
    for (auto &ele : arr) {
        res.emplace_back(static_cast<int>(ele.as_int64()));
    }
    return res;
};

void merge_json_in_place(boost::json::object &base, const boost::json::object &extend) {
    /// std::cout << "base: " << base << std::endl << "extend: " << extend;

    for (const auto &ele : extend) {
        auto k = ele.key();
        auto v = ele.value();
        if (auto vo = v.if_object()) {
            // If value is an object, then try merge two objects.
            if (base.if_contains(k)) {
                // If base contains the same key, then merge.
                merge_json_in_place(base.at(k).as_object(), *vo);
            } else {
                // Else insert.
                base[k] = *vo;
            }
        } else {
            // TODO: currently if base[k] is an object, and extend[k] is a value. Then base[k] will
            // be overridden.
            base.insert_or_assign(k, v);
        }
    }
}

boost::json::object merge_json(const boost::json::object &base, const boost::json::object &extend) {
    boost::json::object res = base;
    merge_json_in_place(res, extend);
    return res;
}

void pretty_print(std::ostream &os, boost::json::value const &jv, std::string *indent) {
    namespace json = boost::json;

    std::string indent_;
    if (!indent)
        indent = &indent_;
    switch (jv.kind()) {
    case json::kind::object: {
        os << "{\n";
        indent->append(4, ' ');
        auto const &obj = jv.get_object();
        if (!obj.empty()) {
            auto it = obj.begin();
            for (;;) {
                os << *indent << json::serialize(it->key()) << " : ";
                pretty_print(os, it->value(), indent);
                if (++it == obj.end())
                    break;
                os << ",\n";
            }
        }
        os << "\n";
        indent->resize(indent->size() - 4);
        os << *indent << "}";
        break;
    }

    case json::kind::array: {
        os << "[\n";
        indent->append(4, ' ');
        auto const &arr = jv.get_array();
        if (!arr.empty()) {
            auto it = arr.begin();
            for (;;) {
                os << *indent;
                pretty_print(os, *it, indent);
                if (++it == arr.end())
                    break;
                os << ",\n";
            }
        }
        os << "\n";
        indent->resize(indent->size() - 4);
        os << *indent << "]";
        break;
    }

    case json::kind::string: {
        os << json::serialize(jv.get_string());
        break;
    }

    case json::kind::uint64:
        os << jv.get_uint64();
        break;

    case json::kind::int64:
        os << jv.get_int64();
        break;

    case json::kind::double_:
        os << jv.get_double();
        break;

    case json::kind::bool_:
        if (jv.get_bool())
            os << "true";
        else
            os << "false";
        break;

    case json::kind::null:
        os << "null";
        break;
    }

    if (indent->empty())
        os << "\n";
}

void write_json(const boost::json::object &obj, const std::string &filename) {
    std::ofstream file(filename);
    pretty_print(file, obj);
}
} // namespace SciLib