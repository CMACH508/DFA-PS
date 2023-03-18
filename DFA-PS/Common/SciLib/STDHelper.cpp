#include "stdafx.h"
#include "STDHelper.hpp"

namespace SciLib {
std::string add_tag_to_filename(const std::string &filename, const std::string &tag) {
    std::string res;
    auto p = std::filesystem::path(filename);
    return (p.parent_path() /
            std::filesystem::path(p.stem().string() + tag + p.extension().string()))
        .string();
};

std::string string_repeat(const std::string &input, size_t num) {
    std::ostringstream os;
    std::fill_n(std::ostream_iterator<std::string>(os), num, input);
    return os.str();
};

void display_progress_indicator(int n, int total) {
    double f = static_cast<double>(n) / total * 100;

    std::cout << std::fixed << std::setprecision(2) << f << "% : "
              << "[" << string_repeat("=", floor(f)) << string_repeat(" ", 100 - floor(f)) << "]"
              << std::endl;
};

void create_parent_directory(const std::string &filename) {
    std::string res;
    auto p = std::filesystem::path(filename);

    std::filesystem::create_directory(p.parent_path());
};
std::string get_filename(const std::string &full_path) {
    auto p = std::filesystem::path(full_path);

    return p.filename().string();
};

void check_path_exists(const std::string &path) {
    if (!std::filesystem::exists(path)) {
        std::cout << "File " + path + " not exists." << std::endl;
        throw std::range_error("File " + path + " not exists.");
    };
};

std::string absolute_path(const std::string &path) {
    return std::filesystem::absolute(std::filesystem::path(path)).string();
}

void remove_files_except(const std::set<std::string> &files, const std::string &dir) {
    auto p = std::filesystem::path(dir);
    for (const auto &entry : std::filesystem::directory_iterator(p)) {
        if (files.count(entry.path().filename().string()) == 0) {
            std::filesystem::remove_all(entry.path());
        }
    }
}

} // namespace SciLib