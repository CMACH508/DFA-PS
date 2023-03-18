#pragma once
#include <filesystem>
#include <string>
#include <sstream>
#include <set>

namespace SciLib {
std::string add_tag_to_filename(const std::string &filename, const std::string &tag);

std::string string_repeat(const std::string &input, size_t num);

void display_progress_indicator(int n, int total);

void create_parent_directory(const std::string &filename);

std::string get_filename(const std::string &full_path);

void check_path_exists(const std::string &path);

template <typename... Args> std::string path_join(Args &&...args) {
    return (std::filesystem::path(args) / ...).string();
}

std::string absolute_path(const std::string &path);

void remove_files_except(const std::set<std::string> &files, const std::string &dir);

template <typename T> void write_vector(const std::vector<T> &v, const std::string &filename) {
    std::ofstream file(filename);
    for (auto &ele : v) {
        file << ele << std::endl;
    }
    file.close();
}

} // namespace SciLib