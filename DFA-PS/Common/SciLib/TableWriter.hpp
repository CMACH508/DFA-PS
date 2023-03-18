#pragma once

#include <fmt/format.h>
#include <variant>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <array>
#include <unordered_set>

namespace SciLib {

/* Usage:
TableWriter tb("A", "B", "C", "F");
tb.set_file("output.csv");
tb.set_col_type(1, "B", "F");
tb.set_col_formatter("{: ^4}", "{: ^4}", "B", "F");
tb.add_sep_after("A" ,"C");
tb.set_monitor_col("A", true);

tb.new_row();
tb["A"] = 3.0;
tb["B"] = 5;
tb["C"] = 12.5;
tb["F"] = 15;
bool tb.update_best_row();  //Test if the current row is best.

tb.print_header();
tb.write_header();
tb.print_row();
tb.write_row();

int get_best_row();

 */
class TableWriter {
  public:
    using CellT = std::variant<std::monostate, double, int, std::string>;

    TableWriter() {}
    template <typename... Args> TableWriter(const Args &...args) { add_cols(args...); }

    void add_col(const std::string &name, int type) {
        col_type.emplace_back(type);
        col_ni_map[name] = col_in_map.size();
        col_in_map.emplace_back(name);
        col_formatter.emplace_back(type_formatter[type]);
        col_formatter_alt.emplace_back(type_formatter_alt[type]);

        data.emplace_back();
    }
    // Default to be double cols.
    template <typename... Cols> void add_cols(const std::string &col, const Cols &...args) {
        add_col(col, 0);
        if constexpr (sizeof...(args) > 0) {
            add_cols(args...);
        }
    }

    void new_row() {
        for (auto &col : data) {
            col.emplace_back();
        }
    }
    std::string format_cell(int col, const CellT &cell) {
        // std::cout << col_formatter[col] << std::endl;
        // std::cout << col_formatter_alt[col] << std::endl;
        // std::cout << cell.index() << std::endl;

        // std::cout << "yes" << std::endl;
        if (cell.index() == 0) {
            return fmt::format(col_formatter_alt[col], "");
        } else if (cell.index() == 1) {
            return fmt::format(col_formatter[col], std::get<double>(cell));
        } else if (cell.index() == 2) {
            return fmt::format(col_formatter[col], std::get<int>(cell));
        } else if (cell.index() == 3) {
            return fmt::format(col_formatter[col], std::get<std::string>(cell));
        } else {
            return "";
        }
    }

    std::string format_row(int r, const std::string &col_sep, const std::string &group_sep = "") {
        std::string res;
        const int n_col = data.size();
        for (int i = 0; i < n_col; ++i) {
            const auto &cell = data[i][r];
            res += format_cell(i, cell);
            if ((i < n_col - 1) && (!col_sep.empty()))
                res += col_sep;
            if ((sep_set.count(i) > 0) && (!group_sep.empty()))
                res += group_sep;
        }

        return res;
    }
    std::string format_row(const std::string &col_sep, const std::string &group_sep = "") {
        return format_row(data[0].size() - 1, col_sep, group_sep);
    }

    void print_row(int r) { std::cout << format_row(r, " ", "| ") << std::endl; }
    void print_row() { print_row(data[0].size() - 1); }
    void print_header() {
        const int N = data.size();
        for (int i = 0; i < N; ++i) {
            const auto &formatter = col_formatter_alt[i];
            const auto &colname = col_in_map[i];
            // std::cout << colname << std::endl;
            std::cout << fmt::format(formatter, colname);

            if (i < N - 1)
                std::cout << " ";
            if (sep_set.count(i) > 0)
                std::cout << "| ";
        }
        std::cout << std::endl;
    }
    std::string format_header() {
        std::string res;
        const int N = col_in_map.size();
        for (int i = 0; i < N; ++i) {
            const auto &colname = col_in_map[i];
            res += colname;
            if (i < N - 1)
                res += ",";
        }
        return res;
    }
    void write_header() { (*file) << format_header() << std::endl; }
    // Write to file.
    void write_header(const std::string &filename) {
        std::ofstream(filename) << format_header() << std::endl;
    }

    void set_file(const std::string &filename) { file = std::make_unique<std::ofstream>(filename); }
    void write_row() { (*file) << format_row(data[0].size() - 1, ", ") << std::endl; }
    void write_row(const std::string &filename) {
        std::ofstream(filename) << format_row(", ") << std::endl;
    }
    void write(const std::string &filename) {
        std::ofstream file(filename);
        file << format_header() << std::endl;
        for (int i = 0; i < data[0].size(); ++i) {
            file << format_row(i, ", ") << std::endl;
        }
    }
    void write_row_with_header(int r, const std::string &filename) {
        std::ofstream file(filename);
        file << format_header() << std::endl;
        file << format_row(r, ", ") << std::endl;
    }

    //
    void set_monitor_col(const std::string &colname, bool min_v) {
        if (col_ni_map.count(colname) == 0) {
            std::cerr << fmt::format("This table doesn't have a row {}.\n", colname);
            return;
        }

        this->min_v = min_v;
        this->monitor_col = colname;
        if (min_v) {
            best_value = 1e99;
        } else {
            best_value = 1e-99;
        }
    }

    bool update_best_row() {
        bool res = false;
        if (!monitor_col.empty()) {
            auto &ele = this->operator[](monitor_col);
            std::visit(
                [this, &res](auto &&e) {
                    using T = std::decay_t<decltype(e)>;
                    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, double>) {
                        if (min_v) {
                            if (e < best_value) {
                                best_value = e;
                                best_row = data[0].size() - 1;
                                res = true;
                            }
                        } else {
                            if (e > best_value) {
                                best_value = e;
                                best_row = data[0].size() - 1;
                                res = true;
                            }
                        }
                    }
                },
                ele);
        }
        return res;
    }
    int get_best_row() { return best_row; }
    // As it return reference to variant, then one must assure that types are matching.
    // Especially pay attention to double and int type.
    CellT &operator[](const std::string &col) {
        if (col_ni_map.count(col) == 0) {
            std::cerr << "Invalid key " << col << std::endl;
            std::terminate();
        }
        return data[col_ni_map[col]].back();
    }

    template <typename... Cols> void set_col_type(int type, int c, const Cols &...cols) {
        _set_formatter_by_type(c, type);
        if constexpr (sizeof...(cols) > 0) {
            set_col_type(type, cols...);
        }
    }
    template <typename... Cols>
    void set_col_type(int type, const std::string &col, const Cols &...cols) {
        int c = col_ni_map[col];
        set_col_type(type, c, cols...);
    }

    template <typename... Cols>
    void set_col_formatter(const std::string &formatter, const std::string &alt_formatter, int c,
                           const Cols &...cols) {
        col_formatter[c] = formatter;
        col_formatter_alt[c] = alt_formatter;
        if constexpr (sizeof...(cols) > 0) {
            set_col_formatter(formatter, alt_formatter, cols...);
        }
    };
    template <typename... Cols>
    void set_col_formatter(const std::string &formatter, const std::string &alt_formatter,
                           const std::string &name, const Cols &...cols) {
        int c = col_ni_map[name];
        set_col_formatter(formatter, alt_formatter, c, cols...);
    }

    template <typename... Cols>
    void set_type_formatter(const std::string &formatter, const std::string &alt_formatter,
                            int type) {
        type_formatter[type] = formatter;
        type_formatter_alt[type] = alt_formatter;
        for (int i = 0; i < col_formatter.size(); ++i) {
            if (col_type[i] == type) {
                col_formatter[i] = formatter;
                col_formatter_alt[i] = alt_formatter;
            }
        }
    }

    template <typename T> T loc(int i, std::string colname) {
        return loc<T>(i, col_ni_map[colname]);
    }
    template <typename T> T loc(int i, int j) { return std::get<T>(data[j][i]); }
    // Add a vertical separate line after cols when print_row().
    template <typename... Cols> void add_sep_after(const std::string &col, const Cols &...cols) {
        int c = col_ni_map[col];
        sep_set.insert(c);
        if constexpr (sizeof...(cols) > 0)
            add_sep_after(cols...);
    }

    int get_best_row(const std::string &colname, bool min_v) {
        const int c = col_ni_map[colname];
        const size_t n_row = data[0].size();

        // The column may contain blank values so we need use a base value to occupy.
        double base_v;
        if (min_v) {
            base_v = 1e99;
        } else {
            base_v = -1e99;
        }
        std::vector<double> cv(n_row, base_v);
        for (int i = 0; i < n_row; ++i) {
            const auto &cell = data[c][i];
            if (cell.index() == 1) {
                cv[i] = std::get<double>(cell);
            } else if (cell.index() == 2) {
                cv[i] = std::get<int>(cell);
            }
        }
        int r = 0;
        if (min_v) {
            auto mit = std::min_element(cv.begin(), cv.end());
            r = mit - cv.begin();
        } else {
            auto mit = std::max_element(cv.begin(), cv.end());
            r = mit - cv.begin();
        }

        return r;
    }

    // if min_v=true, then find row with minimum value in colname.
    // filename: write to file.
    void print_best_row(const std::string &colname, bool min_v, const std::string &filename = "") {
        int r = get_best_row(colname, min_v);
        const auto row = format_row(r, " ", "| ");
        fmt::print(fmt::fg(fmt::color::yellow), "{}\n", row);

        if (!filename.empty()) {
            write_row_with_header(r, filename);
        }
    }

  private:
    std::array<std::string, 3> type_formatter{"{: >9.4f}", "{: >6}", "{: >9}"};
    std::array<std::string, 3> type_formatter_alt{"{: ^9}", "{: ^6}",
                                                  "{: ^9}"}; // Formatter if no value.
    std::vector<std::vector<CellT>> data;
    // std::vector<CellT> row;
    std::vector<int> col_type; // 0: double, 1: int,  2: string
    std::vector<std::string> col_formatter;
    std::vector<std::string> col_formatter_alt;
    std::unordered_map<std::string, int> col_ni_map; // Name to index.
    std::vector<std::string> col_in_map;             // Index to name.
    std::unique_ptr<std::ofstream> file;
    std::unordered_set<int> sep_set;

    // Track new best value
    std::string monitor_col;
    double best_value = 1e99;
    int best_row = 0;
    bool min_v = false;

    void _set_formatter_by_type(int col, int type) {
        col_type[col] = type;
        col_formatter[col] = type_formatter[type];
        col_formatter_alt[col] = type_formatter_alt[type];
    }
};

} // namespace SciLib