/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#ifndef NDARRAY_STRING_UTILS_H
#define NDARRAY_STRING_UTILS_H

#include <regex>
#include <string>

namespace green::ndarray {

  // String trimming from https://stackoverflow.com/a/18465058

  inline std::string ltrim(const std::string& s) {
    static const std::regex lws{"^[[:space:]]*", std::regex_constants::extended};
    return std::regex_replace(s, lws, "");
  }

  inline std::string rtrim(const std::string& s) {
    static const std::regex tws{"[[:space:]]*$", std::regex_constants::extended};
    return std::regex_replace(s, tws, "");
  }

  inline std::string trim(const std::string& s) { return ltrim(rtrim(s)); }

  inline bool        all_latin(const std::string& s) {
    static const std::regex non_latin("[^a-zA-Z]");
    std::smatch             m;
    return !std::regex_search(s, m, non_latin);
  }

}  // namespace green::ndarray
#endif  // NDARRAY_STRING_UTILS_H
