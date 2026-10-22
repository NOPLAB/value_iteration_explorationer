// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#include "value_iteration3/actions.hpp"

#include <cctype>
#include <exception>
#include <fstream>
#include <string>
#include <utility>

namespace value_iteration3 {
namespace {

std::string strip_comment(const std::string &line) {
  bool quote = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '"' || line[i] == '\'') {
      quote = !quote;
    } else if (line[i] == '#' && !quote) {
      return line.substr(0, i);
    }
  }
  return line;
}

std::string trim(std::string text) {
  if (!text.empty() && text.back() == '\r') {
    text.pop_back();
  }
  std::size_t begin = 0;
  while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
    ++begin;
  }
  std::size_t end = text.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
    --end;
  }
  return text.substr(begin, end - begin);
}

std::string unquote(std::string text) {
  text = trim(std::move(text));
  if (text.size() >= 2 && ((text.front() == '"' && text.back() == '"') ||
                           (text.front() == '\'' && text.back() == '\''))) {
    return text.substr(1, text.size() - 2);
  }
  return text;
}

int indent_of(const std::string &line) {
  int indent = 0;
  while (indent < static_cast<int>(line.size()) && line[static_cast<std::size_t>(indent)] == ' ') {
    ++indent;
  }
  return indent;
}

bool parse_double(const std::string &text, double &out) {
  try {
    std::size_t used = 0;
    out = std::stod(text, &used);
    return used == text.size();
  } catch (const std::exception &) {
    return false;
  }
}

}  // namespace

ActionList read_action_list(const std::string &path) {
  ActionList result;
  std::ifstream input(path);
  if (!input) {
    result.error = "cannot open " + path;
    return result;
  }

  bool in_list = false;
  int list_indent = 0;
  bool have_item = false;
  bool saw_forward = false;
  bool saw_rotate = false;
  Action current;

  auto finish = [&]() -> bool {
    if (!have_item) {
      return true;
    }
    if (current.name.empty() || !saw_forward || !saw_rotate) {
      result.error = "action '" + current.name + "' needs name, onestep_forward_m, and onestep_rotation_deg";
      return false;
    }
    result.actions.push_back(current);
    current = {};
    have_item = false;
    saw_forward = false;
    saw_rotate = false;
    return true;
  };

  std::string raw;
  while (std::getline(input, raw)) {
    const int indent = indent_of(raw);
    const std::string line = trim(strip_comment(raw));
    if (line.empty()) {
      continue;
    }
    if (!in_list) {
      if (line == "action_list:") {
        in_list = true;
        list_indent = indent;
      }
      continue;
    }
    if (indent <= list_indent && line[0] != '-') {
      if (!finish()) {
        return result;
      }
      break;
    }
    if (line[0] == '-') {
      if (!finish()) {
        return result;
      }
      const std::string body = trim(line.substr(1));
      const auto colon = body.find(':');
      if (colon == std::string::npos || trim(body.substr(0, colon)) != "name") {
        result.error = "each action starts with '- name:'";
        return result;
      }
      current.name = unquote(body.substr(colon + 1));
      have_item = true;
      continue;
    }
    if (!have_item) {
      result.error = "action field before '- name:'";
      return result;
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      result.error = "bad action line: " + line;
      return result;
    }
    const std::string key = trim(line.substr(0, colon));
    const std::string value = unquote(line.substr(colon + 1));
    double number = 0;
    if (key == "onestep_forward_m") {
      if (!parse_double(value, number)) {
        result.error = "bad onestep_forward_m for " + current.name;
        return result;
      }
      current.forward_m = number;
      saw_forward = true;
    } else if (key == "onestep_rotation_deg") {
      if (!parse_double(value, number)) {
        result.error = "bad onestep_rotation_deg for " + current.name;
        return result;
      }
      current.rotate_deg = number;
      saw_rotate = true;
    } else if (key != "name") {
      result.error = "unknown action field '" + key + "'";
      return result;
    }
  }
  if (in_list && !finish()) {
    return result;
  }
  if (!in_list) {
    result.error = "action_list not found in " + path;
    return result;
  }
  if (result.actions.empty()) {
    result.error = "action_list is empty";
  }
  return result;
}

}  // namespace value_iteration3
