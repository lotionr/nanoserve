// `nanoserve serve` (F035): the OpenAI-style completions endpoint.
#pragma once

#include <string>
#include <vector>

namespace nano {

/// Runs the HTTP server. Blocks forever once listening (the process is the
/// server); returns nonzero only on bad usage or startup failure.
int cmd_serve(const std::vector<std::string>& args);

}  // namespace nano
