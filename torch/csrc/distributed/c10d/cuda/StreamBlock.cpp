#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/cuda/StreamBlock.hpp>

namespace c10d::cuda {

C10_DEFINE_REGISTRY(StreamBlockRegistry, StreamBlock, std::chrono::milliseconds)

std::unique_ptr<StreamBlock> block_stream(
    std::chrono::milliseconds timeout,
    c10::DeviceType device_type) {
  auto key = c10::DeviceTypeName(device_type);
  auto baton = StreamBlockRegistry()->Create(key, timeout);
  TORCH_CHECK(baton, "Failed to create StreamBlock for device ", key);
  return baton;
}

std::unique_ptr<StreamBlock> block_stream(std::chrono::milliseconds timeout) {
  return block_stream(timeout, c10::DeviceType::CUDA);
}

} // namespace c10d::cuda
