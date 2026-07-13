#pragma once

#include <chrono>
#include <memory>

#include <c10/core/DeviceType.h>
#include <c10/util/Registry.h>

namespace c10d::cuda {

enum StreamBlockStatus : int32_t {
  UNKNOWN = 0,
  RUNNING = 1,
  TIMED_OUT = 2,
  ABORTED = 3,
};

/*
StreamBlock implements a baton that will block the active stream
until aborted by the main process.

Third-party backends register their implementation via:
  C10_REGISTER_CLASS(StreamBlockRegistry, <DeviceName>, MyStreamBlock)

The constructor receives a std::chrono::milliseconds timeout.
The implementation must:
  - Block the current device stream in the constructor
  - Release the block when abort() is called (may be called from any thread)
  - Be safe to destruct without waiting if abort() has been called

The backend's <DeviceName> must match c10::DeviceTypeName(device_type).
*/
class TORCH_API StreamBlock {
 public:
  virtual ~StreamBlock() = default;
  virtual void abort() = 0;
  virtual StreamBlockStatus status() = 0;
};

// Block the current stream of the given device type. Looks up the
// StreamBlockRegistry using c10::DeviceTypeName(device_type) as the key.
std::unique_ptr<StreamBlock> block_stream(
    std::chrono::milliseconds timeout,
    c10::DeviceType device_type);

// Backward-compatible overload: defaults to CUDA.
std::unique_ptr<StreamBlock> block_stream(std::chrono::milliseconds timeout);

// Registry for StreamBlock implementations. The CUDA implementation is
// defined in StreamBlock.cu. Third-party backends register their own
// implementation via C10_REGISTER_CLASS with their device name as the key.
TORCH_DECLARE_REGISTRY(
    StreamBlockRegistry,
    StreamBlock,
    std::chrono::milliseconds);

} // namespace c10d::cuda
