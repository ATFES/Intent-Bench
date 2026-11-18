#include <weft/runtime.h>
#include <chrono>
#include <cstdio>
#include <dlfcn.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Dispatch = void (*)(weft_native_entry, void *const *);
struct Call {
  weft_context *context = nullptr;
  weft_kernel *kernel = nullptr;
  weft_call *call = nullptr;
  weft_native_entry entry = nullptr;
  void *library = nullptr;
  void *const *arguments = nullptr;
  Dispatch dispatch = nullptr;
  ~Call() {
    if (call) weft_call_release(call);
    if (kernel) weft_kernel_release(kernel);
    if (context) weft_context_release(context);
    if (library) dlclose(library);
  }
};
void require(weft_status status, const weft_error &error) {
  if (status != WEFT_OK) throw std::runtime_error(error.message);
}
}

extern "C" void *prepare(const char *directory, const weft_view *views,
                         size_t count, weft_error *error) {
  try {
    auto result = std::make_unique<Call>();
    require(weft_context_create(&result->context, error), *error);
    require(weft_kernel_open(result->context, directory, &result->kernel, error), *error);
    weft_kernel_info info{};
    require(weft_kernel_describe(result->kernel, &info, error), *error);
    if (info.argument_count != count) throw std::runtime_error("baseline View argument count differs from native ABI");
    std::vector<weft_value> values(count);
    for (size_t i = 0; i < count; ++i) {
      weft_argument_info argument{};
      require(weft_kernel_argument(result->kernel, i, &argument, error), *error);
      if (argument.kind != WEFT_VIEW) throw std::runtime_error("this baseline binding requires View arguments");
      values[i].kind = WEFT_VIEW;
      values[i].view = views[i];
    }
    require(weft_call_prepare(result->kernel, values.data(), count, nullptr, 0, &result->call, error), *error);
    size_t slots;
    require(weft_call_arguments(result->call, &result->arguments, &slots, error), *error);
    require(weft_kernel_get_entry(result->kernel, &result->entry, error), *error);
    const std::string library = std::string(directory) + "/kernel.so";
    result->library = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!result->library) throw std::runtime_error(dlerror());
    result->dispatch = reinterpret_cast<Dispatch>(dlsym(result->library, "weft_kernel_dispatch"));
    if (!result->dispatch) throw std::runtime_error("native export lacks Weft dispatch");
    return result.release();
  } catch (const std::exception &failure) {
    error->status = WEFT_ABI_ERROR;
    snprintf(error->message, sizeof(error->message), "%s", failure.what());
    return nullptr;
  }
}
extern "C" int validate(Call *call, weft_error *error) {
  return weft_kernel_check_thread(call->kernel, error);
}
extern "C" void launch(Call *call) { call->dispatch(call->entry, call->arguments); }
extern "C" double measure(Call *call) {
  const auto start = std::chrono::steady_clock::now();
  launch(call);
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
extern "C" void release(Call *call) { delete call; }
