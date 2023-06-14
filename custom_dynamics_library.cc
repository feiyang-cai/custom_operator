#include "custom_dynamics_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>
#include <iostream>

#include "core/common/common.h"
#include "cassert"

static const char* c_OpDomain = "custom_dynamics";

struct DynamicsKernel {
  void Compute(OrtKernelContext* context) {
    // Setup inputs
    Ort::KernelContext ctx(context);
    auto input = ctx.GetInput(0);
    auto input_dimensions = input.GetTensorTypeAndShapeInfo().GetShape();
    const size_t input_size = input.GetTensorTypeAndShapeInfo().GetElementCount();
    assert(input_size == 5 && "Input should be 5-D tensor.");

    auto output = ctx.GetOutput(0, input_dimensions);
    auto output_dimensions = output.GetTensorTypeAndShapeInfo().GetShape();
    const size_t output_size = output.GetTensorTypeAndShapeInfo().GetElementCount();
    assert(output_size == 5 && "Output should be 5-D tensor.");

    const float* X = input.GetTensorData<float>();
    // Setup output
    float* out = output.GetTensorMutableData<float>();

    out[0] = X[0]; // z0
    out[1] = X[1]; // z1
    out[2] = X[2] + std::sin(X[3]) * 0.25f; // p
    out[3] = X[3] + std::tan(X[4]) * 0.05f; // theta
    out[4] = X[4];

  }
};

// legacy custom op registration
struct CustomDynamics : Ort::CustomOpBase<CustomDynamics, DynamicsKernel> {
  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* /* info */) const {
    return std::make_unique<DynamicsKernel>().release();
  };

  const char* GetName() const { return "TaxiNetDynamics"; };

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
};

static void AddOrtCustomDynamicsDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_dynamics_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_dynamics_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);

  static const CustomDynamics c_CustomDynamics;
  OrtStatus* result = nullptr;

  ORT_TRY {
    Ort::CustomOpDomain domain{c_OpDomain};
    domain.Add(&c_CustomDynamics);

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    AddOrtCustomDynamicsDomainToContainer(std::move(domain));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      Ort::Status status{e};
      result = status.release();
    });
  }
  return result;
}

OrtStatus* ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api) {
  return RegisterCustomOps(options, api);
}// custom_op_library.cc
