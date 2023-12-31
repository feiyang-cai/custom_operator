#include "custom_dynamics_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>
#include <iostream>

//#include "core/common/common.h"
#include "cassert"

static const char* c_OpDomain = "custom_dynamics";

struct OrtCustomOpDomainDeleter {
  explicit OrtCustomOpDomainDeleter(const OrtApi* ort_api) {
    ort_api_ = ort_api;
  }
  void operator()(OrtCustomOpDomain* domain) const {
    ort_api_->ReleaseCustomOpDomain(domain);
  }

  const OrtApi* ort_api_;
};

using OrtCustomOpDomainUniquePtr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>;
static std::vector<OrtCustomOpDomainUniquePtr> ort_custom_op_domain_container;
static std::mutex ort_custom_op_domain_mutex;

static void AddOrtCustomOpDomainToContainer(OrtCustomOpDomain* domain, const OrtApi* ort_api) {
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  auto ptr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>(domain, OrtCustomOpDomainDeleter(ort_api));
  ort_custom_op_domain_container.push_back(std::move(ptr));
}

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

struct DynamicsKernel {
  DynamicsKernel(OrtApi api)
      : api_(api),
        ort_(api_) {
  }

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    //Ort::KernelContext ctx(context);
    //auto input = ctx.GetInput(0);
    const OrtValue* input = ort_.KernelContext_GetInput(context, 0);

    //auto input_dimensions = input.GetTensorTypeAndShapeInfo().GetShape();
    //const size_t input_size = input.GetTensorTypeAndShapeInfo().GetElementCount();
    //assert(input_size == 5 && "Input should be 5-D tensor.");

    //auto output = ctx.GetOutput(0, input_dimensions);
    //auto output_dimensions = output.GetTensorTypeAndShapeInfo().GetShape();
    //const size_t output_size = output.GetTensorTypeAndShapeInfo().GetElementCount();
    //assert(output_size == 5 && "Output should be 5-D tensor.");

    //const float* X = input.GetTensorData<float>();
    const float* X = ort_.GetTensorData<float>(input);
    // Setup output
    OrtTensorDimensions dimensions(ort_, input);

    OrtTensorTypeAndShapeInfo* input_info = ort_.GetTensorTypeAndShape(input);
    int64_t size = ort_.GetTensorShapeElementCount(input_info);
    //std::cout << "size = " << size << std::endl;
    //std::cout << "dimensions.size() = " << dimensions.data() << std::endl;


    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    float* out = ort_.GetTensorMutableData<float>(output);
    //float* out = output.GetTensorMutableData<float>();

    for (int i = 0; i < size; i++){
      out[i] = X[i];
      //std::cout << "out[" << i << "] = " << out[i] << std::endl;
    }

    //out[0] = X[0]; // z0
    //out[1] = X[1]; // z1
    //out[2] = X[2];
    //out[3] = X[3];
    //out[4] = X[4];

    int64_t p_dim = (size>=5) ? 2 : 0;
    int64_t theta_dim = (size>=5) ? 3 : 1;

    for (int i = 0; i < 20; i++) {
      out[p_dim] = out[p_dim] + std::sin(out[theta_dim]) * 0.25f; // p
      out[theta_dim] = out[theta_dim] + std::tan(out[size-1]) * 0.05f; // theta
    }

  }
  private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};

// legacy custom op registration
struct CustomDynamics : Ort::CustomOpBase<CustomDynamics, DynamicsKernel> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
    return new DynamicsKernel(api);
  };

  //void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* /* info */) const {
  //  return std::make_unique<DynamicsKernel>().release();
  //};

  const char* GetName() const { return "TaxiNetDynamics"; };

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
} c_CustomDynamics;

//static void AddOrtCustomDynamicsDomainToContainer(Ort::CustomOpDomain&& domain) {
//  static std::vector<Ort::CustomOpDomain> ort_custom_dynamics_domain_container;
//  static std::mutex ort_custom_op_domain_mutex;
//  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
//  ort_custom_dynamics_domain_container.push_back(std::move(domain));
//}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

  AddOrtCustomOpDomainToContainer(domain, ortApi);

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomDynamics)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);


  //Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);

  //static const CustomDynamics c_CustomDynamics;
  //OrtStatus* result = nullptr;

  //ORT_TRY {
  //  Ort::CustomOpDomain domain{c_OpDomain};
  //  domain.Add(&c_CustomDynamics);

  //  Ort::UnownedSessionOptions session_options(options);
  //  session_options.Add(domain);
  //  AddOrtCustomDynamicsDomainToContainer(std::move(domain));
  //}
  //ORT_CATCH(const std::exception& e) {
  //  ORT_HANDLE_EXCEPTION([&]() {
  //    Ort::Status status{e};
  //    result = status.release();
  //  });
  //}
  //return result;
}

//OrtStatus* ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api) {
//  return RegisterCustomOps(options, api);
//}// custom_op_library.cc
