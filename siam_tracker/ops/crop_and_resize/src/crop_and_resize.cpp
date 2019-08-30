#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor crop_and_resize_cuda_forward(
    at::Tensor image,
    at::Tensor boxes,
    at::Tensor box_index,
    const float extrapolation_value,
    const int crop_height,
    const int crop_width);

at::Tensor crop_and_resize_cuda_backward(
    at::Tensor grads,
    at::Tensor boxes,
    at::Tensor box_index,
    const int batch,
    const int image_height,
    const int image_width
);

// C++ interface

at::Tensor crop_and_resize_cpu_forward(
    at::Tensor image,
    at::Tensor boxes,
    at::Tensor box_index,
    const float extrapolation_value,
    const int crop_height,
    const int crop_width);

at::Tensor crop_and_resize_cpu_backward(
    at::Tensor grads,
    at::Tensor boxes,
    at::Tensor box_index,
    const int batch,
    const int image_height,
    const int image_width
);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

at::Tensor crop_and_resize_forward(
    at::Tensor image,
    at::Tensor boxes,
    at::Tensor box_index,
    const float extrapolation_value,
    const int crop_height,
    const int crop_width) 
{
    CHECK_CONTIGUOUS(image);
    CHECK_CONTIGUOUS(boxes);
    CHECK_CONTIGUOUS(box_index);
    
    if(image.type().is_cuda())
    {
        CHECK_CUDA(boxes);
        CHECK_CUDA(box_index);
        return crop_and_resize_cuda_forward(image, boxes, box_index, extrapolation_value, crop_height, crop_width);
    }
    else
    {
        return crop_and_resize_cpu_forward(image, boxes, box_index, extrapolation_value, crop_height, crop_width);
    }
}


at::Tensor crop_and_resize_backward(
    at::Tensor grads,
    at::Tensor boxes,
    at::Tensor box_index,
    const int batch,
    const int image_height,
    const int image_width
)
{
    CHECK_CONTIGUOUS(grads);
    CHECK_CONTIGUOUS(boxes);
    CHECK_CONTIGUOUS(box_index);

    if(grads.type().is_cuda())
    {
        CHECK_CUDA(boxes);
        CHECK_CUDA(box_index);
        return crop_and_resize_cuda_backward(grads, boxes, box_index, batch, image_height, image_width);
    }
    else
    {
        return crop_and_resize_cpu_backward(grads, boxes, box_index, batch, image_height, image_width);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &crop_and_resize_forward, "Crop and resize forward");
  m.def("backward", &crop_and_resize_backward, "Crop and resize backward");
}
