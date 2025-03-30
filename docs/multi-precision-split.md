# Multi Precision Split Research

## Introduction to Multi precision

Multi precision refers to the process of converting the model to a lower precision, such as bfloat16 to int8, or float16 to bfloat16. The lower precision model can be used to save time and memory, but the performance of the model may be affected.

| Precision Type | Bytes per Parameter | 671B Storage | Typical Usage                                                |
| :------------: | :-----------------: | :----------: | :----------------------------------------------------------- |
|      FP16      |          2          |   13.42 TB   | Mixed-precision training commonly leverages FP16 to speed up training while keeping key operations in higher precision when needed. This approach strikes a good balance between performance and resource usage, making it popular for many modern training pipelines. |
|      BF16      |          2          |   13.42 TB   | Comparing to FP16, BF16 has a larger exponent part, often provides better numerical stability during large-scale training. It enables efficient training of massive models while maintaining performance close to FP32 but with lower memory consumption. |
|      INT8      |          1          |    671 GB    | This lower precision format is primarily used for inference in environments with limited resources. Although it can introduce some accuracy loss, proper calibration techniques help retain sufficient performance. |

- FP16 has 1bit sign bit, 5bit exponent and 10bit mantissa, which can represent nearly 2^16 distinct values.

- BF16 has 1bit sign bit, 8bit exponent and 7bit mantissa, which can represent nearly 2^128 distinct values.

- INT8 has 1bit sign bit, 7bit exponent and 0bit mantissa, which can represent 2^8 distinct values.

## Implementation

## Feature Introduction

Saving the model with a higher precision, such as float32, can be a time-consuming and memory-consuming process. However, the model can be exported with lower precision but still remaining robust performance to save time and memory. The following is a simple implementation of the split of the model with lower precision.

The split of the model with lower precision is implemented in `export_split.py`.

- **New argument:** `--quantization`, can be set to `int8`, `bf16`... to specify the quantization method. This argument is defaulted to None to indicate that no quantization is performed.
- **New method:** `quantize_tensors`, which is used to convert the model tensors to lower precision. The method should be called in `process_file_sync` to deal with the tensors that need to be quantized.
- **New method:** `check_tensor_quantization`, which is used to check the quantization of the model tensors. The method should be called in `quantize_tensors` to decide whether quantization is needed.

## TODO

- [ ] Add a new argument `--quantization` to `export_split.py` to specify the quantization method.
- [ ] Implement the `quantize_tensors` method to convert the model tensors to lower precision.
- [ ] Implement the `check_tensor_quantization` method to check the quantization of the model tensors.
