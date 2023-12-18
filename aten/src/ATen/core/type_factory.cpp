#include <ATen/core/type_factory.h>

#include <ATen/core/jit_type.h>

namespace c10 {

// Dtype constraints are not constrained in compilation. Therefore, we map
// all tensor subclasses with different dtypes to a same underlying
// Tensor. But, we give warning about possible dtype change whenever user
// uses any of the tensor subclasses such as LongTensor.
//
// Technically "number" is not a python type but we need it when
// parsing serialized methods that use implicit conversions to Scalar
#define FORALL_BASE_PYTHON_TYPES(_) \
  _(Tensor, TensorType)             \
  _(int, IntType)                   \
  _(float, FloatType)               \
  _(bool, BoolType)                 \
  _(complex, ComplexType)           \
  _(str, StringType)                \
  _(Device, DeviceObjType)          \
  _(Stream, StreamObjType)          \
  _(number, NumberType)             \
  _(None, NoneType)                 \
  _(NoneType, NoneType)             \
  _(Any, AnyType)                   \
  _(Capsule, CapsuleType)           \
  _(list, AnyListType)              \
  _(tuple, AnyTupleType)

#define FORALL_SPECIALIZED_TENSOR_TYPES(_) \
  _(LongTensor, ScalarType::Long)          \
  _(DoubleTensor, ScalarType::Double)      \
  _(FloatTensor, ScalarType::Float)        \
  _(IntTensor, ScalarType::Int)            \
  _(ShortTensor, ScalarType::Short)        \
  _(HalfTensor, ScalarType::Half)          \
  _(CharTensor, ScalarType::Char)          \
  _(ByteTensor, ScalarType::Byte)          \
  _(BoolTensor, ScalarType::Bool)

const std::unordered_map<std::string, c10::TypePtr>& DynamicTypeFactory::
    basePythonTypes() {
  static const std::unordered_map<std::string, c10::TypePtr> map = {
#define MAP_ITEM(NAME, TYPE) \
  {#NAME, c10::DynamicTypeTrait<c10::TYPE>::getBaseType()},
      FORALL_BASE_PYTHON_TYPES(MAP_ITEM)
#undef MAP_ITEM
#define MAP_ITEM(NAME, ...) \
  {#NAME, c10::DynamicTypeTrait<c10::TensorType>::getBaseType()},
      FORALL_SPECIALIZED_TENSOR_TYPES(MAP_ITEM)
#undef MAP_ITEM
  };
  return map;
}

const std::unordered_map<std::string, c10::TypePtr>& DefaultTypeFactory::
    basePythonTypes() {
  static const std::unordered_map<std::string, c10::TypePtr> map = {
#define MAP_ITEM(NAME, TYPE) {#NAME, c10::TYPE::get()},
      FORALL_BASE_PYTHON_TYPES(MAP_ITEM)
#undef MAP_ITEM
#define MAP_ITEM(NAME, DTYPE) {#NAME, c10::TensorType::get()->withScalarType(DTYPE)},
      FORALL_SPECIALIZED_TENSOR_TYPES(MAP_ITEM)
#undef MAP_ITEM
  };
  return map;
}

c10::TypePtr DefaultTypeFactory::createNamedTuple(
    const std::string& name,
    const std::vector<c10::string_view>& fields,
    const std::vector<c10::TypePtr>& types) {
  return c10::TupleType::createNamed(name, fields, types);
}

} // namespace c10
