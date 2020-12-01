---
title: "[TF2.0]keras Layer Wrapping 하기"
date: 2020-12-01T21:31:40+09:00
description: TF2 keras에서 모델의 Layer들에 Wrapper를 씌우는법을 알아봅시다.
---
실험을 위해서, 이미 존재하는 keras모델의 특정레이어들에서, Input/Output Tensor들에 약간의 조작이 필요한일이 생겼습니다. 따라서 Layer를 특정 Wrapper로 감싸려고 하였는데요,

처음에는 단순히 기존 모델생성코드를 수정하여, 사이사이 Wrapper Function을 추가하는걸 생각해봤습니다. 

하지만 레이어가 수십개라면 너무 귀찮은 작업이고 코드의 범용성도 떨어지기에 이참에 범용적으로 적용가능하게 코드를 작성하려고 해봤습니다.

우선 생각해보니, tfmot패키지의 prune_low_magnitude함수가 이러한 작업을 하던것이 기억나, 우선 tfmot의 소스코드의 분석부터 진행해 보았습니다.

## TFMOT 소스코드 분석

우선 prune_low_magnitude() 함수부터 시작합니다.

### prune_low_magnitude(model,...)

[소스코드](https://github.com/tensorflow/model-optimization/blob/a6b401c28d7f0ee5d36f31b00d34c75a36e08362/tensorflow_model_optimization/python/core/sparsity/keras/prune.py)

```python
def _add_pruning_wrapper(layer):
    if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
      return layer
    return pruning_wrapper.PruneLowMagnitude(layer, **params)

...

elif is_sequential_or_functional:
    return keras.models.clone_model(
        to_prune, input_tensors=None, clone_function=_add_pruning_wrapper)
```

핵심은 이부분인데, 보시게 되면
keras.models.clone_model의 clone_function인자를 사용하여 Wrapper function을 주고 있습니다.

wrapper function으로 PruneLowMagnitude란 클래스에 레이어를 인자로 주고 있는데, 이 클래스를 보게 

### Class:PruneLowMagnitude(keras.layers.Wrapper)

[소스코드](https://github.com/tensorflow/model-optimization/blob/e8e5266b96498fe6b1807665ce26ca8f0213b2f0/tensorflow_model_optimization/python/core/sparsity/keras/pruning_wrapper.py)

- 결론적인 방법은 **결론 절**에 적어두었습니다

```python
Wrapper = keras.layers.Wrapper

class PruneLowMagnitude(Wrapper):
  _PRUNE_CALLBACK_ERROR_MSG = (
      'Prune() wrapper requires the UpdatePruningStep callback to be provided '
      'during training. Please add it as a callback to your model.fit call.')

  def __init__(self,
               layer,
               pruning_schedule=pruning_sched.ConstantSparsity(0.5, 0),
               block_size=(1, 1),
               block_pooling_type='AVG',
               **kwargs):
    #내용생략

    if isinstance(layer, prunable_layer.PrunableLayer):
      # Custom layer in client code which supports pruning.
      super(PruneLowMagnitude, self).__init__(layer, **kwargs)
    elif prune_registry.PruneRegistry.supports(layer):
      # Built-in keras layers which support pruning.
      super(PruneLowMagnitude, self).__init__(
          prune_registry.PruneRegistry.make_prunable(layer), **kwargs)
    else:
      raise ValueError(
          'Please initialize `Prune` with a supported layer. Layers should '
          'either be a `PrunableLayer` instance, or should be supported by the '
          'PruneRegistry. You passed: {input}'.format(input=layer.__class__))

    #내용생략

  def build(self, input_shape):
    super(PruneLowMagnitude, self).build(input_shape)
		self.add_variable( some_needed_variable )
		
		#내용생략

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    #내용생략

    if 'training' in args:
      return self.layer.call(inputs, training=training)

    return self.layer.call(inputs)

  def compute_output_shape(self, input_shape):
    return self.layer.compute_output_shape(input_shape)

  @property
  def trainable(self):
    return self.layer.trainable

  @trainable.setter
  def trainable(self, value):
    self.layer.trainable = value

  @property
  def trainable_weights(self):
    return self.layer.trainable_weights

  @property
  def non_trainable_weights(self):
    return self.layer.non_trainable_weights + self._non_trainable_weights

  @property
  def updates(self):
    return self.layer.updates + self._updates

  @property
  def losses(self):
    return self.layer.losses + self._losses

  def get_weights(self):
    return self.layer.get_weights()

  def set_weights(self, weights):
    self.layer.set_weights(weights)
```

위와 같은 구조를 하고 있습니다. 소스코드를 좀 많이 쳐냈는데, 밑에서 자세히 옮겨놨기도 하고Wrapper의 사용법이 궁금한거지 Pruning 하는 방법이 궁금한게 아니므로 쳐냈습니다.

### class : keras.layers.Wrapper

이 클래스는[keras.layers.Wrapper](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Wrapper)란 Abstract Class를 Implement 하고 있습니다.

이 Abstract class는 [이 소스코드](https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/wrappers.py#L40-L81) 에서 확인할 수 있는데, 큰 내용이 있진 않습니다. 한번 확인해보세요.

```python
class Wrapper(Layer):
  """Abstract wrapper base class.
  Wrappers take another layer and augment it in various ways.
  Do not use this class as a layer, it is only an abstract base class.
  Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.
  Arguments:
    layer: The layer to be wrapped.
  """

  def __init__(self, layer, **kwargs):
    assert isinstance(layer, Layer)
    self.layer = layer
    super(Wrapper, self).__init__(**kwargs)

def build(self, input_shape=None):
    if not self.layer.built:
      self.layer.build(input_shape)
      self.layer.built = True
    self.built = True

  @property
  def activity_regularizer(self):
    if hasattr(self.layer, 'activity_regularizer'):
      return self.layer.activity_regularizer
    else:
      return None

  def get_config(self):
    config = {'layer': generic_utils.serialize_keras_object(self.layer)}
    base_config = super(Wrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    # Avoid mutating the input dict
    config = copy.deepcopy(config)
    layer = deserialize_layer(
        config.pop('layer'), custom_objects=custom_objects)
    return cls(layer, **config)
```

- 중요한 부분은, self.layer에 원본 레이어를 저장한다는 것입니다.

### Class:PruneLowMagnitude(keras.layers.Wrapper) 분석

keras.Wrapper class가 따로 Document가 존재하지 않아서 PruneLowMagnitude의 실제 구현에서 사용방법을 살펴볼 필요가 있습니다.

우리가 Wrapper Class를 구현할때 가장 크게 관심가질 부분은

- __init__
- build
- call

입니다. 그부분의 핵심 코드를 확인하면 다음과 같습니다.

__**Init__**

```python
  if isinstance(layer, prunable_layer.PrunableLayer):
      # Custom layer in client code which supports pruning.
      super(PruneLowMagnitude, self).__init__(layer, **kwargs)
    elif prune_registry.PruneRegistry.supports(layer):
      # Built-in keras layers which support pruning.
      super(PruneLowMagnitude, self).__init__(
          prune_registry.PruneRegistry.make_prunable(layer), **kwargs)
    else:
```

Wrapper class를 layer넣어서 Init 해주는 걸로 충분해보입니다.

**Build**

```python
def build(self, input_shape):
    super(PruneLowMagnitude, self).build(input_shape)

    weight_vars, mask_vars, threshold_vars = [], [], []

    self.prunable_weights = self.layer.get_prunable_weights()

    # For each of the prunable weights, add mask and threshold variables
    for weight in self.prunable_weights:
      mask = self.add_variable(
          'mask',
          shape=weight.shape,
          initializer=tf.keras.initializers.get('ones'),
          dtype=weight.dtype,
          trainable=False,
          aggregation=tf.VariableAggregation.MEAN)
      threshold = self.add_variable(
          'threshold',
          shape=[],
          initializer=tf.keras.initializers.get('zeros'),
          dtype=weight.dtype,
          trainable=False,
          aggregation=tf.VariableAggregation.MEAN)

      weight_vars.append(weight)
      mask_vars.append(mask)
      threshold_vars.append(threshold)
    self.pruning_vars = list(zip(weight_vars, mask_vars, threshold_vars))

    # Add a scalar tracking the number of updates to the wrapped layer.
    self.pruning_step = self.add_variable(
        'pruning_step',
        shape=[],
        initializer=tf.keras.initializers.Constant(-1),
        dtype=tf.int64,
        trainable=False)
```

`super(PruneLowMagnitude, self).build(input_shape)` 로 원래 layur의 build를 호출해주고,

Wrapper에서 따로 필요한 Variables를 할당하는것으로 충분해보입니다.

**Call**

```python
def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase() #Training 상태가 명시되지 않았을 경우 Keras Backend에서 읽어옴.

    def add_update():
      with tf.control_dependencies([
          tf.debugging.assert_greater_equal(
              self.pruning_step,
              np.int64(0),
              message=self._PRUNE_CALLBACK_ERROR_MSG)
      ]):
        with tf.control_dependencies(
            [self.pruning_obj.conditional_mask_update()]):
          return tf.no_op('update')

    def no_op():
      return tf.no_op('no_update')

    update_op = utils.smart_cond(training, add_update, no_op)
    self.add_update(update_op)
    # Always execute the op that performs weights = weights * mask
    # Relies on UpdatePruningStep callback to ensure the weights
    # are sparse after the final backpropagation.
    #
    # self.add_update does nothing during eager execution.
    self.add_update(self.pruning_obj.weight_mask_op())
    # TODO(evcu) remove this check after dropping py2 support. In py3 getargspec
    # is deprecated.
    if hasattr(inspect, 'getfullargspec'):
      args = inspect.getfullargspec(self.layer.call).args
    else:
      args = inspect.getargspec(self.layer.call).args
    # Propagate the training bool to the underlying layer if it accepts
    # training as an arg.
    if 'training' in args:
      return self.layer.call(inputs, training=training)

    return self.layer.call(inputs)
```

이것도 역시 `self.layer.call(inputs)` 만 해주면 되는것으로 보이네요.

## 결론

분석해보니 필요한 구현은 세가지 입니다.

- keras.Wrapper class를 Implement한 WrapperClass 생성
    - 필수는 아니지만, 공식 Abstract Class가 있으니 따릅시다.
- 어떤 Layer을 Wrap할지 체크하는 함수
- 이 clone function를 keras.clone_model( clone_function인자에 전달)

제가 사용하는 구현의 의사코드는 다음과 같습니다.

```python
class MyWrapper(keras.Wrapper):
	def __init__(self,layer, some_options... ):
		super(MyWrapper, self).__init__(layer, **kwargs)
		... initialize some other stuff ...
	
	def build(self, input_shape):
		super(MyWrapper, self).build(input_shape)
		self.addVariable('some_needed_variable', ... )
		...

	def call(self, input, training=None):
		if training is None:
	    training = K.learning_phase()

		"""DO SOMETHING WHEN CALLING"""

		if 'training' in args:
      return self.layer.call(inputs, training=training)
    return self.layer.call(inputs)

def _wrapper_func(layer):
	if layer is some_condition :
		return MyWrapper(layer, some_option...)
	else :
		return layer
		
def _unwrapper_func(layer):
	if isinstance(layer,MyWrapper):
		return layer.layer #keras.Wrapper클래스는 self.layer에 원본레이어 저장
	else :
		return layer

def wrap_model(model):
	return keras.models.clone_model(
        model, input_tensors=None, clone_function=_wrapper_func)

def unwrap_model(model):
	return keras.models.clone_model(
        model, input_tensors=None, clone_function=_unwrapper_func)

```

제 실험에서는, Training시만 Wrapper가 필요하고 Inference시에는 필요없으므로 Unwrap_model 함수또한 만들어 두었습니다.

감사합니다.
