<!-- translated-from: 135_forwarder_and_perception.jp.md blob:ebacf41daac668178c1fb44d4ff202d60afc1968 date:2026-11-25 progress:done -->
# Forwarder and Perception

## 1. Forwarder Contract

`Forwarder::Forward` in `anet/forwarder.hpp` takes a `TensorDict` and returns the transformed `TensorDict`. Network implements this interface too. Intermediate Network values are available through an optional `ForwardCapture`. See [forward capture in Neural Networks](130_neural_networks.en.md#76-forward-capture-and-partial-forward).

`ForwarderBuilder` assembles the modules written in configuration. The caller passes `BuildForwarder` a `ForwarderRequest` and the specs of the data it will actually supply. `ForwarderRequest` holds a `ForwardKind` naming the kind of transformation and accompanying information read by modules.

`GetInputSpecs` and `GetOutputSpecs` of the created `SpecForwarder` return the input and output specs for that request. For transformations that need per-row history, the created Forwarder holds the history.

## 2. Observation Specs

A `TensorSpec` shape excludes the batch axis. Value scale and the stack axis are recorded in JSON as `scale` and `stack_axis`.

| Attribute | Meaning |
|---|---|
| `scale = Scaled` | The scale has been set by the producer of the value or by normalization. Includes byte images and indices representing discrete values. |
| `scale = Unscaled` | A continuous value whose scale is aligned by normalization. Not allowed for integer dtypes or discrete keys. |
| `has_stack_axis = true` | The leading axis of the shape excluding the batch axis is the stack axis. Each position on that axis corresponds to one Frame. |

## 3. DefaultDQN Observation Processing

DefaultDQN uses the same specs as the Env Observation as the Frame specs. It creates the Forwarder for experience acceptance first and checks that its output keys, shapes, and dtypes match the Frame. The Frame specs are also passed to ReplayBuffer.

| Request | Input | Transformation |
|---|---|---|
| `FrameWithStatsUpdate` | `obs` and `next_obs`, one frame each | Updates statistics and passes the original values to ReplayBuffer. |
| `PerceptionFromSequence` | Observations moved to the Actor device and a boolean `episode_start` on CPU | Stacks per-lane history. Lanes whose start marker is true refill their history with the current Frame, and the start marker is removed from the output. |
| `PerceptionFromStack` | The stack assembled by ReplayBuffer. `GetSampleObsSpecs` returns the specs of the stack. | Learner transforms observations, next observations, and TempoRL window endpoints. |
| `PerceptionFromFrame` | One Frame prepared for visualization | `FrameStack` repeats the same Frame the specified number of times. |

Stacking is configured by `frame_stack.count`, `frame_stack.keys.mode`, and `frame_stack.keys.listed.names` directly under the Agent. The code default is one frame and the shipped baseline profile uses four. `all` selects every Frame key; `listed` selects the Frame keys in the list. Agent passes the same count and keys to ReplayBuffer and to each request.

In the final output of the module sequence, DefaultDQN adds an axis of length one to selected keys without a stack axis and removes the start marker if it remains. When `FrameStack` is assembled, the start-marker spec is checked to be a scalar `bool` per row without a stack axis (an empty shape). Before action selection, Actor checks that the actual output no longer contains the start marker. The visualization output specs are used as the Network input. The output specs for action selection and Sample are also checked against this input at construction. IQN taus are added to the Network input specs.

## 4. Standard Modules and Configuration

List names in `perception.structure`, as in `stack > norm`, and select modules with `perception.block.[name].type`. Builder assembles them in declaration order. Each module checks the specs that reach it and passes its output specs to the next module.

| Type | Configuration and behavior |
|---|---|
| `FrameStack` | The block setting is `type`. Takes the count and keys from the request and stacks the target keys that reach it. Sample keys that already have a stack axis pass through unchanged. |
| `Normalize` | `mode` is `none`, `constant`, or `running`. Changes the scale of normalized keys to `Scaled`. |
| `DropKeys` | Removes `keys` from Perception data and specs. Passes the keys through on experience acceptance. |
| `SetSpec` | Changes `scale`, `min_values`, `max_values`, and `labels` of the keys specified in `keys`. Passes values through unchanged. `forward_kinds.mode=listed` narrows it to the specified kinds. |

`running` statistics are prepared when the Forwarder for experience acceptance is created, using the `Unscaled` float keys that reach the module. Statistics are also collected for keys dropped later. A key that needs no statistics can be made `Scaled` by a preceding `SetSpec`. Construction stops if a key for which another kind requested normalization has no statistics or a different single-frame shape.

`running` updates observation statistics and `obs_norm.clip_ratio` on experience acceptance. `constant` updates the clip ratio. Processing that produces a Perception reads the statistics. If values are listed in `constant.mean` and `constant.std`, their count is checked against the number of elements in one frame.

`SetSpec` checks only the specified attributes. To set range values, specify either one value common to all elements or one value for each element of one frame. Specify labels for every element of a Vector, or for the leading dimension of one frame of a Grid or Sequence. Narrowing the kinds can change which keys are normalized on each path.

## 5. Diagrams and Extension

DefaultDQN records `perception.structure.dot` and `perception.detail.dot`, generated by the visualization Forwarder, in the Run's `dot/`. Each diagram is titled with the kind of request and connects the input, the modules in configured order, and the output. The detailed diagram shows output specs and each module's seed and configuration values.

To add a module, declare the supported forward kinds in `ForwarderModuleFactory` and have `ForwarderModule` return a `SpecForwarder` matching the request and input specs. Register the configuration `type` name with `RegisterForwarderModuleFactory`. Modules that need no change for a request can use `MakePassThroughForwarder`.

Builder calls the factory's `ReadConfigKeys` for every selected block and checks for unread keys. A module with configuration reads the values of its accepted keys as strings in this interface. The factory's `CreateModule` checks the types and ranges of configuration values and creates the module. Modules that check their combination with the forward kinds available across the whole sequence implement `ValidateForwardKinds`. This interface is called after `CreateModule` succeeds for every block.

`[forwarder]` tests registration and the public interface. `[perception]` tests the input and output specs of the four kinds, the round trip between history and Sample, statistics, configuration, and diagrams.
