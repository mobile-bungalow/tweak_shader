use crate::parsing::FromRanges;
use bytemuck::*;
use wgpu::naga;
use wgpu::naga::{ScalarKind, TypeInner};

/// A wrapper around mutable values that will be written to the custom uniforms.
/// should be used when updating input values, i.e. when a user adjusts sliders
/// or a host program animates and input.
#[derive(Debug)]
pub struct MutInput<'a> {
    inner: &'a mut InputType,
}

impl<'a> MutInput<'a> {
    pub(crate) fn new(inner: &'a mut InputType) -> Self {
        MutInput { inner }
    }

    /// Returns the type of this variant
    pub fn variant(&self) -> InputVariant {
        InputVariant::from(&*self.inner)
    }

    /// if the variant is stored as a texture get the
    /// texture status
    pub fn texture_status(&mut self) -> Option<TextureStatus> {
        match self.inner {
            InputType::Image(s) | InputType::Audio(s, _) | InputType::AudioFft(s, _) => Some(*s),
            _ => None,
        }
    }

    /// Get the number of maximum samples or buckets that the input
    /// asks for if it is audio or audiofft
    pub fn audio_samples(&mut self) -> Option<u32> {
        match self.inner {
            InputType::Audio(_, s) | InputType::AudioFft(_, s) => *s,
            _ => None,
        }
    }

    /// Returns a reference to the internal f32 if the input is a float
    /// and none otherwise
    pub fn as_float(&mut self) -> Option<&mut FloatInput> {
        if let InputType::Float(bounded_input) = self.inner {
            Some(bounded_input)
        } else {
            None
        }
    }

    /// Returns a reference to the internal i32 if the input is an int
    /// and none otherwise, includes and optional vec of labels for i32 values
    pub fn as_int(&mut self) -> Option<MutInputInt> {
        if let InputType::Int(value, labels) = self.inner {
            Some(MutInputInt { value, labels })
        } else {
            None
        }
    }

    /// Returns a reference to the internal point if the input is 2d point
    /// and none otherwise
    pub fn as_point(&mut self) -> Option<&mut PointInput> {
        if let InputType::Point(bounded_input) = self.inner {
            Some(bounded_input)
        } else {
            None
        }
    }

    /// Returns a reference to the internal bool if the input is bool
    /// and none otherwise
    pub fn as_bool(&mut self) -> Option<&mut BoolInput> {
        if let InputType::Bool(unbound_input) = self.inner {
            Some(unbound_input)
        } else {
            None
        }
    }

    /// Returns a reference to the internal color if the variant is a color
    /// and none otherwise
    pub fn as_color(&mut self) -> Option<&mut ColorInput> {
        if let InputType::Color(unbound_input) = self.inner {
            Some(unbound_input)
        } else {
            None
        }
    }

    /// Returns a reference to a byte slice representing the
    /// binding, this can be used to access unsupported types
    /// such as matrices or arrays.
    pub fn as_unknown_bytes(&mut self) -> &mut [u8] {
        self.inner.as_bytes_mut()
    }

    /// Returns a reference to the internal event if the variant is an event
    /// and none otherwise, events are 0 if high, 1 if not.
    pub fn as_event(&mut self) -> Option<&mut u32> {
        if let InputType::Event(value) = self.inner {
            Some(value)
        } else {
            None
        }
    }

    /// Copies `self` `other` other if `self`s current value respects
    /// Others boundaries. returns true if the copy happened
    pub fn copy_into(&mut self, other: &mut Self) -> bool {
        match (&mut self.inner, &mut other.inner) {
            (InputType::Int(s, s_list), InputType::Int(o, o_list)) => {
                if let (Some(s_list), Some(o_list)) = (s_list, o_list) {
                    // if the lists both contain s.current under the same name then copy
                    if s_list.iter().find(|(_, val)| s.current == *val)
                        == o_list.iter().find(|(_, val)| s.current == *val)
                    {
                        o.current = s.current;
                        return true;
                    }
                }

                if s.current <= o.max && s.current >= o.min && s.default == o.default {
                    o.current = s.current;
                    true
                } else {
                    false
                }
            }
            (InputType::Float(s), InputType::Float(o)) => {
                if s.current <= o.max && s.current >= o.min && s.default == o.default {
                    o.current = s.current;
                    true
                } else {
                    false
                }
            }
            (InputType::Point(s), InputType::Point(o)) => {
                if s.current <= o.max && s.current >= o.min && s.default == o.default {
                    o.current = s.current;
                    true
                } else {
                    false
                }
            }
            // no boundaries
            (InputType::Color(s), InputType::Color(o)) => {
                o.current = s.current;
                true
            }
            (InputType::Bool(s), InputType::Bool(o)) => {
                o.current = s.current;
                true
            }
            (InputType::Image(s), InputType::Image(o))
            | (InputType::Audio(s, _), InputType::Audio(o, _))
            | (InputType::AudioFft(s, _), InputType::AudioFft(o, _)) => {
                *o = *s;
                true
            }
            (InputType::Event(_), InputType::Event(_)) => true,
            _ => false,
        }
    }
}

pub struct MutInputInt<'a> {
    pub value: &'a mut IntInput,
    pub labels: &'a mut Option<Vec<(String, i32)>>,
}

pub type AudioFftInput = (TextureStatus, Option<u32>);

pub type AudioInput = (TextureStatus, Option<u32>);

#[derive(Debug, Clone, Copy, Default)]
/// The state of a texture maintained by this context
pub enum TextureStatus {
    /// The texture is not loaded in the uniforms
    #[default]
    Uninit,
    /// The texture is already loaded and has width and height
    Loaded { width: u32, height: u32 },
}

/// If a type is not specified by an
/// input pragma, it will be stored internally
/// as a [`Vec<u8>`], If you query it by name with
/// [crate::RenderContext::get_input_mut] you will only
/// be able to access an `&mut[u8]` the same size as the input type.
#[derive(Debug, Clone)]
pub struct RawBytes {
    pub inner: Vec<u8>,
}

/// A struct representing a float input declared in the document with
/// an input pragma.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct FloatInput {
    pub current: f32,
    pub min: f32,
    pub max: f32,
    pub default: f32,
}

impl FromRanges<f32> for FloatInput {
    fn from_ranges(min: Option<f32>, max: Option<f32>, default: Option<f32>) -> Self {
        let default_f32 = default.map_or(0.0, |v| v);
        let range = f32::abs(default_f32).max(1.0) * 10.0;
        let min_f32 = min.map_or(-range, |v| v);
        let max_f32 = max.map_or(range, |v| v);

        FloatInput {
            current: default_f32,
            min: min_f32,
            max: max_f32,
            default: default_f32,
        }
    }
}

/// A struct representing an int input.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct IntInput {
    pub current: i32,
    pub min: i32,
    pub max: i32,
    pub default: i32,
}

impl FromRanges<i32> for IntInput {
    fn from_ranges(min: Option<i32>, max: Option<i32>, default: Option<i32>) -> Self {
        let default_i32 = default.unwrap_or(0);
        let range = default_i32.abs().max(1) * 10;
        let min_i32 = min.unwrap_or(-range);
        let max_i32 = max.unwrap_or(range);

        IntInput {
            current: default_i32,
            min: min_i32,
            max: max_i32,
            default: default_i32,
        }
    }
}

/// A struct representing a 2D point.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct PointInput {
    pub current: [f32; 2],
    pub min: [f32; 2],
    pub max: [f32; 2],
    pub default: [f32; 2],
}

impl FromRanges<[f32; 2]> for PointInput {
    fn from_ranges(
        min: Option<[f32; 2]>,
        max: Option<[f32; 2]>,
        default: Option<[f32; 2]>,
    ) -> Self {
        let default_f32 = default.unwrap_or([0.0, 0.0]);
        let range_x = default_f32[0].abs().max(1.0) * 10.0;
        let range_y = default_f32[1].abs().max(1.0) * 10.0;

        let min_x = min.unwrap_or([-range_x, -range_y]);
        let max_x = max.unwrap_or([range_x, range_y]);

        PointInput {
            current: default_f32,
            min: min_x,
            max: max_x,
            default: default_f32,
        }
    }
}

/// A struct representing a bool input. Naga, the reflection
/// and compilation crate used by this library does not allow for
/// boolean values in uniform blocks. This type is only semantically
/// different from a u32.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct BoolInput {
    pub current: u32,
    pub default: u32,
}

impl FromRanges<bool> for BoolInput {
    fn from_ranges(_min: Option<bool>, _max: Option<bool>, default: Option<bool>) -> Self {
        let default_bool = if default.is_some_and(|b| b) { 1 } else { 0 };

        BoolInput {
            current: default_bool,
            default: default_bool,
        }
    }
}

/// A struct representing a color input.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct ColorInput {
    pub current: [f32; 4],
    pub default: [f32; 4],
}

impl FromRanges<[f32; 4]> for ColorInput {
    fn from_ranges(
        _min: Option<[f32; 4]>,
        _max: Option<[f32; 4]>,
        default: Option<[f32; 4]>,
    ) -> Self {
        let default_f32 = default.unwrap_or([0.0, 0.0, 0.0, 1.0]);

        ColorInput {
            current: default_f32,
            default: default_f32,
        }
    }
}

/// An event input, an event is meant to be a momentary boolean value,
/// It is only semantically different from a [BoolInput].
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct EventInput {
    pub value: u32,
}

impl FromRanges<()> for EventInput {
    fn from_ranges(_min: Option<()>, _max: Option<()>, _default: Option<()>) -> Self {
        EventInput { value: 0 }
    }
}

/// The kinds of inputs that can specified by the user.
#[derive(Debug, Clone)]
pub enum InputType {
    /// A float input declared with an input pragma in the document.
    /// ```text
    /// #pragma input(float, name="foo", max=1.0, min=0.0, default=0.05)
    /// ```
    Float(FloatInput),
    /// A signed integer input declared with an input pragma.
    /// ```text
    /// #pragma input(int, name="bar", max=10, min=-10, default=3)
    /// #pragma input(int, name="bar", default=3, values=[1,2,3], labels=["options_1", "option_2", "option_3"])
    /// ```
    Int(IntInput, Option<Vec<(String, i32)>>),
    /// A vec2 representing a 2d point.
    /// ```text
    /// #pragma input(point, name="baz", max=[1.0, 1.0], min=[-1.0, -1.0], default=[0.0, 0.0])
    /// ```
    Point(PointInput),
    /// An int uniform, implying a bool.
    /// ```text
    /// #pragma input(bool, name="qux", default=true)
    /// ```
    Bool(BoolInput),
    /// A vec4 representing an rgba color.
    /// ```text
    /// #pragma input(color, name="quux",  default=[0.0, 0.0, 0.0, 1.0])
    /// ```
    Color(ColorInput),
    /// A status handle indicating the state of an internally maintained texture
    /// ```text
    /// #pragma input(image, name="quux",  default=[0.0, 0.0, 0.0, 1.0])
    /// ```
    Image(TextureStatus),
    /// A status handle indicating the state of an internally maintained texture
    /// and an optional maximum number of samplers.
    ///```text
    /// #pragma input(audio, name="garply", path="./demo.mp4")
    ///````
    Audio(TextureStatus, Option<u32>),
    /// A status handle indicating the state of an internally maintained texture
    /// and an optional maximum number of fft buckets.
    ///```text
    /// #pragma input(audiofft, name="waldo")
    ///````
    AudioFft(TextureStatus, Option<u32>),
    /// An int uniform, implying a momentary bool.
    /// ```text
    /// #pragma input(event, name="Swampus")
    /// ```
    Event(u32),
    /// The default type of any uniform with no pragma specifying how to interpret it.
    RawBytes(RawBytes),
}

impl InputType {
    pub fn is_stored_as_texture(&self) -> bool {
        matches!(
            self,
            Self::Audio(_, _) | Self::Image(_) | Self::AudioFft(_, _)
        )
    }

    /// Returns the status of a still image, audio, or video variable
    /// will always return some if `is_stored_as_texture` returns true.
    pub fn texture_status(&self) -> Option<&TextureStatus> {
        match self {
            Self::Audio(s, _) | Self::Image(s) | Self::AudioFft(s, _) => Some(s),
            _ => None,
        }
    }

    /// Returns a mutable slice of any pod type, or the empty slice.
    /// This excludes the bool event, and image types.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Self::Float(v) => bytes_of_mut(&mut v.current),
            Self::Int(v, _) => bytes_of_mut(&mut v.current),
            Self::Point(v) => bytes_of_mut(&mut v.current),
            Self::Color(v) => bytes_of_mut(&mut v.current),
            Self::RawBytes(v) => v.inner.as_mut_slice(),
            _ => &mut [],
        }
    }

    /// Returns the bytes of any type kept in cpu memory,
    /// this excludes images and audio
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Float(v) => bytes_of(&v.current),
            Self::Int(v, _) => bytes_of(&v.current),
            Self::Point(v) => bytes_of(&v.current),
            Self::Bool(v) => bytes_of(&v.current),
            Self::Event(ev) => bytes_of(ev),
            Self::Color(v) => bytes_of(&v.current),
            Self::RawBytes(v) => v.inner.as_slice(),
            _ => &[],
        }
    }

    pub(crate) fn type_check_struct_member(&self, refl: &TypeInner) -> bool {
        match self {
            InputType::Float(_) => {
                *refl
                    == TypeInner::Scalar {
                        kind: ScalarKind::Float,
                        width: 4,
                    }
            }
            InputType::Int(_, _) => {
                *refl
                    == TypeInner::Scalar {
                        kind: ScalarKind::Sint,
                        width: 4,
                    }
            }
            InputType::Point(_) => {
                *refl
                    == TypeInner::Vector {
                        size: naga::VectorSize::Bi,
                        kind: ScalarKind::Float,
                        width: 4,
                    }
            }
            InputType::Bool(_) | InputType::Event(_) => {
                *refl
                    == TypeInner::Scalar {
                        kind: ScalarKind::Sint,
                        width: 4,
                    }
            }
            InputType::Color(_) => {
                *refl
                    == TypeInner::Vector {
                        size: naga::VectorSize::Quad,
                        kind: ScalarKind::Float,
                        width: 4,
                    }
            }
            InputType::Image(_)
            | InputType::Audio(_, _)
            | InputType::AudioFft(_, _)
            | InputType::RawBytes(_) => unreachable!(),
        }
    }
}

impl<'a> From<&'a mut InputType> for MutInput<'a> {
    fn from(val: &'a mut InputType) -> Self {
        MutInput { inner: val }
    }
}

impl std::fmt::Display for InputType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            InputType::Float(_) => "Float",
            InputType::Int(_, _) => "Int",
            InputType::Point(_) => "Point",
            InputType::Bool(_) => "Bool",
            InputType::Color(_) => "Color",
            InputType::Image(_) => "Image",
            InputType::Audio(_, _) => "Audio",
            InputType::AudioFft(_, _) => "AudioFft",
            InputType::Event(_) => "Event",
            InputType::RawBytes(_) => "Raw Bytes",
        };
        write!(f, "{s}")
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[repr(C)]
/// A variant used to indicate the inner type of a
/// [MutInput]
pub enum InputVariant {
    Float,
    Int,
    Point,
    Bool,
    Color,
    Image,
    Audio,
    AudioFft,
    Event,
    Bytes,
}

impl From<&InputType> for InputVariant {
    fn from(variant: &InputType) -> InputVariant {
        match variant {
            InputType::Float(_) => InputVariant::Float,
            InputType::Int(_, _) => InputVariant::Int,
            InputType::Point(_) => InputVariant::Point,
            InputType::Bool(_) => InputVariant::Bool,
            InputType::Color(_) => InputVariant::Color,
            InputType::Image(_) => InputVariant::Image,
            InputType::Audio(_, _) => InputVariant::Audio,
            InputType::AudioFft(_, _) => InputVariant::AudioFft,
            InputType::Event(_) => InputVariant::Event,
            InputType::RawBytes(_) => InputVariant::Bytes,
        }
    }
}
