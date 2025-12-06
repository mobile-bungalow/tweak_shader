use crate::parsing::FromRanges;
use bytemuck::*;

macro_rules! extract {
    ($expression:expr, $(
        $(|)? $( $pattern:pat_param )|+ $( if $guard: expr )? => $output:expr
    ),+ $(,)?) => {
        match $expression {
            $($( $pattern )|+ $( if $guard )? => Some($output),)+
            _ => None
        }
    }
}

// In the future we should remove a lot of this code by using "derive_more"

/// RGBA 4 component color.
pub type Color = [f32; 4];

/// glsl 450 does not support booleans in uniforms
/// well, so this is our friendly type alias.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, NoUninit)]
pub enum ShaderBool {
    False = 0,
    True,
}

impl ShaderBool {
    pub fn is_true(&self) -> bool {
        matches!(self, ShaderBool::True)
    }
}

/// User defined event codes, reset to 0 at the start of the next render.
pub type EventCode = u32;

// For bounded inputs (Float, Int, Point)
#[derive(Debug, Clone)]
pub struct BoundedInput<T>
where
    T: Clone + Copy,
{
    pub current: T,
    pub min: T,
    pub max: T,
    pub default: T,
}

// For discrete inputs (Bool, Color, Event)
#[derive(Debug, Clone)]
pub struct DiscreteInput<T>
where
    T: Clone + Copy + PartialEq,
{
    pub current: T,
    pub default: T,
}

/// A wrapper around mutable values that will be written to the custom uniforms.
/// should be used when updating input values, i.e. when a user adjusts sliders
/// or a host program animates and input.
#[derive(Debug)]
pub struct MutInput<'a> {
    pub(crate) inner: &'a mut InputType,
}

impl<'a> MutInput<'a> {
    /// Returns the type of this variant
    pub fn variant(&self) -> InputVariant {
        InputVariant::from(&*self.inner)
    }

    /// if the variant is stored as a texture get the
    /// texture status
    pub fn texture_status(&mut self) -> Option<TextureStatus> {
        extract!(self.inner, InputType::Image(i) => *i)
    }

    /// Returns a reference to the internal f32 if the input is a float
    /// and none otherwise
    pub fn as_float(&mut self) -> Option<&mut BoundedInput<f32>> {
        extract!(self.inner, InputType::Float(i) => i)
    }

    /// Returns a reference to the internal i32 if the input is an int
    /// and none otherwise, includes and optional vec of labels for i32 values
    pub fn as_int(&mut self) -> Option<MutInputInt<'_>> {
        extract!(self.inner, InputType::Int(value, labels) => MutInputInt { value, labels})
    }

    /// Returns a reference to the internal point if the input is 2d point
    /// and none otherwise
    pub fn as_point(&mut self) -> Option<&mut BoundedInput<[f32; 2]>> {
        extract!(self.inner, InputType::Point(i) => i)
    }

    /// Returns a reference to the internal bool if the input is bool
    /// and none otherwise
    pub fn as_bool(&mut self) -> Option<&mut DiscreteInput<ShaderBool>> {
        extract!(self.inner, InputType::Bool(i) => i)
    }

    /// Returns a reference to the internal color if the variant is a color
    /// and none otherwise
    pub fn as_color(&mut self) -> Option<&mut DiscreteInput<Color>> {
        extract!(self.inner, InputType::Color(i) => i)
    }

    /// Returns a reference to a byte slice representing the
    /// binding, this can be used to access unsupported types
    /// such as matrices or arrays.
    pub fn as_unknown_bytes(&mut self) -> &mut [u8] {
        self.inner.as_bytes_mut()
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
            (InputType::Image(s), InputType::Image(o)) => {
                *o = *s;
                true
            }
            _ => false,
        }
    }
}

pub struct MutInputInt<'a> {
    pub value: &'a mut BoundedInput<i32>,
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

impl FromRanges<f32> for BoundedInput<f32> {
    fn from_ranges(min: Option<f32>, max: Option<f32>, default: Option<f32>) -> Self {
        let default_f32 = default.map_or(0.0, |v| v);
        let range = f32::abs(default_f32).max(1.0) * 10.0;
        let min_f32 = min.map_or(-range, |v| v);
        let max_f32 = max.map_or(range, |v| v);

        Self {
            current: default_f32,
            min: min_f32,
            max: max_f32,
            default: default_f32,
        }
    }
}

impl FromRanges<EventCode> for DiscreteInput<EventCode> {
    fn from_ranges(
        _min: Option<EventCode>,
        _max: Option<EventCode>,
        _default: Option<EventCode>,
    ) -> Self {
        Self {
            current: 0,
            default: 0,
        }
    }
}

impl FromRanges<i32> for BoundedInput<i32> {
    fn from_ranges(min: Option<i32>, max: Option<i32>, default: Option<i32>) -> Self {
        let default_i32 = default.unwrap_or(0);
        let range = default_i32.abs().max(1) * 10;
        let min_i32 = min.unwrap_or(-range);
        let max_i32 = max.unwrap_or(range);

        Self {
            current: default_i32,
            min: min_i32,
            max: max_i32,
            default: default_i32,
        }
    }
}

impl FromRanges<[f32; 2]> for BoundedInput<[f32; 2]> {
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

        Self {
            current: default_f32,
            min: min_x,
            max: max_x,
            default: default_f32,
        }
    }
}

impl FromRanges<ShaderBool> for DiscreteInput<ShaderBool> {
    fn from_ranges(
        _min: Option<ShaderBool>,
        _max: Option<ShaderBool>,
        default: Option<ShaderBool>,
    ) -> Self {
        let default_bool = default.unwrap_or(ShaderBool::False);

        Self {
            current: default_bool,
            default: default_bool,
        }
    }
}

impl FromRanges<[f32; 4]> for DiscreteInput<[f32; 4]> {
    fn from_ranges(
        _min: Option<[f32; 4]>,
        _max: Option<[f32; 4]>,
        default: Option<[f32; 4]>,
    ) -> Self {
        let default_f32 = default.unwrap_or([0.0, 0.0, 0.0, 1.0]);

        Self {
            current: default_f32,
            default: default_f32,
        }
    }
}

/// The kinds of inputs that can specified by the user.
#[derive(Debug, Clone)]
pub enum InputType {
    /// A float input declared with an input pragma in the document.
    /// ```text
    /// #pragma input(float, name="foo", max=1.0, min=0.0, default=0.05)
    /// ```
    Float(BoundedInput<f32>),
    /// A signed integer input declared with an input pragma.
    /// ```text
    /// #pragma input(int, name="bar", max=10, min=-10, default=3)
    /// #pragma input(int, name="bar", default=3, values=[1,2,3], labels=["options_1", "option_2", "option_3"])
    /// ```
    Int(BoundedInput<i32>, Option<Vec<(String, i32)>>),
    /// A vec2 representing a 2d point.
    /// ```text
    /// #pragma input(point, name="baz", max=[1.0, 1.0], min=[-1.0, -1.0], default=[0.0, 0.0])
    /// ```
    Point(BoundedInput<[f32; 2]>),
    /// An int uniform, implying a bool.
    /// ```text
    /// #pragma input(bool, name="qux", default=true)
    /// ```
    Bool(DiscreteInput<ShaderBool>),
    /// A vec4 representing an rgba color.
    /// ```text
    /// #pragma input(color, name="quux",  default=[0.0, 0.0, 0.0, 1.0])
    /// ```
    ///
    Color(DiscreteInput<Color>),
    /// A status handle indicating the state of an internally maintained texture
    /// ```text
    /// #pragma input(image, name="quux",  default=[0.0, 0.0, 0.0, 1.0])
    /// ```
    Image(TextureStatus),
    /// The default type of any uniform with no pragma specifying how to interpret it.
    RawBytes(RawBytes),
}

pub trait TryAsMut<T> {
    fn try_as_mut(&mut self) -> Option<&mut T>;
}

impl TryAsMut<f32> for InputType {
    fn try_as_mut(&mut self) -> Option<&mut f32> {
        extract!(self, InputType::Float(f) => &mut f.current)
    }
}

impl TryAsMut<i32> for InputType {
    fn try_as_mut(&mut self) -> Option<&mut i32> {
        extract!(self, InputType::Int(i, _) => &mut i.current)
    }
}

impl TryAsMut<[f32; 2]> for InputType {
    fn try_as_mut(&mut self) -> Option<&mut [f32; 2]> {
        extract!(self, InputType::Point(p) => &mut p.current)
    }
}

impl TryAsMut<ShaderBool> for InputType {
    fn try_as_mut(&mut self) -> Option<&mut ShaderBool> {
        extract!(self, InputType::Bool(b) => &mut b.current)
    }
}

impl TryAsMut<Color> for InputType {
    fn try_as_mut(&mut self) -> Option<&mut Color> {
        extract!(self, InputType::Color(c) => &mut c.current)
    }
}

impl InputType {
    pub fn is_stored_as_texture(&self) -> bool {
        matches!(self, Self::Image(_))
    }

    pub fn as_mut<T>(&mut self) -> Option<&mut T>
    where
        InputType: TryAsMut<T>,
    {
        self.try_as_mut()
    }

    /// Returns the status of a still image, audio, or video variable
    /// will always return some if `is_stored_as_texture` returns true.
    pub fn texture_status(&self) -> Option<&TextureStatus> {
        match self {
            Self::Image(s) => Some(s),
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
            Self::Color(v) => bytes_of(&v.current),
            Self::RawBytes(v) => v.inner.as_slice(),
            _ => &[],
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
            InputType::RawBytes(_) => "Raw Bytes",
        };
        write!(f, "{s}")
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
/// A variant used to indicate the inner type of a
/// [MutInput]
pub enum InputVariant {
    Float,
    Int,
    Point,
    Bool,
    Color,
    Image,
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
            InputType::RawBytes(_) => InputVariant::Bytes,
        }
    }
}
