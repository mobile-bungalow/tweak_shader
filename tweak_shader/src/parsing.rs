use std::{collections::BTreeMap, str::FromStr};

use crate::input_type::{InputType, ShaderBool, TextureStatus};
use std::fmt;

// In the future I would like to replace the parsing module with a more principled parser
// for each pragma instead of the current adhoc version.

#[derive(Debug, Clone)]
pub struct Document {
    // it's an f32 for convenience sake
    pub version: f32,
    pub utility_block_name: Option<String>,
    pub stage: wgpu::naga::ShaderStage,
    pub inputs: BTreeMap<String, crate::input_type::InputType>,
    pub passes: Vec<RenderPass>,
    pub buffers: Vec<Buffer>,
    pub targets: Vec<Target>,
    pub samplers: Vec<SamplerDesc>,
}

impl Default for Document {
    fn default() -> Self {
        Document {
            utility_block_name: None,
            stage: wgpu::naga::ShaderStage::Fragment,
            version: 1.0,
            buffers: vec![],
            passes: vec![],
            targets: vec![],
            inputs: BTreeMap::new(),
            samplers: vec![],
        }
    }
}

#[derive(Debug)]
pub enum Error {
    UnknownType(String),
    MalformedInput(String),
    MissingName(String),
    InvalidPassDescriptor(String),
    InvalidSamplerDescriptor(String),
    InvalidTarget(String),
    InvalidBuffer(String),
    MissinBufferLength,
    MultipleScreenTargets,
    MalformedPass(String, String),
    InvalidVersion(String),
    Parsing(String),
    MultipleUtilityBlocks,
    MalformedUtilityBlock(String),
    StageSpecifier(String),
    MalformedIntList(String),
    UnexpectedType(String),
    Input(String, String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Input(pragma, error) => {
                write!(f, "Error Parsing input directive {pragma}: {error}")
            }
            Error::UnexpectedType(found) => {
                write!(f, "Type Error: {found}")
            }
            Error::MalformedIntList(pragma) => {
                write!(
                    f,
                    "Invalid Int Input {pragma}: List inputs must have a `values` and `labels` key of equal lengths."
                )
            }
            Error::MalformedUtilityBlock(pragma) => {
                write!(
                    f,
                    "Invalid utility block directive {pragma}: utility block should have one and only one argument, the utility block uniform name."
                )
            }
            Error::MultipleUtilityBlocks => {
                write!(
                    f,
                    "Utility block defined two or more times in the same document."
                )
            }
            Error::InvalidVersion(pragma) => {
                write!(f, "Invalid Tweak Version directive: #pragma version{pragma}, must be #pragma version(version=<number>)")
            }
            Error::UnknownType(pragma) => {
                write!(f, "Unknown type in input directive: #pragma input{pragma} must be one of color, float, int, event, point, image, audio, audiofft, bool.")
            }
            Error::MalformedInput(pragma) => {
                write!(f, "Malformed input directive in #pragma input{pragma}: must start with one of color, float, int, event, point, image, audio, audiofft, bool.")
            }
            Error::MissingName(pragma) => {
                write!(f, "Missing name in input directive: #pragma input{pragma}")
            }
            Error::MalformedPass(pragma, e) => {
                write!(f, "Invalid pass descriptor #pragma pass{pragma}: {e}")
            }
            Error::InvalidPassDescriptor(pragma) => {
                write!(
                    f,
                    "Invalid pass directive #pragma pass{pragma}: needs an index specifier as the first field, such as #pragma pass(1)."
                )
            }
            Error::InvalidSamplerDescriptor(pragma) => {
                write!(
                    f,
                    "Invalid sampler directive #pragma sampler{pragma}: must be of the form #pragma sampler(name='foo', linear|nearest, <clamp|repeat|mirror>)."
                )
            }
            Error::Parsing(msg) => {
                write!(f, "Error in input: {msg}")
            }
            Error::StageSpecifier(pragma) => {
                write!(f, "There must only be one stage spacifier of the form: #pragma stage('compute'|'fragment'), found #pragma stage{pragma}")
            }

            Error::InvalidTarget(pragma) => {
                write!(f, "Invalid target directive, must be of the form #pragma target(name='var_name', <persistent>, <height=n>, <width=n>), found #pragma target{pragma}")
            }
            Error::InvalidBuffer(pragma) => {
                write!(f, "Invalid buffer directive, must be of the form #pragma buffer(name='var_name', <persistent>, length=n), found #pragma target{pragma}")
            }

            Error::MissinBufferLength => {
                write!(f, "Buffer directive missing length attribute.")
            }
            Error::MultipleScreenTargets => {
                write!(f, "Multiple screen targets defined, only one target directive can use the screen keyword at a time.")
            }
        }
    }
}

fn map_input_err(err: String, line: &str) -> Error {
    Error::Input(line.to_owned(), err)
}

#[derive(Debug, PartialEq, Clone)]
pub enum QVal {
    Id(String),
    String(String),
    Num(f32),
    List(Vec<f32>),
    StringList(Vec<String>),
}

#[derive(Debug, Clone, Default)]
pub struct RenderPass {
    pub index: usize,
    // whether or not the buffer is cleared every render
    pub persistent: bool,
    // a height, or the render height as default
    pub width: Option<u32>,
    // a width, or the render height as default
    pub height: Option<u32>,
    // The texture variable name, if it exists
    pub target_texture: Option<String>,
}

impl FromStr for RenderPass {
    type Err = Error;

    fn from_str(pass: &str) -> Result<Self, Self::Err> {
        let (_, list) = parse_qualifier_list(pass).map_err(|e| display_err(pass, e))?;
        match list.as_slice() {
            [(QVal::Num(pass_idx), None), rest @ ..] => {
                let pass = create_pass(rest, *pass_idx as usize)
                    .map_err(|e| Error::MalformedPass(pass.to_owned(), format!("{e}")))?;
                Ok(pass)
            }
            _ => Err(Error::InvalidPassDescriptor(pass.into())),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Target {
    pub name: String,
    // a texture to copy this target into in future passes
    // this is useful because on the web textures are write only,
    // so this may be our only option for what would otherwise
    // be single pass renders.
    pub forward_target: Option<String>,
    // whether or not the buffer is cleared every render
    pub persistent: bool,
    // a height, or the render height as default
    pub width: Option<u32>,
    // a width, or the render height as default
    pub height: Option<u32>,
}

impl FromStr for Target {
    type Err = Error;

    fn from_str(target: &str) -> Result<Self, Self::Err> {
        let (_, list) = parse_qualifier_list(target).map_err(|e| display_err(target, e))?;

        let [(QVal::Id(name_literal), Some(QVal::String(name) | QVal::Id(name))), rest @ ..] =
            list.as_slice()
        else {
            return Err(Error::InvalidTarget(target.into()));
        };

        match (name_literal.as_str(), name, rest) {
            ("name", name, rest) => {
                let target = create_target(name, rest)?;
                Ok(target)
            }
            _ => Err(Error::InvalidTarget(target.into())),
        }
    }
}

struct InputEntry(String, InputType);

impl FromStr for InputEntry {
    type Err = Error;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let (_, list) = parse_qualifier_list(input).map_err(|e| display_err(input, e))?;
        let bail = |e| map_input_err(e, input);

        let [(QVal::Id(type_id), None), rest @ ..] = list.as_slice() else {
            return Err(Error::MalformedInput(input.into()));
        };

        match type_id.as_str() {
            "color" => {
                let (name, color) = create_input(rest).map_err(bail)?;
                Ok(InputEntry(name, InputType::Color(color)))
            }
            "float" => {
                let (name, float) = create_input(rest).map_err(bail)?;
                Ok(InputEntry(name, InputType::Float(float)))
            }
            "int" => {
                let (name, int) = create_input(rest).map_err(bail)?;

                let list = if let Some(labels) = seek::<Vec<String>>(rest, "labels") {
                    let labels = labels?;
                    if let Some(values) = seek::<Vec<i32>>(rest, "values") {
                        let values = values?;
                        Some(labels.into_iter().zip(values.into_iter()).collect())
                    } else {
                        Err(Error::MalformedIntList(input.to_owned()))?
                    }
                } else {
                    None
                };

                Ok(InputEntry(name, InputType::Int(int, list)))
            }
            "point" => {
                let (name, point) = create_input(rest).map_err(bail)?;
                Ok(InputEntry(name, InputType::Point(point)))
            }
            "bool" => {
                let (name, booli) = create_input::<ShaderBool, _>(rest).map_err(bail)?;
                Ok(InputEntry(name, InputType::Bool(booli)))
            }
            "image" => {
                let name = seek(rest, "name").ok_or(Error::MissingName(input.into()))?;

                let name: String = name.map_err(|e| map_input_err(format!("{e}"), input))?;

                Ok(InputEntry(name, InputType::Image(TextureStatus::Uninit)))
            }
            _ => Err(Error::UnknownType(input.into())),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Buffer {
    // name of the buffer variable
    pub name: String,
    // whether or not this is cleared between renders, (not passes)
    pub persistent: bool,
    // the default allocated length of the buffer
    pub length: u32,
}

impl FromStr for Buffer {
    type Err = Error;

    fn from_str(buffer: &str) -> Result<Self, Self::Err> {
        let (_, list) = parse_qualifier_list(buffer).map_err(|e| display_err(buffer, e))?;

        let [(QVal::Id(name_literal), Some(QVal::String(name) | QVal::Id(name))), rest @ ..] =
            list.as_slice()
        else {
            return Err(Error::InvalidTarget(buffer.into()));
        };

        match (name_literal.as_str(), name, rest) {
            ("name", name, rest) => {
                let buffer = create_buffer(name, rest)?;
                Ok(buffer)
            }
            _ => Err(Error::InvalidTarget(buffer.into())),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SamplerDesc {
    pub name: String,
    pub filter_mode: wgpu::FilterMode,
    pub clamp_mode: wgpu::AddressMode,
}

impl FromStr for SamplerDesc {
    type Err = Error;

    fn from_str(sampler: &str) -> Result<Self, Self::Err> {
        let (_, list) = parse_qualifier_list(sampler).map_err(|e| display_err(sampler, e))?;

        let [(QVal::Id(name_literal), Some(QVal::String(name) | QVal::Id(name))), rest @ ..] =
            list.as_slice()
        else {
            return Err(Error::InvalidSamplerDescriptor(sampler.into()));
        };

        match (name_literal.as_str(), name, rest) {
            ("name", name, rest) => {
                let mut filter_mode = wgpu::FilterMode::Nearest;
                let mut clamp_mode = wgpu::AddressMode::ClampToEdge;

                for item in rest {
                    let (QVal::Id(specifier), None) = item else {
                        return Err(Error::InvalidSamplerDescriptor(sampler.into()));
                    };

                    match specifier.as_str() {
                        "linear" => filter_mode = wgpu::FilterMode::Linear,
                        "nearest" => filter_mode = wgpu::FilterMode::Nearest,
                        "clamp" => clamp_mode = wgpu::AddressMode::ClampToEdge,
                        "repeat" => clamp_mode = wgpu::AddressMode::Repeat,
                        "mirror" => clamp_mode = wgpu::AddressMode::MirrorRepeat,
                        _ => {
                            return Err(Error::InvalidPassDescriptor(sampler.into()));
                        }
                    }
                }

                Ok(SamplerDesc {
                    name: name.to_owned(),
                    filter_mode,
                    clamp_mode,
                })
            }
            _ => Err(Error::InvalidPassDescriptor(sampler.into())),
        }
    }
}

pub fn parse_document(input: &str) -> Result<Document, Error> {
    let mut desc = Document::default();

    let version = input
        .lines()
        .find_map(|line| line.trim().strip_prefix("#pragma version"));

    if let Some(rest) = version {
        let (_, list) = parse_qualifier_list(rest).map_err(|e| display_err(rest, e))?;

        if let Some(version) = seek::<f32>(&list, "version") {
            desc.version = version.map_err(|_| Error::InvalidVersion(rest.to_owned()))?;
        }
    }

    let mut stage = input
        .lines()
        .filter_map(|line| line.trim().strip_prefix("#pragma stage"));

    if let Some(rest) = stage.next() {
        let (_, list) = parse_qualifier_list(rest).map_err(|e| display_err(rest, e))?;
        match list.as_slice() {
            [(QVal::Id(id), None), ..] => {
                match id.as_str() {
                    "fragment" => {
                        desc.stage = ShaderStage::Fragment;
                    }
                    "compute" => {
                        desc.stage = ShaderStage::Compute;
                    }
                    _ => {}
                };

                if !stage.next().is_none() {
                    Err(Error::StageSpecifier(rest.to_owned()))?;
                }
            }
            _ => {
                Err(Error::StageSpecifier(rest.to_owned()))?;
            }
        }
    }

    let mut utility_block = input
        .lines()
        .filter_map(|line| line.trim().strip_prefix("#pragma utility_block"));

    if let Some(rest) = utility_block.next() {
        let (_, list) = parse_qualifier_list(rest).map_err(|e| display_err(rest, e))?;
        match list.as_slice() {
            [(QVal::Id(id), None), ..] => {
                if utility_block.next().is_none() {
                    desc.utility_block_name = Some(id.clone());
                } else {
                    Err(Error::MultipleUtilityBlocks)?;
                }
            }
            _ => {
                Err(Error::MalformedUtilityBlock(rest.to_owned()))?;
            }
        }
    }

    for line in input.lines() {
        if let Some(pass) = line.trim().strip_prefix("#pragma pass") {
            desc.passes.push(pass.parse()?);
        }

        if let Some(buffer) = line.trim().strip_prefix("#pragma buffer") {
            desc.buffers.push(buffer.parse()?);
        }

        if let Some(target) = line.trim().strip_prefix("#pragma target") {
            desc.targets.push(target.parse()?);
        }

        if let Some(sampler) = line.trim().strip_prefix("#pragma sampler") {
            desc.samplers.push(sampler.parse()?);
        }

        if let Some(input) = line.trim().strip_prefix("#pragma input") {
            let InputEntry(name, entry) = input.parse()?;
            desc.inputs.insert(name, entry);
        }
    }

    Ok(desc)
}

pub fn seek<I: TryFrom<QVal, Error = Error>>(
    slice: &[(QVal, Option<QVal>)],
    id: &str,
) -> Option<Result<I, Error>> {
    for (key, val) in slice {
        let QVal::Id(identifier) = key else {
            continue;
        };

        if identifier != id {
            continue;
        }

        if let Some(value) = val {
            return Some(I::try_from(value.clone()));
        } else {
            return None;
        }
    }
    None
}

fn create_buffer(name: &String, slice: &[(QVal, Option<QVal>)]) -> Result<Buffer, Error> {
    let mut buffer = Buffer {
        name: name.to_owned(),
        persistent: false,
        length: 0,
    };

    buffer.length = seek::<u32>(slice, "length")
        .transpose()?
        .ok_or(Error::MissinBufferLength)?;

    buffer.persistent = slice
        .iter()
        .any(|(q, _)| matches!(q, QVal::Id(id) if id == "persistent"));

    Ok(buffer)
}

fn create_target(name: &String, slice: &[(QVal, Option<QVal>)]) -> Result<Target, Error> {
    let mut pass = Target {
        forward_target: None,
        name: name.to_owned(),
        persistent: false,
        width: None,
        height: None,
    };

    pass.height = seek::<u32>(slice, "height").transpose()?;
    pass.width = seek::<u32>(slice, "width").transpose()?;
    pass.forward_target = seek::<String>(slice, "forward").transpose()?;

    pass.persistent = slice
        .iter()
        .any(|(q, _)| matches!(q, QVal::Id(id) if id == "persistent"));

    Ok(pass)
}

fn create_pass(slice: &[(QVal, Option<QVal>)], index: usize) -> Result<RenderPass, Error> {
    let mut pass = RenderPass {
        index,
        persistent: false,
        width: None,
        height: None,
        target_texture: None,
    };

    pass.height = seek::<u32>(slice, "height").transpose()?;
    pass.width = seek::<u32>(slice, "width").transpose()?;
    pass.target_texture = seek::<String>(slice, "target").transpose()?;

    pass.persistent = slice
        .iter()
        .any(|(q, _)| matches!(q, QVal::Id(id) if id == "persistent"));

    Ok(pass)
}

fn create_input<I: TryFrom<QVal, Error = Error>, T: FromRanges<I>>(
    slice: &[(QVal, Option<QVal>)],
) -> Result<(String, T), String> {
    let mut name: Option<String> = None; // Initialize the name variable
    let mut min: Option<I> = None; // Initialize the name variable
    let mut max: Option<I> = None; // Initialize the name variable
    let mut default: Option<I> = None; // Initialize the name variable

    for (key, val) in slice {
        let QVal::Id(identifier) = key else { continue };
        let Some(val) = val.as_ref() else { continue };

        match identifier.as_str() {
            "min" => min = Some(I::try_from(val.clone()).map_err(|e| format!("{e}"))?),
            "max" => max = Some(I::try_from(val.clone()).map_err(|e| format!("{e}"))?),
            "default" => default = Some(I::try_from(val.clone()).map_err(|e| format!("{e}"))?),
            "name" => match val {
                QVal::Id(name_id) | QVal::String(name_id) => {
                    name = Some(name_id.clone());
                }
                _ => return Err("Name should be a string or quoted string'".to_string()),
            },
            _ => {}
        }
    }

    match name {
        Some(name_value) => Ok((name_value, T::from_ranges(min, max, default))),
        None => Err("name field not found".to_string()),
    }
}

pub trait FromRanges<I> {
    fn from_ranges(min: Option<I>, max: Option<I>, default: Option<I>) -> Self;
}

impl TryFrom<QVal> for () {
    type Error = Error;
    fn try_from(_: QVal) -> Result<Self, Self::Error> {
        Ok(())
    }
}

impl TryFrom<QVal> for ShaderBool {
    type Error = Error;

    fn try_from(value: QVal) -> Result<Self, Self::Error> {
        if let QVal::Id(id) | QVal::String(id) = value {
            match id.as_str() {
                "false" => Ok(ShaderBool::False),
                "true" => Ok(ShaderBool::True),
                _ => Err(Error::UnexpectedType(format!(
                    "expect value `true` or `false` found {id}"
                ))),
            }
        } else {
            Err(Error::UnexpectedType("Expected type bool".to_string()))
        }
    }
}

impl TryFrom<QVal> for Vec<String> {
    type Error = Error;

    fn try_from(value: QVal) -> Result<Self, Self::Error> {
        if let QVal::StringList(v) = value {
            Ok(v.clone())
        } else {
            Err(Error::UnexpectedType(
                "Expected list of strings".to_string(),
            ))
        }
    }
}

impl TryFrom<QVal> for String {
    type Error = Error;

    fn try_from(value: QVal) -> Result<Self, Self::Error> {
        if let QVal::Id(str) | QVal::String(str) = value {
            Ok(str.clone())
        } else {
            Err(Error::UnexpectedType("Expected type String".to_string()))
        }
    }
}

impl TryFrom<QVal> for Vec<i32> {
    type Error = Error;

    fn try_from(value: QVal) -> Result<Self, Self::Error> {
        if let QVal::List(v) = value {
            Ok(v.iter().map(|f| *f as i32).collect())
        } else {
            Err(Error::UnexpectedType(
                "Expected list of strings".to_string(),
            ))
        }
    }
}

impl TryFrom<QVal> for u32 {
    type Error = Error;

    fn try_from(value: QVal) -> Result<Self, Self::Error> {
        if let QVal::Num(num) = value {
            Ok(f32::abs(num) as u32)
        } else {
            Err(Error::UnexpectedType("Expected type Int".to_string()))
        }
    }
}

impl TryFrom<QVal> for i32 {
    type Error = Error;

    fn try_from(value: QVal) -> Result<Self, Self::Error> {
        if let QVal::Num(num) = value {
            Ok(num as i32)
        } else {
            Err(Error::UnexpectedType("Expected type Int".to_string()))
        }
    }
}

impl TryFrom<QVal> for f32 {
    type Error = Error;

    fn try_from(value: QVal) -> Result<Self, Self::Error> {
        if let QVal::Num(num) = value {
            Ok(num)
        } else {
            Err(Error::UnexpectedType("Expected type Float".to_string()))
        }
    }
}

impl TryFrom<QVal> for [f32; 2] {
    type Error = Error;

    fn try_from(value: QVal) -> Result<Self, Self::Error> {
        if let QVal::List(list) = value {
            if list.len() == 2 {
                Ok([list[0], list[1]])
            } else {
                Err(Error::UnexpectedType(
                    "Wrong number of elements in point list".to_string(),
                ))
            }
        } else {
            Err(Error::UnexpectedType(
                "Expected type Point ([f32 ; 2])".to_string(),
            ))
        }
    }
}

impl TryFrom<QVal> for [f32; 4] {
    type Error = Error;

    fn try_from(value: QVal) -> Result<Self, Self::Error> {
        if let QVal::List(list) = value {
            if list.len() == 4 {
                Ok([list[0], list[1], list[2], list[3]])
            } else {
                Err(Error::UnexpectedType(
                    "Wrong number of elements in color list".to_string(),
                ))
            }
        } else {
            Err(Error::UnexpectedType(
                "Expected type Color ([f32 ; 4])".to_string(),
            ))
        }
    }
}

use nom::{
    branch::alt,
    bytes::complete::{escaped, tag},
    character::complete::none_of,
    character::complete::{alpha1, alphanumeric1, char, multispace0, space0},
    combinator::{map, recognize},
    error::convert_error,
    error::VerboseError,
    multi::{many0_count, separated_list0},
    number::complete::float,
    sequence::{delimited, pair, preceded, separated_pair, tuple},
    IResult as DumbResult,
};
use wgpu::naga::ShaderStage;

pub type IResult<I, O, E = VerboseError<I>> = DumbResult<I, O, E>;

fn display_err(input: &str, e: nom::Err<VerboseError<&str>>) -> Error {
    let out = match e {
        nom::Err::Incomplete(_) => format!("pragma Incomplete: {input}"),
        nom::Err::Error(e) => convert_error(input, e),
        nom::Err::Failure(e) => convert_error(input, e),
    };
    Error::Parsing(out)
}

fn parse_qualifier_list(input: &str) -> IResult<&str, Vec<(QVal, Option<QVal>)>> {
    delimited(
        delimited(multispace0, char('('), multispace0),
        separated_list0(
            delimited(multispace0, char(','), multispace0),
            parse_qualifier,
        ),
        delimited(multispace0, char(')'), multispace0),
    )(input)
}

fn parse_identifier(input: &str) -> IResult<&str, QVal> {
    map(
        recognize(pair(
            alt((alpha1, tag("_"))),
            many0_count(alt((alphanumeric1, tag("_")))),
        )),
        |s: &str| QVal::Id(s.to_string()),
    )(input)
}

fn parse_floating_point(input: &str) -> IResult<&str, QVal> {
    map(float, QVal::Num)(input)
}

fn parse_quoted_string_inner(input: &str) -> IResult<&str, String> {
    let esc = escaped(none_of("\\\""), '\\', tag("\""));
    let esc_or_empty = alt((esc, tag("")));
    let res = delimited(tag("\""), esc_or_empty, tag("\""))(input)?;

    Ok((res.0, res.1.to_owned()))
}

fn parse_quoted_string(input: &str) -> IResult<&str, QVal> {
    let res = parse_quoted_string_inner(input)?;
    Ok((res.0, QVal::String(res.1)))
}

fn parse_string_list(input: &str) -> IResult<&str, QVal> {
    delimited(
        char('['),
        map(
            separated_list0(char(','), preceded(space0, parse_quoted_string_inner)),
            QVal::StringList,
        ),
        char(']'),
    )(input)
}

fn parse_list(input: &str) -> IResult<&str, QVal> {
    delimited(
        char('['),
        map(
            separated_list0(char(','), preceded(space0, float)),
            QVal::List,
        ),
        char(']'),
    )(input)
}

fn parse_value(input: &str) -> IResult<&str, QVal> {
    alt((
        parse_floating_point,
        parse_quoted_string,
        parse_identifier,
        parse_list,
        parse_string_list,
    ))(input)
}

fn parse_key_name(input: &str) -> IResult<&str, QVal> {
    map(
        alt((parse_identifier, parse_quoted_string)),
        |out| match out {
            QVal::Id(id) | QVal::String(id) => QVal::Id(id),
            _ => unreachable!(),
        },
    )(input)
}

//<name> = <number | string>
fn parse_key_value(input: &str) -> IResult<&str, (QVal, Option<QVal>)> {
    separated_pair(
        parse_identifier,
        tuple((multispace0, char('='), multispace0)),
        parse_value,
    )(input)
    .map(|(rest, (key, value))| (rest, (key, Some(value))))
}

fn parse_qualifier(input: &str) -> IResult<&str, (QVal, Option<QVal>)> {
    alt((
        parse_key_value,
        map(parse_key_name, |s| (s, None)),
        map(parse_floating_point, |s| (s, None)),
    ))(input)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_parse_qualifier_list_single_qualifier() {
        let input = "(name = 42)";
        let expected = vec![(QVal::Id("name".to_string()), Some(QVal::Num(42.0)))];
        assert_eq!(parse_qualifier_list(input), Ok(("", expected)));
    }

    #[test]
    fn test_parse_qualifier_list_malformed_qualifiers() {
        let input = "(name = 2.14, =, key = \"Value\" )";
        assert!(parse_qualifier_list(input).is_err());
    }
    #[test]
    fn test_parse_identifier() {
        let input = "abc";
        let expected = QVal::Id("abc".to_string());
        assert_eq!(parse_identifier(input), Ok(("", expected)));
    }

    #[test]
    fn test_parse_floating_point() {
        let input = "2.14";
        let expected = QVal::Num(2.14);
        assert_eq!(parse_floating_point(input), Ok(("", expected)));
    }

    #[test]
    fn test_parse_quoted_string() {
        let input = r#""Hello, World!""#;
        let expected = QVal::String("Hello, World!".to_string());
        assert_eq!(parse_quoted_string(input), Ok(("", expected)));
    }

    #[test]
    fn test_parse_list() {
        let input = "[1.2, 3.4, 5.6]";
        let expected = QVal::List(vec![1.2, 3.4, 5.6]);
        assert_eq!(parse_list(input), Ok(("", expected)));
    }

    #[test]
    fn test_parse_key_value() {
        let input = "name = 42";
        let expected = (QVal::Id("name".to_string()), Some(QVal::Num(42.0)));
        assert_eq!(parse_key_value(input), Ok(("", expected)));
    }

    #[test]
    fn test_parse_qualifier() {
        let input = "key = 3.33";
        let expected = (QVal::Id("key".to_string()), Some(QVal::Num(3.33)));
        assert_eq!(parse_qualifier(input), Ok(("", expected)));
    }

    #[test]
    fn test_parse_qualifier_no_value() {
        let input = "key";
        let expected = (QVal::Id("key".to_string()), None);
        assert_eq!(parse_qualifier(input), Ok(("", expected)));
    }

    #[test]
    fn test_parse_qualifier_no_value_with_space() {
        let input = "key ";
        let expected = (QVal::Id("key".to_string()), None);
        assert_eq!(parse_qualifier(input), Ok((" ", expected)));
    }

    #[test]
    fn test_parse_qualifier_list() {
        let input = "(name = 42, abc, def = \"Hello\")";
        let expected = vec![
            (QVal::Id("name".to_string()), Some(QVal::Num(42.0))),
            (QVal::Id("abc".to_string()), None),
            (
                QVal::Id("def".to_string()),
                Some(QVal::String("Hello".to_string())),
            ),
        ];
        assert_eq!(parse_qualifier_list(input), Ok(("", expected)));
    }

    #[test]
    fn test_parse_qualifier_list_whitepsace() {
        let input = "  \t\t\t (version = 1.0, def = \"JAMBO\")";
        let expected = vec![
            (QVal::Id("version".to_string()), Some(QVal::Num(1.0))),
            (
                QVal::Id("def".to_string()),
                Some(QVal::String("JAMBO".to_string())),
            ),
        ];
        assert_eq!(parse_qualifier_list(input), Ok(("", expected)));
    }

    #[test]
    fn simple_parse_doc() {
        let pragma = "#pragma input(float, name=test)";
        parse_document(pragma).unwrap();

        let pragma = "#pragma sampler(name=josh)";
        parse_document(pragma).unwrap();

        let pragma = "#pragma sampler(name=josh, linear)";
        parse_document(pragma).unwrap();

        let pragma = "#pragma stage(name=josh)";
        assert!(parse_document(pragma).is_err());

        let pragma = "#pragma buffer(name=josh)";
        assert!(parse_document(pragma).is_err());

        let pragma = "#pragma pass(1, height=1)";
        parse_document(pragma).unwrap();

        let pragma = "#pragma buffer(name=sliminy, length=1)";
        parse_document(pragma).unwrap();

        let pragma = "#pragma buffer(name=sliminy, length=1, persistent)";
        parse_document(pragma).unwrap();

        let pragma = "#pragma stage(fragment)";
        parse_document(pragma).unwrap();

        let pragma = "#pragma stage(compute)";
        parse_document(pragma).unwrap();

        let pragma = "#pragma target(name=done_2, screen, height=100, width=100)";
        parse_document(pragma).unwrap();

        let pragma = "#pragma sampler(name=josh, fake)";
        assert!(parse_document(pragma).is_err());

        let pragma = "#pragma input(float, name=\"test\")";
        parse_document(pragma).unwrap();

        let pragma = "#pragma input(float, name=\"MessedUp_dumb-identifier1992\")";
        parse_document(pragma).unwrap();

        let pragma = "#pragma input(gobingo, name=\"test\")";
        assert!(parse_document(pragma).is_err());

        let pragma = "#pragma input(float, missing_name)";
        assert!(parse_document(pragma).is_err());

        let pragma = "#pragma utility_block(identifier)";
        parse_document(pragma).unwrap();

        let pragma = "#pragma utility_block(strang)\n#pragma utility_block(crusty)";
        assert!(parse_document(pragma).is_err());
    }

    #[test]
    fn doc_validation() {
        let pragma = r#"
            #pragma input(float, name=test)

            #pragma input(image, name=another)
            
            #pragma input(color, name=third, default = [1.0, 0.0, 0.0, 3.0])

            #pragma input(int, name="good", max=100, min=200, default=3)
 
            #pragma input  (point , name="foo", max = [100.0, 200.0])

            #pragma pass(1, persistent, target="something")

            #pragma version(version = 3.0)
            
            #pragma utility_block("Temp")
            "#;

        let out = parse_document(pragma).unwrap();

        assert_eq!(out.inputs.len(), 5);

        assert!(matches!(
            out.inputs.get("foo").unwrap(),
            InputType::Point(crate::input_type::BoundedInput { .. })
        ));

        assert!(matches!(
            out.inputs.get("third").unwrap(),
            InputType::Color(crate::input_type::DiscreteInput { .. })
        ));

        assert_eq!(out.version, 3.0);

        assert_eq!(out.passes.len(), 1);

        assert_eq!(out.utility_block_name, Some("Temp".to_string()));
    }
}
