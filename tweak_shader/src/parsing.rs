use std::collections::BTreeMap;

use crate::input_type::{DiscreteInput, EventCode, InputType, ShaderBool, TextureStatus};
use std::fmt;

#[derive(Debug)]
pub enum Error {
    UnknownType(String),
    MalformedInput(String),
    MissingName(String),
    InvalidMaxSamples(String),
    InputPathNotString(String, String),
    InvalidPassDescriptor(String),
    MalformedPass(String, String),
    InvalidVersion(String),
    Parsing(String),
    MultipleUtilityBlocks,
    MalformedUtilityBlock(String),
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
                write!(f, "Invalid Tweak Version directive: {pragma}, must be #pragma tweak_version(version=<number>)")
            }
            Error::UnknownType(pragma) => {
                write!(f, "Unknown type in input directive: {pragma} must be one of color, float, int, event, point, image, audio, audiofft, bool.")
            }
            Error::MalformedInput(pragma) => {
                write!(f, "Malformed input directive in {pragma}: must start with one of color, float, int, event, point, image, audio, audiofft, bool.")
            }
            Error::MissingName(pragma) => {
                write!(f, "Missing name in input directive: {pragma}")
            }
            Error::InvalidMaxSamples(pragma) => {
                write!(f, "Input max_samples invalid: {}", pragma)
            }
            Error::InputPathNotString(pragma, path) => {
                write!(f, "Input Path is not a string: {}: path: {}", pragma, path)
            }
            Error::MalformedPass(pragma, e) => {
                write!(f, "Invalid pass descriptor {pragma}: {e}")
            }
            Error::InvalidPassDescriptor(pragma) => {
                write!(
                    f,
                    "Invalid pass directive {pragma}: needs an index specifier as the first field, such as #pragma pass(1)."
                )
            }
            Error::Parsing(msg) => {
                write!(f, "Error in input: {msg}")
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

#[derive(Debug, Clone, Default)]
pub struct SamplerDesc {
    filter_mode: wgpu::FilterMode,
    clamp_mode: wgpu::AddressMode,
}

#[derive(Debug, Clone)]
pub struct DocumentDescriptor {
    // it's an f32 for convenience sake
    pub version: f32,
    pub utility_block_name: Option<String>,
    pub stage: wgpu::naga::ShaderStage,
    pub inputs: BTreeMap<String, crate::input_type::InputType>,
    pub preloads: BTreeMap<String, String>,
    pub passes: Vec<RenderPass>,
    pub samplers: BTreeMap<String, SamplerDesc>,
}

pub fn parse_document(input: &str) -> Result<DocumentDescriptor, Error> {
    let mut desc = DocumentDescriptor {
        utility_block_name: None,
        stage: wgpu::naga::ShaderStage::Fragment,
        version: 1.0,
        passes: vec![],
        preloads: BTreeMap::new(),
        inputs: BTreeMap::new(),
        samplers: BTreeMap::new(),
    };

    let version = input
        .lines()
        .find_map(|line| line.trim().strip_prefix("#pragma version"));

    if let Some(rest) = version {
        let (_, list) = parse_qualifier_list(rest).map_err(|e| display_err(rest, e))?;

        if let Some(version) = seek::<f32>(&list, "version") {
            desc.version = version.map_err(|_| Error::InvalidVersion(rest.to_owned()))?;
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

    let passes = input
        .lines()
        .filter_map(|line| line.trim().strip_prefix("#pragma pass"));

    for pass in passes {
        let (_, list) = parse_qualifier_list(pass).map_err(|e| display_err(pass, e))?;
        match list.as_slice() {
            [(QVal::Num(pass_idx), None), rest @ ..] => {
                let pass = create_pass(rest, *pass_idx as usize)
                    .map_err(|e| Error::MalformedPass(pass.to_owned(), format!("{e}")))?;
                desc.passes.push(pass);
            }
            _ => {
                Err(Error::InvalidPassDescriptor(pass.into()))?;
            }
        }
    }

    let inputs = input
        .lines()
        .filter_map(|line| line.trim().strip_prefix("#pragma input"));

    for input in inputs {
        let (_, list) = parse_qualifier_list(input).map_err(|e| display_err(input, e))?;
        let bail = |e| map_input_err(e, input);

        let [(QVal::Id(type_id), None), rest @ ..] = list.as_slice() else {
            return Err(Error::MalformedInput(input.into()));
        };

        match type_id.as_str() {
            "color" => {
                let (name, color) = create_input(rest).map_err(bail)?;
                desc.inputs.insert(name, InputType::Color(color));
            }
            "float" => {
                let (name, float) = create_input(rest).map_err(bail)?;
                desc.inputs.insert(name, InputType::Float(float));
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
                desc.inputs.insert(name, InputType::Int(int, list));
            }
            "point" => {
                let (name, point) = create_input(rest).map_err(bail)?;
                desc.inputs.insert(name, InputType::Point(point));
            }
            "event" => {
                let (name, _) =
                    create_input::<EventCode, DiscreteInput<EventCode>>(rest).map_err(bail)?;
                desc.inputs.insert(
                    name,
                    InputType::Event(DiscreteInput {
                        default: 0,
                        current: 0,
                    }),
                );
            }
            "bool" => {
                let (name, booli) = create_input::<ShaderBool, _>(rest).map_err(bail)?;
                desc.inputs.insert(name, InputType::Bool(booli));
            }
            "image" => {
                let name = seek(rest, "name").ok_or(Error::MissingName(input.into()))?;

                let name: String = name.map_err(|e| map_input_err(format!("{e}"), input))?;

                desc.inputs
                    .insert(name.clone(), InputType::Image(TextureStatus::Uninit));

                let path = seek(rest, "path").transpose()?;

                if let Some(path) = path {
                    desc.preloads.insert(name, path);
                }
            }
            "audio" => {
                let name = seek(rest, "name").ok_or(Error::MissingName(input.into()))?;

                let name: String = name.map_err(|e| map_input_err(format!("{e}"), input))?;

                let samples = seek::<u32>(rest, "max_samples").transpose()?;

                desc.inputs.insert(
                    name.clone(),
                    InputType::Audio(TextureStatus::Uninit, samples),
                );

                let path = seek(rest, "path").transpose()?;

                if let Some(path) = path {
                    desc.preloads.insert(name, path);
                }
            }
            "audiofft" => {
                let name = seek(rest, "name").ok_or(Error::MissingName(input.into()))?;

                let name: String = name.map_err(|e| map_input_err(format!("{e}"), input))?;

                let samples = seek::<u32>(rest, "max_columns").transpose()?;

                desc.inputs.insert(
                    name.clone(),
                    InputType::AudioFft(TextureStatus::Uninit, samples),
                );

                let path = seek(rest, "path").transpose()?;

                if let Some(path) = path {
                    desc.preloads.insert(name, path);
                }
            }
            _ => {
                Err(Error::UnknownType(input.into()))?;
            }
        }
    }

    desc.passes.sort_by_key(|p| p.index);

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

impl TryFrom<QVal> for bool {
    type Error = Error;

    fn try_from(value: QVal) -> Result<Self, Self::Error> {
        if let QVal::Id(id) | QVal::String(id) = value {
            match id.as_str() {
                "false" => Ok(false),
                "true" => Ok(true),
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

            #pragma input(event, name="jim")
            
            #pragma input  (point , name="foo", max = [100.0, 200.0])

            #pragma pass(1, persistent, target="something")

            #pragma version(version = 3.0)
            
            #pragma utility_block("Temp")
            "#;

        let out = parse_document(pragma).unwrap();

        assert_eq!(out.inputs.len(), 6);

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
