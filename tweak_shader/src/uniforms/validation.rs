use crate::input_type::InputType;
use crate::parsing::Document;
use naga::{ScalarKind, TypeInner};
use wgpu::naga;

use super::{BindingEntry, Error, GlobalData, PushConstant, Uniforms};

impl Uniforms {
    pub fn validate(&self, document: &Document) -> Result<(), Error> {
        // Look for input pragmas that are missing bindings
        let missing_input: Vec<_> = document
            .inputs
            .keys()
            .filter(|key| {
                let not_in_push_constants =
                    if let Some(PushConstant::Struct { inputs, .. }) = &self.push_constants {
                        !inputs.contains_key(*key)
                    } else {
                        true
                    };

                let not_in_uni = self.sets.iter().all(|binding| {
                    self.lookup_table
                        .get(*key)
                        .map(|addr| binding.get(key, addr))
                        .is_none()
                });
                not_in_push_constants && not_in_uni
            })
            .cloned()
            .collect();

        // error out early if a specified input does not exists
        if !missing_input.is_empty() {
            Err(Error::MissingInput(missing_input))?;
        }

        // look for storage buffer targets that are missing bindings
        let missing_targets: Vec<_> = document
            .targets
            .iter()
            .filter_map(|target| {
                let found = self.sets.iter().find(|binding| {
                    binding.binding_entries.iter().any(|entry| {
                        if let BindingEntry::StorageTexture { name, .. } = entry {
                            *name == target.name
                        } else {
                            false
                        }
                    })
                });

                match found {
                    None => Some(target.name.clone()),
                    Some(_) => None,
                }
            })
            .collect();

        if !missing_targets.is_empty() {
            Err(Error::MissingTarget(missing_targets))?;
        }

        let no_util_present = !self.sets.iter().any(|b| b.contains_util());
        let push_is_util = self
            .push_constants
            .as_ref()
            .is_some_and(|p| matches!(p, PushConstant::UtilityBlock { .. }));

        // error out early a utility block was specified and not found
        if document.utility_block_name.is_some() && no_util_present && !push_is_util {
            Err(Error::UtilityBlockMissing(
                document.utility_block_name.as_ref().unwrap().clone(),
            ))?;
        }

        Ok(())
    }
}

use bytemuck::offset_of;

impl GlobalData {
    pub fn validate(it: impl Iterator<Item = (naga::TypeInner, usize)>) -> Result<(), Error> {
        let f32_ty = naga::TypeInner::Scalar(naga::Scalar {
            width: 4,
            kind: naga::ScalarKind::Float,
        });
        let u32_ty = naga::TypeInner::Scalar(naga::Scalar {
            width: 4,
            kind: naga::ScalarKind::Uint,
        });

        let vec4_ty = naga::TypeInner::Vector {
            scalar: naga::Scalar {
                kind: naga::ScalarKind::Float,
                width: 4,
            },
            size: naga::VectorSize::Quad,
        };
        let vec3 = naga::TypeInner::Vector {
            scalar: naga::Scalar {
                kind: naga::ScalarKind::Float,
                width: 4,
            },
            size: naga::VectorSize::Tri,
        };
        let self_iter = [
            (f32_ty.clone(), offset_of!(GlobalData, time)),
            (f32_ty.clone(), offset_of!(GlobalData, time_delta)),
            (f32_ty.clone(), offset_of!(GlobalData, frame_rate)),
            (u32_ty.clone(), offset_of!(GlobalData, frame)),
            (vec4_ty.clone(), offset_of!(GlobalData, mouse)),
            (vec4_ty.clone(), offset_of!(GlobalData, date)),
            (vec3, offset_of!(GlobalData, resolution)),
            (u32_ty.clone(), offset_of!(GlobalData, pass_index)),
        ];
        if self_iter.iter().zip(it).all(|(a, b)| *a == b) {
            Ok(())
        } else {
            Err(Error::UtilityBlockType)
        }
    }
}

impl InputType {
    pub(crate) fn validate(&self, reflection: &TypeInner) -> Result<(), Error> {
        match (self, reflection) {
            (
                InputType::Float(_),
                TypeInner::Scalar(naga::Scalar {
                    kind: ScalarKind::Float,
                    width: 4,
                }),
            ) => Ok(()),
            (
                InputType::Int(_, _),
                TypeInner::Scalar(naga::Scalar {
                    kind: ScalarKind::Sint,
                    width: 4,
                }),
            ) => Ok(()),
            (
                InputType::Point(_),
                TypeInner::Vector {
                    scalar:
                        naga::Scalar {
                            kind: ScalarKind::Float,
                            width: 4,
                        },
                    size: naga::VectorSize::Bi,
                },
            ) => Ok(()),
            (
                InputType::Bool(_),
                TypeInner::Scalar(naga::Scalar {
                    kind: ScalarKind::Sint,
                    width: 4,
                }),
            ) => Ok(()),
            (
                InputType::Color(_),
                TypeInner::Vector {
                    scalar:
                        naga::Scalar {
                            kind: ScalarKind::Float,
                            width: 4,
                        },
                    size: naga::VectorSize::Quad,
                },
            ) => Ok(()),
            (user_pragma, naga) => Err(Error::InputTypeErr(
                format!("{user_pragma}"),
                format!("{naga:?}"),
            )),
        }
    }
}
