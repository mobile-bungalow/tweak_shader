use crate::input_type::InputType;
use crate::parsing::Document;
use bytemuck::offset_of;
use naga::{ScalarKind, TypeInner};
use wgpu::naga;

use super::{BindingEntry, Error, GlobalData, PushConstant, Uniforms};

impl Uniforms {
    pub fn validate(&self, document: &Document, format: &wgpu::TextureFormat) -> Result<(), Error> {
        self.validate_compute_passes(document)?;
        self.validate_missing_inputs(document)?;
        self.validate_missing_targets(document)?;
        self.validate_target_format_mismatch(format)?;
        self.validate_missing_buffers(document)?;
        self.validate_utility_block(document)?;
        Ok(())
    }

    fn validate_compute_passes(&self, document: &Document) -> Result<(), Error> {
        if document.stage == wgpu::naga::ShaderStage::Compute {
            if let Some(i) = document
                .passes
                .iter()
                .position(|pass| !pass.is_compute_compatible())
            {
                return Err(Error::ComputePass(i));
            }
        }
        Ok(())
    }

    fn validate_missing_inputs(&self, document: &Document) -> Result<(), Error> {
        let missing_input: Vec<_> = document
            .inputs
            .keys()
            .filter(|key| self.is_input_missing(key))
            .cloned()
            .collect();

        if !missing_input.is_empty() {
            return Err(Error::MissingInput(missing_input));
        }
        Ok(())
    }

    fn is_input_missing(&self, key: &str) -> bool {
        let not_in_push_constants = match &self.push_constants {
            Some(PushConstant::Struct { inputs, .. }) => !inputs.contains_key(key),
            _ => true,
        };

        let not_in_uni = self.sets.iter().all(|binding| {
            self.lookup_table
                .get(key)
                .map(|addr| binding.get(key, addr))
                .is_none()
        });

        not_in_push_constants && not_in_uni
    }

    fn validate_missing_targets(&self, document: &Document) -> Result<(), Error> {
        let missing_targets: Vec<_> = document
            .targets
            .iter()
            .filter_map(|target| self.find_missing_target(target))
            .collect();

        if !missing_targets.is_empty() {
            return Err(Error::MissingTarget(missing_targets));
        }
        Ok(())
    }

    fn find_missing_target(&self, target: &crate::parsing::Target) -> Option<String> {
        let found = self.sets.iter().any(|binding| {
            binding.binding_entries.iter().any(|entry| {
            matches!(entry, BindingEntry::StorageTexture { name, .. } if *name == target.name)
        })
        });
        if !found {
            Some(target.name.clone())
        } else {
            None
        }
    }

    fn validate_target_format_mismatch(&self, format: &wgpu::TextureFormat) -> Result<(), Error> {
        let mismatch_target_textures: Vec<_> = self
            .sets
            .iter()
            .flat_map(|binding| binding.binding_entries.iter())
            .filter_map(|b| {
                if let BindingEntry::StorageTexture {
                    tex,
                    name,
                    purpose: super::StorageTexturePurpose::Target { .. },
                    ..
                } = b
                {
                    Some((tex.format(), name.clone()))
                } else {
                    None
                }
            })
            .filter(|(fmt, _name)| fmt != format)
            .collect();

        if !mismatch_target_textures.is_empty() {
            return Err(Error::TargetFormatMismatch(
                mismatch_target_textures,
                *format,
            ));
        }
        Ok(())
    }

    fn validate_missing_buffers(&self, document: &Document) -> Result<(), Error> {
        let missing_buffers: Vec<_> = document
            .buffers
            .iter()
            .filter_map(|target| self.find_missing_buffer(target))
            .collect();

        if !missing_buffers.is_empty() {
            return Err(Error::MissingBuffer(missing_buffers));
        }
        Ok(())
    }

    fn find_missing_buffer(&self, target: &crate::parsing::Buffer) -> Option<String> {
        let found = self.sets.iter().any(|binding| {
            binding.binding_entries.iter().any(|entry| {
            matches!(entry, BindingEntry::StorageTexture { name, .. } if *name == target.name)
        })
        });
        if !found {
            Some(target.name.clone())
        } else {
            None
        }
    }

    fn validate_utility_block(&self, document: &Document) -> Result<(), Error> {
        let no_util_present = !self.sets.iter().any(|b| b.contains_util());
        let push_is_util = self
            .push_constants
            .as_ref()
            .is_some_and(|p| matches!(p, PushConstant::UtilityBlock { .. }));

        if document.utility_block_name.is_some() && no_util_present && !push_is_util {
            return Err(Error::UtilityBlockMissing(
                document.utility_block_name.as_ref().unwrap().clone(),
            ));
        }
        Ok(())
    }
}

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
