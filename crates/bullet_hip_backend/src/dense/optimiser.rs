use bullet_core::backend::device::DeviceBuffer;

use crate::{
    backend::{ops, Buffer},
    DeviceError,
};

#[allow(clippy::too_many_arguments)]
pub fn adam(
    size: usize,
    params: &mut Buffer<f32>,
    gradient: &Buffer<f32>,
    momentum: &mut Buffer<f32>,
    velocity: &mut Buffer<f32>,
    beta1: f32,
    beta2: f32,
    gradient_factor: f32,
    learning_rate: f32,
    denom: bool,
) -> Result<(), DeviceError> {
    if size > params.size() || size > gradient.size() || size > momentum.size() || size > velocity.size() {
        return Err(DeviceError::ExpectedIllegalAddressAccess);
    }

    unsafe {
        ops::Adam(
            size,
            beta1,
            beta2,
            gradient_factor,
            learning_rate,
            denom,
            params.mut_ptr(),
            momentum.mut_ptr(),
            velocity.mut_ptr(),
            gradient.ptr(),
        );
    }

    Ok(())
}

pub fn clip(size: usize, params: &mut Buffer<f32>, min: f32, max: f32) -> Result<(), DeviceError> {
    if size > params.size() {
        return Err(DeviceError::ExpectedIllegalAddressAccess);
    }

    unsafe {
        ops::clip(size, params.mut_ptr(), min, max);
    }

    Ok(())
}
