use crate::{
    backend::ExecutionContext,
    tensor::{DenseMatrix, Shape, Tensor},
};

pub fn output_tensor(m: usize, inputs: &[Shape]) -> Result<Shape, String> {
    if inputs.len() == 2 {
        if inputs[0] == inputs[1] {
            if inputs[0].cols() == 1 {
                if inputs[0].rows() % m == 0 {
                    let inp = Shape::new(m, inputs[0].rows() / m);
                    let out = inp.transpose() * inp;
                    Ok(Shape::new(out.size(), 1))
                } else {
                    Err(format!("Input vector ({}) must have dimension divisible by {m}!", inputs[0]))
                }
            } else {
                Err("Input must be a vector!".to_string())
            }
        } else {
            Err(format!("Inputs must have same shape! {} != {}", inputs[0], inputs[1]))
        }
    } else {
        Err(format!("Invalid number of inputs in linear! Expected 2, got {}", inputs.len()))
    }
}

pub fn forward(ctx: &mut ExecutionContext, m: usize, inputs: &[&Tensor], output: &mut Tensor) {
    DenseMatrix::submatrix_product(
        ctx,
        m,
        inputs[0].values.dense(),
        inputs[1].values.dense(),
        output.values.dense_mut(),
    );
}

pub fn backprop(ctx: &mut ExecutionContext, m: usize, output: &Tensor, inputs: &mut [&mut Tensor]) {
    let (input1, input2) = inputs.split_at_mut(1);

    DenseMatrix::backprop_submatrix_product(
        ctx,
        m,
        input1[0].values.dense(),
        input1[0].gradients.as_mut(),
        input2[0].values.dense(),
        input2[0].gradients.as_mut(),
        output.gradients.as_ref().unwrap(),
    );
}