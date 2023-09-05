use crate::{
    network::{Accumulator, Activation, NNUEParams, HIDDEN},
    position::{Features, Position},
    util::sigmoid,
};

pub fn gradients<Act: Activation>(
    positions: &[Position],
    nnue: &NNUEParams,
    error: &mut f32,
    blend: f32,
    skip_prop: f32,
    scale: f32,
) -> Box<NNUEParams> {
    let mut grad = NNUEParams::new();
    let mut rand = crate::rng::Rand::default();
    for pos in positions {
        if rand.rand(1.0) < skip_prop {
            continue;
        }

        update_single_grad::<Act>(pos, nnue, &mut grad, error, blend, scale);
    }
    grad
}

fn update_single_grad<Act: Activation>(
    pos: &Position,
    nnue: &NNUEParams,
    grad: &mut NNUEParams,
    error: &mut f32,
    blend: f32,
    scale: f32,
) {
    let bias = Accumulator::load_biases(nnue);
    let mut accs = [bias; 2];
    let mut activated = [[0.0; HIDDEN]; 2];
    let mut features = Features::default();

    let stm = pos.stm();

    let eval = nnue.forward::<Act>(pos, stm, &mut accs, &mut activated, &mut features);

    let result = pos.blended_result(blend, stm, scale);

    let sigmoid = sigmoid(eval, 1.0);
    let err = (sigmoid - result) * sigmoid * (1. - sigmoid);
    *error += (sigmoid - result).powi(2);

    nnue.backprop::<Act>(err, stm, grad, &accs, &activated, &mut features);
}