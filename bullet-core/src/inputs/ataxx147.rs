use bulletformat::AtaxxBoard;

use super::InputType;

#[derive(Clone, Copy, Debug, Default)]
pub struct Ataxx147;
impl InputType for Ataxx147 {
    type RequiredDataType = AtaxxBoard;
    type FeatureIter = Ataxx147Iter;

    fn inputs(&self) -> usize {
        147
    }

    fn buckets(&self) -> usize {
        1
    }

    fn feature_iter(&self, pos: &Self::RequiredDataType) -> Self::FeatureIter {
        Ataxx147Iter { board_iter: pos.into_iter() }
    }
}

pub struct Ataxx147Iter {
    board_iter: <AtaxxBoard as std::iter::IntoIterator>::IntoIter,
}

impl Iterator for Ataxx147Iter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.board_iter.next().map(|(piece, square)| {
            let pc = usize::from(piece);
            let sq = usize::from(square);

            let stm_idx = 49 * pc + sq;
            let nstm_idx = if pc == 2 { stm_idx } else { 49 * (pc ^ 1) + sq };

            (stm_idx, nstm_idx)
        })
    }
}