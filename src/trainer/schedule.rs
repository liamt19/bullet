use std::ops::Rem;
use std::f32::consts::PI;
use crate::ansi;

#[derive(Clone, Debug)]
pub struct TrainingSchedule {
    pub net_id: String,
    pub eval_scale: f32,
    pub ft_regularisation: f32,
    pub batch_size: usize,
    pub batches_per_superbatch: usize,
    pub start_superbatch: usize,
    pub end_superbatch: usize,
    pub wdl_scheduler: WdlScheduler,
    pub lr_scheduler: LrScheduler,
    pub loss_function: Loss,
    pub save_rate: usize,
}

impl TrainingSchedule {
    pub fn net_id(&self) -> String {
        self.net_id.clone()
    }

    pub fn should_save(&self, superbatch: usize) -> bool {
        superbatch % self.save_rate == 0 || superbatch == self.end_superbatch
    }

    pub fn lr(&self, superbatch: usize) -> f32 {
        self.lr_scheduler.lr(superbatch)
    }

    pub fn wdl(&self, superbatch: usize) -> f32 {
        self.wdl_scheduler.blend(superbatch, self.end_superbatch)
    }

    pub fn display(&self) {
        println!("Scale                  : {}", ansi(format!("{:.0}", self.eval_scale), 31));
        println!("1 / FT Regularisation  : {}", ansi(format!("{:.0}", 1.0 / self.ft_regularisation), 31));
        println!("Batch Size             : {}", ansi(self.batch_size, 31));
        println!("Batches / Superbatch   : {}", ansi(self.batches_per_superbatch, 31));
        println!("Positions / Superbatch : {}", ansi(self.batches_per_superbatch * self.batch_size, 31));
        println!("Start Superbatch       : {}", ansi(self.start_superbatch, 31));
        println!("End Superbatch         : {}", ansi(self.end_superbatch, 31));
        println!("Save Rate              : {}", ansi(self.save_rate, 31));
        println!("WDL Scheduler          : {}", self.wdl_scheduler.colourful());
        println!("LR Scheduler           : {}", self.lr_scheduler.colourful());
    }

    pub fn power(&self) -> f32 {
        match self.loss_function {
            Loss::SigmoidMSE => 2.0,
            Loss::SigmoidMPE(x) => x,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Loss {
    SigmoidMSE,
    SigmoidMPE(f32),
}

#[derive(Clone, Copy, Debug)]
pub enum LrScheduler {
    /// Constant Rate
    Constant { value: f32 },
    /// Drop once at superbatch `drop`, by a factor of `gamma`.
    Drop { start: f32, gamma: f32, drop: usize },
    /// Drop every `step` superbatches by a factor of `gamma`.
    Step { start: f32, gamma: f32, step: usize },
    /// Drop every `step` superbatches by a factor of `gamma` after warming up for `warmup_batches` superbatches at a modified starting LR of `warmup_lr`.
    StepWithWarmup { start: f32, gamma: f32, step: usize, warmup_batches: usize, warmup_lr: f32 },
    /// Drop every `step` superbatches by a factor of `gamma`, resetting every (ciel((# resets) / 2) * step_size) superbatches.
    CosineAnnealing { start: f32, gamma: f32, step: usize },
}

impl LrScheduler {

    pub fn get_sdg_step(&self, superbatch: usize, step_size: usize) -> usize {
        return match superbatch {
            s if s < (step_size * 1) => step_size * 1,
            s if s < (step_size * 2) => step_size * 1,
            s if s < (step_size * 4) => step_size * 2,
            s if s < (step_size * 8) => step_size * 4,
            _ => step_size * 8,
        }
    }

    pub fn lr(&self, superbatch: usize) -> f32 {
        match *self {
            Self::Constant { value } => value,
            Self::Drop { start, gamma, drop } => {
                if superbatch > drop {
                    start * gamma
                } else {
                    start
                }
            }
            Self::Step { start, gamma, step } => {
                let steps = superbatch.saturating_sub(1) / step;
                start * gamma.powi(steps as i32)
            }
            Self::StepWithWarmup { start, gamma, step, warmup_batches, warmup_lr } => {
                if superbatch <= warmup_batches {
                    let steps = superbatch.saturating_sub(1) / step;
                    warmup_lr * gamma.powi(steps as i32)
                }
                else {
                    let actual_batch = superbatch - warmup_batches;
                    let steps = actual_batch.saturating_sub(1) / step;
                    start * gamma.powi(steps as i32)
                }
            }
            Self::CosineAnnealing { start, gamma, step } => {
                let sdg_step = self.get_sdg_step(superbatch, step);
                let decay = gamma.powi(superbatch as i32);
                let factor = PI * (superbatch.rem(sdg_step) as f32) / sdg_step as f32;
                let cosine = factor.cos();
                let min_val = 0.00001;
                0.5 * start * decay * (1.0 + min_val + cosine)
            }
        }
    }

    pub fn colourful(&self) -> String {
        match *self {
            Self::Constant { value } => format!("constant {}", ansi(value, 31)),
            Self::Drop { start, gamma, drop } => {
                format!("start {} gamma {} drop at {} superbatches", ansi(start, 31), ansi(gamma, 31), ansi(drop, 31),)
            }
            Self::Step { start, gamma, step } => {
                format!(
                    "start {} gamma {} drop every {} superbatches",
                    ansi(start, 31),
                    ansi(gamma, 31),
                    ansi(step, 31),
                )
            }
            Self::StepWithWarmup { start, gamma, step, warmup_batches, warmup_lr } => {
                format!(
                    "warmup {} for {} superbatches, start {} gamma {} drop every {} superbatches",
                    ansi(warmup_lr, 31),
                    ansi(warmup_batches, 31),
                    ansi(start, 31),
                    ansi(gamma, 31),
                    ansi(step, 31),
                )
            }
            Self::CosineAnnealing { start, gamma, step } => {
                format!(
                    "start {} gamma {} resets every {} superbatches",
                    ansi(start, 31),
                    ansi(gamma, 31),
                    ansi(step, 31),
                )
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum WdlScheduler {
    Constant { value: f32 },
    Linear { start: f32, end: f32 },
}

impl WdlScheduler {
    pub fn blend(&self, superbatch: usize, max: usize) -> f32 {
        match *self {
            Self::Constant { value } => value,
            Self::Linear { start, end } => {
                let grad = (end - start) / (max - 1).max(1) as f32;
                start + grad * (superbatch - 1) as f32
            }
        }
    }

    pub fn colourful(&self) -> String {
        match *self {
            Self::Constant { value } => format!("constant {}", ansi(value, 31)),
            Self::Linear { start, end } => {
                format!("linear taper start {} end {}", ansi(start, 31), ansi(end, 31))
            }
        }
    }
}
