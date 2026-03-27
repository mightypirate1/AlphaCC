use rand::prelude::*;
use rand_distr::multi::Dirichlet;

use crate::cc::rollouts::mcts::MCTSParams;

/// Sample Dirichlet noise and blend it into `pi` in-place.
/// Returns the original `pi` unmodified if there are fewer than 2 actions.
pub fn apply_dirichlet_noise(pi: &mut [f32], params: &MCTSParams) {
    if let Ok(noise) = generate_dirichlet_noise(pi, params) {
        blend_with_noise(pi, &noise, params.dirichlet_weight);
    }
}

pub fn blend_with_noise(pi: &mut [f32], noise: &[f32], weight: f32) {
    for (p, n) in pi.iter_mut().zip(noise.iter()) {
        *p = *p * (1.0 - weight) + n * weight;
    }
}

pub fn generate_dirichlet_noise(pi: &[f32], params: &MCTSParams) -> anyhow::Result<Vec<f32>> {
    if params.dirichlet_weight <= 0.0 || pi.len() <= 1 {
        return Ok(pi.to_vec());
    }
    let alpha: Vec<f32> = pi.iter()
        .map(|x| (x * params.dirichlet_alpha).max(f32::EPSILON))
        .collect();
    if let Ok(dirichlet) = Dirichlet::new(&alpha) {
        let noise = dirichlet.sample(&mut rand::rng());
        Ok(noise)
    } else {
        Err(anyhow::anyhow!("Failed to generate Dirichlet noise"))
    }
}
