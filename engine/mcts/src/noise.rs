use rand::prelude::*;
use rand_distr::Gumbel;
use rand_distr::multi::Dirichlet;

pub fn gumbel() -> f32 {
    let gumbel = Gumbel::new(0.0, 1.0).unwrap();
    gumbel.sample(&mut rand::rng())
}

/// Sample a Dirichlet vector whose concentration is `pi * alpha`. Returns
/// `pi.to_vec()` unchanged when there are fewer than 2 actions or sampling fails.
pub fn sample_dirichlet(pi: &[f32], alpha: f32) -> Vec<f32> {
    if pi.len() <= 1 {
        return pi.to_vec();
    }
    let conc: Vec<f32> = pi.iter()
        .map(|x| (x * alpha).max(f32::EPSILON))
        .collect();
    if let Ok(dist) = Dirichlet::new(&conc) {
        dist.sample(&mut rand::rng())
    } else {
        pi.to_vec()
    }
}

/// Linearly blend `noise` into `pi` in-place: pi = pi*(1-w) + noise*w
pub fn blend_with_noise(pi: &mut [f32], noise: &[f32], weight: f32) {
    for (p, n) in pi.iter_mut().zip(noise.iter()) {
        *p = *p * (1.0 - weight) + n * weight;
    }
}
