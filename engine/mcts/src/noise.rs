use rand::prelude::*;
use rand_distr::Gumbel;

pub fn gumbel() -> f32 {
    let gumbel = Gumbel::new(0.0, 1.0).unwrap();
    gumbel.sample(&mut rand::rng())
}
