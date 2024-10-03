extern crate pyo3;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::Dirichlet;
use crate::cc::rollouts::MCTSParams;


#[pyclass(module="alpha_cc_engine")]
#[derive(Clone)]
pub struct MCTSNode {
    #[pyo3(get)]
    pub n: Vec<u32>,
    #[pyo3(get)]
    pub q: Vec<f32>,
    #[pyo3(get)]
    pub pi: Vec<f32>,
    #[pyo3(get)]
    pub v: f32,
}


impl MCTSNode {
    pub fn update_on_visit(&mut self, action: usize, value: f32) {
        let n_a = self.n[action] as f32;
        self.n[action] += 1;
        self.q[action] = (n_a * self.q[action] + value) / (n_a + 1.0);
    }

    pub fn with_noised_pi(&self, mcts_params: &MCTSParams) -> MCTSNode {
        let weight = mcts_params.dirichlet_weight;
        let alpha = mcts_params.dirichlet_alpha;

        if weight == 0.0 || self.pi.len() == 1 {
            return self.clone();
        }
        
        let alpha_vec = self.pi.iter().map(|x| x * alpha).collect::<Vec<f32>>();
        match Dirichlet::new(&alpha_vec) {
            Ok(dirichlet) => {
                let noise = dirichlet.sample(&mut rand::thread_rng());
                let pi_noised = self.pi.iter()
                    .zip(noise.iter())
                    .map(|(p, n)| p * (1.0 - weight) + n * weight)
                    .collect();
                MCTSNode {
                    n: self.n.clone(),
                    q: self.q.clone(),
                    pi: pi_noised,
                    v: self.v,
                }
            },
            Err(e) => {
                println!("WARNING: Failed to create Dirichlet distribution: {}", e);
                self.clone()
            }
        }
    }
}
