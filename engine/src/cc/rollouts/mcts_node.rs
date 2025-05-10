extern crate pyo3;

use pyo3::prelude::*;


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
    pub fn new_leaf(pi: Vec<f32>, v: f32) -> Self {
        let num_actions = pi.len();
        let n = vec![0; num_actions];
        let q = vec![0.0; num_actions];
        MCTSNode { n, q, pi, v }
    }

    pub fn update_on_visit(&mut self, action: usize, value: f32) {
        let n_a = self.n[action] as f32;
        self.n[action] += 1;
        self.q[action] = (n_a * self.q[action] + value) / (n_a + 1.0);
    }
}
