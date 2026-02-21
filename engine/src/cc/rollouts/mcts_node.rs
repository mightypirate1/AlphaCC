extern crate pyo3;

use pyo3::prelude::*;

use crate::cc::Move;
use crate::cc::dtypes::{NNQuantizedPi, NNQuantizedValue};


#[pyclass(module="alpha_cc_engine", from_py_object)]
#[derive(Clone)]
pub struct MCTSNode {
    #[pyo3(get)]
    pub n: Vec<u32>,
    pub q: Vec<NNQuantizedValue>,
    pub pi: Vec<NNQuantizedPi>,
    pub v: NNQuantizedValue,
    #[pyo3(get)]
    pub moves: Vec<Move>,
}


impl MCTSNode {
    pub fn new_leaf(pi: Vec<f32>, v: f32, moves: Vec<Move>) -> Self {
        let num_actions = pi.len();
        let n = vec![0; num_actions];
        let q = vec![NNQuantizedValue::quantize(0.0); num_actions];
        MCTSNode {
            n,
            q,
            pi: NNQuantizedPi::quantize_vec(&pi),
            v: NNQuantizedValue::quantize(v),
            moves,
        }
    }

    pub fn get_pi(&self, action: usize) -> f32 {
        self.pi[action].dequantize()
    }

    pub fn get_q(&self, action: usize) -> f32 {
        self.q[action].dequantize()
    }

    pub fn get_v(&self) -> f32 {
        self.v.dequantize()
    }

    pub fn update_on_visit(&mut self, action: usize, value: f32) {
        let n_a = self.n[action] as f32;
        let q = self.q[action].dequantize();
        self.q[action] = NNQuantizedValue::quantize((n_a * q + value) / (n_a + 1.0));
        self.n[action] += 1;
    }

    pub fn rollout_value(&self) -> f32 {
        let v = self.v.dequantize();
        (0..self.n.len())
            .map(|a| {
                let n_a = self.n[a] as f32;
                let v_tot = n_a * self.q[a].dequantize() + v;
                v_tot / (n_a + 1.0)
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }
}

#[pymethods]
impl MCTSNode {
    #[getter(q)]
    fn get_q_py(&self) -> Vec<f32> {
        self.q.iter().map(|q| q.dequantize()).collect()
    }

    #[getter(pi)]
    fn get_pi_py(&self) -> Vec<f32> {
        self.pi.iter().map(|q| q.dequantize()).collect()
    }

    #[getter(v)]
    fn get_v_py(&self) -> f32 {
        self.v.dequantize()
    }
}
