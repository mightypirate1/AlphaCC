use alpha_cc_nn::softmax;

use crate::mcts_node::MCTSNode;
use crate::search::descent::Descent;

#[derive(Clone)]
pub struct SigmaParams {
    pub c_visit: f32,
    pub c_scale: f32,
}

/// σ(q) = (c_visit + N_max) * c_scale * q
#[inline]
pub fn sigma(q_a: f32, n_a: u32, n_max: u32, _c_visit: f32, c_scale: f32) -> f32 {
    // let ratio = if n_max == 0 { 0.0 } else { _n_a as f32 / n_max as f32 };
    // c_scale * (c_visit + ratio) * q
    // (c_visit + n_max as f32) * c_scale * q
    c_scale * (n_a as f32 / n_max as f32).sqrt() * q_a
}

pub struct ImprovedPolicyDescent {
    params: SigmaParams,
}

impl ImprovedPolicyDescent {
    pub fn new(config: SigmaParams) -> Self { Self { params: config } }
    pub fn sigma_params(&self) -> &SigmaParams { &self.params }
}

impl Descent for ImprovedPolicyDescent {
    type Config = SigmaParams;
    type RootState = ();

    fn fresh_root_state(&self, _root: &MCTSNode) -> Option<Self::RootState> { None }

    /// argmax_a [ π'(a) - N(a) / (1 + Σ N(b)) ] where π' = softmax(logits + σ(completedQ))
    fn select(&self, node: &MCTSNode, _root: Option<&Self::RootState>) -> usize {
        let n_actions = node.num_actions();
        let n_max = (0..n_actions).map(|a| node.get_n(a)).max().unwrap_or(0);
        let denom = 1.0 + node.total_visits() as f32;

        let improved_logits: Vec<f32> = (0..n_actions)
            .map(|a| node.pi_logits[a]
                + sigma(node.completed_q(a), node.get_n(a), n_max, self.params.c_visit, self.params.c_scale))
            .collect();
        let pi_improved = softmax(&improved_logits);

        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for a in 0..n_actions {
            let score = pi_improved[a] - (node.get_n(a) as f32 / denom);
            if score > best_score {
                best_score = score;
                best = a;
            }
        }
        best
    }

    /// Greedy tie-breaker: argmax_a [ logits(a) + σ(completedQ(a)) ]
    fn best_child(&self, node: &MCTSNode) -> usize {
        let n_actions = node.num_actions();
        let n_max = (0..n_actions).map(|a| node.get_n(a)).max().unwrap_or(0);
        (0..n_actions)
            .max_by(|&a, &b| {
                let sa = node.pi_logits[a]
                    + sigma(node.completed_q(a), node.get_n(a), n_max, self.params.c_visit, self.params.c_scale);
                let sb = node.pi_logits[b]
                    + sigma(node.completed_q(b), node.get_n(b), n_max, self.params.c_visit, self.params.c_scale);
                sa.partial_cmp(&sb).unwrap()
            })
            .unwrap_or(0)
    }
}
