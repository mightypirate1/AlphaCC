use crate::mcts_node::MCTSNode;
use crate::noise;
use crate::search::descent::Descent;

#[derive(Clone)]
pub struct DirichletParams {
    pub weight: f32,
    pub alpha: f32,
}

#[derive(Clone)]
pub struct PuctParams {
    pub c_puct_init: f32,
    pub c_puct_base: f32,
    /// None disables root-noise injection entirely.
    pub dirichlet: Option<DirichletParams>,
}

pub struct PuctDescent {
    params: PuctParams,
}

impl PuctDescent {
    pub fn new(config: PuctParams) -> Self { Self { params: config } }
    pub fn params(&self) -> &PuctParams { &self.params }
}

impl Descent for PuctDescent {
    type Config = PuctParams;
    /// Pre-sampled dirichlet noise vector (same dimension as the root's pi).
    type RootState = Vec<f32>;

    fn fresh_root_state(&self, root: &MCTSNode) -> Option<Self::RootState> {
        let d = self.params.dirichlet.as_ref()?;
        if d.weight <= 0.0 { return None; }
        let pi = root.pi_softmax();
        Some(noise::sample_dirichlet(&pi, d.alpha))
    }

    /// PUCT: argmax_a [ Q(a) + c_puct * π(a) * √ΣN / (1 + N(a)) ].
    /// At the root, π is blended with dirichlet noise when `root` is Some.
    fn select(&self, node: &MCTSNode, root: Option<&Self::RootState>) -> usize {
        let sum_n_f = node.total_visits() as f32;
        let c_puct = self.params.c_puct_init
            + ((sum_n_f + self.params.c_puct_base + 1.0) / self.params.c_puct_base).ln();

        let mut pi = node.pi_softmax();
        if let (Some(noise_vec), Some(d)) = (root, self.params.dirichlet.as_ref()) {
            noise::blend_with_noise(&mut pi, noise_vec, d.weight);
        }

        let mut best_action = 0usize;
        let mut best_u = f32::NEG_INFINITY;
        for (i, &pi_a) in pi.iter().enumerate() {
            let n_a = node.get_n(i) as f32;
            let u = node.get_q(i) + c_puct * pi_a * sum_n_f.sqrt() / (1.0 + n_a);
            if u > best_u {
                best_u = u;
                best_action = i;
            }
        }
        best_action
    }

    /// Greedy tie-breaker: most-visited child.
    fn best_child(&self, node: &MCTSNode) -> usize {
        (0..node.num_actions())
            .max_by_key(|&a| node.get_n(a))
            .unwrap_or(0)
    }
}
