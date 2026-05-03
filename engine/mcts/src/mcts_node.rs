use std::sync::atomic::{AtomicI32, AtomicU32, Ordering::Relaxed};

use alpha_cc_core::WDL;
use alpha_cc_nn::{NNQuantizedValue};

use crate::outcome::Outcome;

/// Fixed-point scale for cumulative value sums (Q16.16).
const W_SCALE: f32 = 65536.0;

/// Statistics for a board position, shared across transpositions.
/// Used directly in the DashMap tree and exposed to Python via PyO3 getters.
/// Moves are not stored — regenerate via `find_all_moves(board)` when needed.
pub struct MCTSNode {

    pub pi_logits: Vec<f32>,
    pub v: NNQuantizedValue,
    pub nn_wdl: [f32; 3],
    pub n: Vec<AtomicU32>,
    pub w: Vec<AtomicI32>,
    /// Empirical WDL outcome counters (ticked by rollouts that reach terminals).
    pub n_win: AtomicU32,
    pub n_draw: AtomicU32,
    pub n_loss: AtomicU32,
    /// Reference count: total rollout visits. Used by tree pruning.
    pub refcount: AtomicU32,
}

impl Clone for MCTSNode {
    fn clone(&self) -> Self {
        self.snapshot()
    }
}

impl MCTSNode {
    pub fn new(pi_logits: Vec<f32>, v: f32, nn_wdl: [f32; 3], num_actions: usize) -> Self {
        Self {
            pi_logits,
            v: NNQuantizedValue::quantize(v),
            nn_wdl,
            n: (0..num_actions).map(|_| AtomicU32::new(0)).collect(),
            w: (0..num_actions).map(|_| AtomicI32::new(0)).collect(),
            n_win: AtomicU32::new(0),
            n_draw: AtomicU32::new(0),
            n_loss: AtomicU32::new(0),
            refcount: AtomicU32::new(1),
        }
    }

    /// Create an owned snapshot (cloning atomic values).
    pub fn snapshot(&self) -> Self {
        MCTSNode {
            pi_logits: self.pi_logits.clone(),
            v: self.v,
            nn_wdl: self.nn_wdl,
            n: self.n.iter().map(|a| AtomicU32::new(a.load(Relaxed))).collect(),
            w: self.w.iter().map(|a| AtomicI32::new(a.load(Relaxed))).collect(),
            n_win: AtomicU32::new(self.n_win.load(Relaxed)),
            n_draw: AtomicU32::new(self.n_draw.load(Relaxed)),
            n_loss: AtomicU32::new(self.n_loss.load(Relaxed)),
            refcount: AtomicU32::new(self.refcount.load(Relaxed)),
        }
    }

    /// Estimated heap bytes for this node (for memory diagnostics).
    pub fn estimated_bytes(&self) -> usize {
        let n_actions = self.pi_logits.len();
        std::mem::size_of::<Self>()
            + n_actions * std::mem::size_of::<f32>()
            + n_actions * std::mem::size_of::<AtomicU32>()
            + n_actions * std::mem::size_of::<AtomicI32>()
    }

    /// Tick the appropriate WDL counter for an observed rollout outcome.
    pub fn tick_outcome(&self, outcome: Outcome) {
        match outcome {
            Outcome::Win  => { self.n_win.fetch_add(1, Relaxed); }
            Outcome::Draw => { self.n_draw.fetch_add(1, Relaxed); }
            Outcome::Loss => { self.n_loss.fetch_add(1, Relaxed); }
        }
    }

    /// Bayesian-blended WDL: NN prior (pseudo-count 1) mixed with empirical counts.
    pub fn blended_wdl(&self) -> WDL {
        let nn = self.nn_wdl;
        let nw = self.n_win.load(Relaxed) as f32;
        let nd = self.n_draw.load(Relaxed) as f32;
        let nl = self.n_loss.load(Relaxed) as f32;
        let total = 1.0 + nw + nd + nl;
        WDL {
            win:  (nn[0] + nw) / total,
            draw: (nn[1] + nd) / total,
            loss: (nn[2] + nl) / total,
        }
    }

    #[inline]
    pub fn num_actions(&self) -> usize {
        self.pi_logits.len()
    }

    #[inline]
    pub fn get_q(&self, action: usize) -> f32 {
        let n = self.n[action].load(Relaxed);
        if n == 0 { return 0.0; }
        self.w[action].load(Relaxed) as f32 / (n as f32 * W_SCALE)
    }

    /// Paper's v_mix: interpolates value-net scalar with prior-weighted
    /// average of visited Q values. Used as the fallback for unvisited
    /// actions in completed_q.
    #[inline]
    pub fn v_mix(&self) -> f32 {
        let total_n = self.total_visits() as f32;
        if total_n == 0.0 {
            return self.get_v();
        }

        let pi = alpha_cc_nn::softmax(&self.pi_logits);

        let mut visited_pi_sum = 0.0;
        let mut visited_pi_q_sum = 0.0;
        for a in 0..self.num_actions() {
            if self.get_n(a) > 0 {
                visited_pi_sum += pi[a];
                visited_pi_q_sum += pi[a] * self.get_q(a);
            }
        }

        if visited_pi_sum < 1e-8 {
            return self.get_v();
        }

        let weighted_q = visited_pi_q_sum / visited_pi_sum;
        (self.get_v() + total_n * weighted_q) / (1.0 + total_n)
    }

    /// Softmaxed prior probabilities (derived from stored logits).
    /// Allocates a fresh vector each call.
    #[inline]
    pub fn pi_softmax(&self) -> Vec<f32> {
        alpha_cc_nn::softmax(&self.pi_logits)
    }

    #[inline]
    pub fn completed_qs(&self) -> Vec<f32> {
        (0..self.num_actions()).map(|a| self.completed_q(a)).collect()
    }

    #[inline]
    pub fn completed_q(&self, action: usize) -> f32 {
        if self.get_n(action) == 0 {
            self.v_mix()
        } else {
            self.get_q(action)
        }
    }

    #[inline]
    pub fn get_v(&self) -> f32 {
        self.v.dequantize()
    }

    #[inline]
    pub fn get_n(&self, action: usize) -> u32 {
        self.n[action].load(Relaxed)
    }

    #[inline]
    pub fn total_visits(&self) -> u32 {
        self.n.iter().map(|n| n.load(Relaxed)).sum()
    }

    #[inline]
    pub fn apply_virtual_loss(&self, action: usize) {
        self.n[action].fetch_add(1, Relaxed);
        self.w[action].fetch_add((-W_SCALE) as i32, Relaxed);
    }

    #[inline]
    pub fn resolve_virtual_loss(&self, action: usize, value: f32) {
        self.w[action].fetch_add(((value + 1.0) * W_SCALE) as i32, Relaxed);
    }
}
