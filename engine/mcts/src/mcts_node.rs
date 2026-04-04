use std::sync::atomic::{AtomicI32, AtomicU32, Ordering::Relaxed};

use alpha_cc_nn::{NNQuantizedPi, NNQuantizedValue};

/// Fixed-point scale for cumulative value sums (Q16.16).
const W_SCALE: f32 = 65536.0;

/// Statistics for a board position, shared across transpositions.
/// Used directly in the DashMap tree and exposed to Python via PyO3 getters.
/// Moves are not stored — regenerate via `find_all_moves(board)` when needed.
pub struct MCTSNode {

    pub pi: Vec<NNQuantizedPi>,
    pub v: NNQuantizedValue,
    pub n: Vec<AtomicU32>,
    pub w: Vec<AtomicI32>,
    /// Reference count: total rollout visits. Used by tree pruning.
    pub refcount: AtomicU32,
}

impl Clone for MCTSNode {
    fn clone(&self) -> Self {
        self.snapshot()
    }
}

impl MCTSNode {
    pub fn new(pi: Vec<f32>, v: f32, num_actions: usize) -> Self {
        Self {
            pi: NNQuantizedPi::quantize_vec(&pi),
            v: NNQuantizedValue::quantize(v),
            n: (0..num_actions).map(|_| AtomicU32::new(0)).collect(),
            w: (0..num_actions).map(|_| AtomicI32::new(0)).collect(),
            refcount: AtomicU32::new(1),
        }
    }

    /// Create an owned snapshot (cloning atomic values).
    pub fn snapshot(&self) -> Self {
        MCTSNode {
            pi: self.pi.clone(),
            v: self.v,
            n: self.n.iter().map(|a| AtomicU32::new(a.load(Relaxed))).collect(),
            w: self.w.iter().map(|a| AtomicI32::new(a.load(Relaxed))).collect(),
            refcount: AtomicU32::new(self.refcount.load(Relaxed)),
        }
    }

    /// Estimated heap bytes for this node (for memory diagnostics).
    pub fn estimated_bytes(&self) -> usize {
        let n_actions = self.pi.len();
        std::mem::size_of::<Self>()
            + n_actions * std::mem::size_of::<NNQuantizedPi>()
            + n_actions * std::mem::size_of::<AtomicU32>()
            + n_actions * std::mem::size_of::<AtomicI32>()
    }

    #[inline]
    pub fn num_actions(&self) -> usize {
        self.pi.len()
    }

    #[inline]
    pub fn get_q(&self, action: usize) -> f32 {
        let n = self.n[action].load(Relaxed);
        if n == 0 { return 0.0; }
        self.w[action].load(Relaxed) as f32 / (n as f32 * W_SCALE)
    }

    #[inline]
    pub fn get_pi(&self, action: usize) -> f32 {
        self.pi[action].dequantize()
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
        self.w[action].fetch_add((-1.0 * W_SCALE) as i32, Relaxed);
    }

    #[inline]
    pub fn resolve_virtual_loss(&self, action: usize, value: f32) {
        self.w[action].fetch_add(((value + 1.0) * W_SCALE) as i32, Relaxed);
    }
}

