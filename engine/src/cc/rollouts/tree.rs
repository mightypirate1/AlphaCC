use std::sync::atomic::{AtomicI32, AtomicU32, Ordering::Relaxed};

use dashmap::DashMap;

use crate::cc::dtypes::{NNQuantizedPi, NNQuantizedValue};
use crate::cc::game::board::Board;
use crate::cc::game::moves::find_all_moves;
use crate::cc::Move;

/// Fixed-point scale for cumulative value sums (Q16.16).
const W_SCALE: f32 = 65536.0;

/// Statistics for a board position, shared across transpositions.
pub struct NodeData {
    pub pi: Vec<NNQuantizedPi>,
    pub v: NNQuantizedValue,
    pub moves: Vec<Move>,
    pub n: Vec<AtomicU32>,
    pub w: Vec<AtomicI32>,
}

impl NodeData {
    pub fn new(pi: Vec<f32>, v: f32, board: &Board) -> Self {
        let moves = find_all_moves(board);
        let num_actions = moves.len();
        Self {
            pi: NNQuantizedPi::quantize_vec(&pi),
            v: NNQuantizedValue::quantize(v),
            moves,
            n: (0..num_actions).map(|_| AtomicU32::new(0)).collect(),
            w: (0..num_actions).map(|_| AtomicI32::new(0)).collect(),
        }
    }

    #[inline]
    pub fn num_actions(&self) -> usize {
        self.moves.len()
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


/// Concurrent node store keyed by board position.
/// Rollouts traverse via board lookups with no global lock — DashMap provides
/// shard-level locking for inserts and lock-free reads. NodeData fields use
/// atomics for concurrent updates.
pub struct Tree {
    data: DashMap<Board, NodeData>,
}

impl Tree {
    pub fn new() -> Self {
        Self { data: DashMap::new() }
    }

    pub fn clear(&self) {
        self.data.clear();
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Insert node data for a board position. Returns true if newly inserted.
    /// Uses DashMap's entry API to avoid races between contains_key and insert.
    pub fn insert_data(&self, board: &Board, data: NodeData) -> bool {
        use dashmap::mapref::entry::Entry;
        match self.data.entry(board.clone()) {
            Entry::Occupied(_) => false,
            Entry::Vacant(e) => {
                e.insert(data);
                true
            }
        }
    }

    /// Get a reference to node data for a board position.
    /// Returns a DashMap Ref guard that can be dereferenced to &NodeData.
    pub fn get_data(&self, board: &Board) -> Option<dashmap::mapref::one::Ref<'_, Board, NodeData>> {
        self.data.get(board)
    }

    /// Iterate over all node data (for internal node sampling).
    /// Note: DashMap iteration yields RefMulti guards.
    pub fn iter_data(&self) -> impl Iterator<Item = dashmap::mapref::multiple::RefMulti<'_, Board, NodeData>> {
        self.data.iter()
    }
}
