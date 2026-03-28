use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use dashmap::DashMap;

use crate::cc::game::board::Board;
use crate::cc::rollouts::mcts_node::MCTSNode;


// ── CheapNode: lightweight non-transposing tree for prune tracking ──

struct CheapNode {
    board_hash: u64,
    visits: u32,
    children: HashMap<usize, CheapNode>,
}

impl CheapNode {
    fn new(board_hash: u64) -> Self {
        Self { board_hash, visits: 0, children: HashMap::new() }
    }

    fn insert_path(&mut self, actions: &[usize], reached_hashes: &[u64]) {
        self.visits += 1;
        if let Some((&first_action, rest_actions)) = actions.split_first() {
            if let Some((&child_hash, rest_hashes)) = reached_hashes.split_first() {
                let child = self.children
                    .entry(first_action)
                    .or_insert_with(|| CheapNode::new(child_hash));
                child.insert_path(rest_actions, rest_hashes);
            }
        }
    }

    fn decrement_all(&self, data: &DashMap<Board, MCTSNode>, board_lookup: &HashMap<u64, Board>) {
        if let Some(board) = board_lookup.get(&self.board_hash) {
            if let Some(node) = data.get(board) {
                let prev = node.refcount.fetch_sub(self.visits, Relaxed);
                drop(node);
                if prev <= self.visits {
                    data.remove(board);
                }
            }
        }
        for child in self.children.values() {
            child.decrement_all(data, board_lookup);
        }
    }

    fn count_nodes(&self) -> usize {
        1 + self.children.values().map(|c| c.count_nodes()).sum::<usize>()
    }

    fn estimated_bytes(&self) -> usize {
        let self_size = std::mem::size_of::<Self>()
            + self.children.capacity() * (std::mem::size_of::<usize>() + std::mem::size_of::<Self>());
        self_size + self.children.values().map(|c| c.estimated_bytes()).sum::<usize>()
    }
}


// ── PruningTree state ──

struct PruningTree {
    board_lookup: HashMap<u64, Board>,
    /// Per-thread list of rollout paths. Each inner vec is one rollout's path.
    thread_paths: Vec<Mutex<Vec<Vec<(u64, usize)>>>>,
    root: Option<CheapNode>,
}

impl PruningTree {
    fn new(n_threads: usize) -> Self {
        Self {
            board_lookup: HashMap::new(),
            thread_paths: (0..n_threads).map(|_| Mutex::new(Vec::new())).collect(),
            root: None,
        }
    }
}


// ── Tree ──

pub struct Tree {
    data: DashMap<Board, MCTSNode>,
    tracking: Option<Mutex<PruningTree>>,
    debug_prints: bool,
    /// Tracks all board hashes ever inserted, for re-expansion counting (debug only).
    ever_seen: Option<Mutex<HashSet<u64>>>,
    reexpansion_count: AtomicU64,
}

impl Tree {
    pub fn new(n_threads: usize, pruning_tree: bool, debug_prints: bool) -> Self {
        Self {
            data: DashMap::new(),
            tracking: if pruning_tree { Some(Mutex::new(PruningTree::new(n_threads))) } else { None },
            debug_prints,
            ever_seen: if debug_prints { Some(Mutex::new(HashSet::new())) } else { None },
            reexpansion_count: AtomicU64::new(0),
        }
    }

    /// Look up a board position, incrementing its refcount.
    pub fn visit(&self, board: &Board, _thread_id: usize) -> Option<dashmap::mapref::one::Ref<'_, Board, MCTSNode>> {
        let node = self.data.get(board)?;
        node.refcount.fetch_add(1, Relaxed);
        Some(node)
    }

    /// Start a new rollout for a thread. Must be called before each rollout.
    pub fn begin_rollout(&self, thread_id: usize) {
        if let Some(ref tracking_mutex) = self.tracking {
            let tracking = tracking_mutex.lock().unwrap();
            tracking.thread_paths[thread_id].lock().unwrap().push(Vec::new());
        }
    }

    /// Record the action chosen by a rollout thread (appends to current rollout's path).
    pub fn record_action(&self, thread_id: usize, board: &Board, action: usize) {
        if let Some(ref tracking_mutex) = self.tracking {
            let tracking = tracking_mutex.lock().unwrap();
            let hash = board.compute_hash();
            let mut paths = tracking.thread_paths[thread_id].lock().unwrap();
            if let Some(current_path) = paths.last_mut() {
                current_path.push((hash, action));
            }
        }
    }

    /// Insert a new leaf node.
    pub fn insert(&self, board: &Board, node: MCTSNode) -> bool {
        use dashmap::mapref::entry::Entry;
        let hash = board.compute_hash();
        let inserted = match self.data.entry(board.clone()) {
            Entry::Occupied(_) => false,
            Entry::Vacant(e) => {
                e.insert(node);
                true
            }
        };
        if inserted {
            if let Some(ref ever_seen) = self.ever_seen {
                let mut seen = ever_seen.lock().unwrap();
                if !seen.insert(hash) {
                    // Hash was already in the set — this board was pruned and re-expanded.
                    self.reexpansion_count.fetch_add(1, Relaxed);
                }
            }
        }
        inserted
    }

    /// Merge thread paths into the tracking tree. Called single-threaded after thread::scope.
    pub fn finalize_rollouts(&self, root_board: &Board) {
        let Some(ref tracking_mutex) = self.tracking else { return };
        let mut tracking = tracking_mutex.lock().unwrap();

        // Collect and clear thread paths first (flatten per-thread Vec<Vec<...>> into one list).
        let mut collected_paths: Vec<Vec<(u64, usize)>> = Vec::new();
        for path_mutex in &tracking.thread_paths {
            let mut thread_paths = path_mutex.lock().unwrap();
            for path in thread_paths.drain(..) {
                if !path.is_empty() {
                    collected_paths.push(path);
                }
            }
        }

        // Update board_lookup from current DashMap contents.
        let root_hash = root_board.compute_hash();
        tracking.board_lookup.entry(root_hash).or_insert_with(|| root_board.clone());
        for entry in self.data.iter() {
            let hash = entry.key().compute_hash();
            tracking.board_lookup.entry(hash).or_insert_with(|| entry.key().clone());
        }

        // Ensure root exists and merge paths.
        // Path format: [(child_hash, action), ...] — each entry is the board reached
        // and the action from its parent that led there.
        let root = tracking.root.get_or_insert_with(|| CheapNode::new(root_hash));
        for path in &collected_paths {
            let actions: Vec<usize> = path.iter().map(|&(_, a)| a).collect();
            let reached_hashes: Vec<u64> = path.iter().map(|&(h, _)| h).collect();
            root.insert_path(&actions, &reached_hashes);
        }
    }

    /// Prune discarded branches after a move. Re-root at the played child.
    fn prune(&self, played_action: usize) {
        let Some(ref tracking_mutex) = self.tracking else { return };
        let mut tracking = tracking_mutex.lock().unwrap();

        let Some(mut root) = tracking.root.take() else { return };

        // Walk and decrement all children except the played action.
        for (&action, child) in &root.children {
            if action != played_action {
                child.decrement_all(&self.data, &tracking.board_lookup);
            }
        }

        // Decrement root by discarded visits.
        let kept_visits = root.children.get(&played_action).map_or(0, |c| c.visits);
        let discarded_visits = root.visits.saturating_sub(kept_visits);
        if discarded_visits > 0 {
            if let Some(board) = tracking.board_lookup.get(&root.board_hash) {
                if let Some(node) = self.data.get(board) {
                    let prev = node.refcount.fetch_sub(discarded_visits, Relaxed);
                    drop(node);
                    if prev <= discarded_visits {
                        self.data.remove(board);
                    }
                }
            }
        }

        // Re-root.
        tracking.root = root.children.remove(&played_action);

        // Clean up stale board_lookup entries.
        tracking.board_lookup.retain(|_, board| self.data.contains_key(board));
    }

    /// Detect board change, prune, and return the played action.
    /// Panics if tracking is enabled and board is not a child of the root.
    pub fn maybe_prune(&self, board: &Board) -> Option<usize> {
        let Some(ref tracking_mutex) = self.tracking else { return None };
        let tracking = tracking_mutex.lock().unwrap();
        let Some(ref root) = tracking.root else { return None };

        let board_hash = board.compute_hash();
        if root.board_hash == board_hash {
            return None;
        }

        // Find the played action by generating all moves from the root board
        // and comparing child hashes. This works even if the action wasn't explored.
        let root_board = tracking.board_lookup.get(&root.board_hash)
            .expect("[tree] root board not in board_lookup — this is a bug")
            .clone();
        drop(tracking);

        let moves = crate::cc::game::moves::find_all_moves(&root_board);
        let played_action = moves.iter().enumerate()
            .find(|(_, m)| root_board.apply(m).compute_hash() == board_hash)
            .map(|(i, _)| i);

        match played_action {
            Some(action) => {
                self.prune(action);
                Some(action)
            }
            None => {
                panic!(
                    "[tree] board hash {:016x} not reachable from root {:016x}. \
                     This is a bug — the board should be one move away from the root.",
                    board_hash, root_board.compute_hash(),
                );
            }
        }
    }

    /// Clear all data and tracking state.
    pub fn clear(&self) {
        self.data.clear();
        if let Some(ref tracking_mutex) = self.tracking {
            let mut tracking = tracking_mutex.lock().unwrap();
            tracking.root = None;
            tracking.board_lookup.clear();
            for path_mutex in &tracking.thread_paths {
                path_mutex.lock().unwrap().clear();
            }
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Read-only lookup (no refcount increment, no tracking).
    pub fn get_data(&self, board: &Board) -> Option<dashmap::mapref::one::Ref<'_, Board, MCTSNode>> {
        self.data.get(board)
    }

    pub fn iter_data(&self) -> impl Iterator<Item = dashmap::mapref::multiple::RefMulti<'_, Board, MCTSNode>> {
        self.data.iter()
    }

    /// Memory diagnostics.
    pub fn memory_report(&self) -> MemoryReport {
        let mut dashmap_bytes: usize = 0;
        let mut dashmap_nodes: usize = 0;
        for entry in self.data.iter() {
            dashmap_bytes += std::mem::size_of::<Board>() + entry.value().estimated_bytes();
            dashmap_nodes += 1;
        }
        dashmap_bytes += dashmap_nodes * 50; // DashMap per-entry overhead estimate

        let (cheap_tree_nodes, cheap_tree_bytes, board_lookup_entries) =
            if let Some(ref tracking_mutex) = self.tracking {
                let tracking = tracking_mutex.lock().unwrap();
                let (ct_nodes, ct_bytes) = if let Some(ref root) = tracking.root {
                    (root.count_nodes(), root.estimated_bytes())
                } else {
                    (0, 0)
                };
                (ct_nodes, ct_bytes, tracking.board_lookup.len())
            } else {
                (0, 0, 0)
            };

        let board_lookup_bytes = board_lookup_entries * (8 + std::mem::size_of::<Board>() + 32);

        let reexpansion_count = self.reexpansion_count.load(Relaxed);
        let ever_seen_count = self.ever_seen.as_ref()
            .map_or(0, |s| s.lock().unwrap().len());

        MemoryReport {
            dashmap_nodes,
            dashmap_bytes,
            cheap_tree_nodes,
            cheap_tree_bytes,
            board_lookup_entries,
            board_lookup_bytes,
            reexpansion_count,
            ever_seen_count,
        }
    }

    pub fn debug_prints(&self) -> bool {
        self.debug_prints
    }
}

pub struct MemoryReport {
    pub dashmap_nodes: usize,
    pub dashmap_bytes: usize,
    pub cheap_tree_nodes: usize,
    pub cheap_tree_bytes: usize,
    pub board_lookup_entries: usize,
    pub board_lookup_bytes: usize,
    pub reexpansion_count: u64,
    pub ever_seen_count: usize,
}

impl MemoryReport {
    pub fn total_bytes(&self) -> usize {
        self.dashmap_bytes + self.cheap_tree_bytes + self.board_lookup_bytes
    }
}

impl std::fmt::Display for MemoryReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dashmap={} nodes ({:.1} KB), cheap_tree={} nodes ({:.1} KB), lookup={} ({:.1} KB), total={:.1} KB, reexpansions={}, ever_seen={}",
            self.dashmap_nodes, self.dashmap_bytes as f64 / 1024.0,
            self.cheap_tree_nodes, self.cheap_tree_bytes as f64 / 1024.0,
            self.board_lookup_entries, self.board_lookup_bytes as f64 / 1024.0,
            self.total_bytes() as f64 / 1024.0,
            self.reexpansion_count,
            self.ever_seen_count,
        )
    }
}
