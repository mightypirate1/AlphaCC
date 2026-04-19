use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use alpha_cc_nn::BoardEncoding;
use alpha_cc_mcts::{GumbelParams, HalvingScheduler, Scheduler, SigmaParams};

/// Raw NN output for a position.
#[derive(Clone, Default)]
pub struct NNData {
    pub pi: Vec<f32>,
    pub value: f32,
}

/// Progress update sent from AI thread to the app.
#[derive(Clone)]
pub struct AiProgress {
    pub board_hash: u64,
    pub mcts_pi: Vec<f32>,
    pub mcts_value: f32,
    pub nn: NNData,
    pub total_rollouts: usize,
}

/// Move decision sent from AI thread to the app.
pub struct AiMove {
    pub board_hash: u64,
    pub action_index: usize,
}

pub enum AiUpdate {
    Progress(AiProgress),
    Move(AiMove),
}

/// Shared state between the app and one AI thread.
struct SharedState<B: BoardEncoding> {
    board: Mutex<B>,
    should_think: AtomicBool,
    should_move: AtomicBool,
    sample: AtomicBool,
    shutdown: AtomicBool,
}

/// Handle held by the app to control one AI player.
pub struct AiHandle<B: BoardEncoding> {
    shared: Arc<SharedState<B>>,
    update_rx: std::sync::mpsc::Receiver<AiUpdate>,
    _thread: thread::JoinHandle<()>,
}

impl<B: BoardEncoding + 'static> AiHandle<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nn_addr: String,
        model_id: u32,
        gamma: f32,
        sigma: SigmaParams,
        gumbel: GumbelParams,
        n_threads: usize,
        rollout_depth: usize,
        pruning_tree: bool,
        initial_board: B,
    ) -> Self {
        let shared = Arc::new(SharedState {
            board: Mutex::new(initial_board),
            should_think: AtomicBool::new(false),
            should_move: AtomicBool::new(false),
            sample: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
        });
        let (update_tx, update_rx) = std::sync::mpsc::channel();
        let shared_clone = shared.clone();

        let handle = thread::spawn(move || {
            ai_thread(
                &nn_addr, model_id, gamma, sigma, gumbel, n_threads, rollout_depth, pruning_tree,
                shared_clone, update_tx,
            );
        });

        Self { shared, update_rx, _thread: handle }
    }

    pub fn set_board(&self, board: &B) {
        *self.shared.board.lock().unwrap() = board.clone();
    }

    pub fn set_thinking(&self, on: bool) {
        self.shared.should_think.store(on, Ordering::Relaxed);
    }

    pub fn request_move(&self, sample: bool) {
        self.shared.sample.store(sample, Ordering::Relaxed);
        self.shared.should_move.store(true, Ordering::Release);
    }

    pub fn try_recv(&self) -> Option<AiUpdate> {
        self.update_rx.try_recv().ok()
    }

    pub fn shutdown(self) {
        self.shared.shutdown.store(true, Ordering::Relaxed);
        self.shared.should_think.store(false, Ordering::Relaxed);
    }
}

// ── AI thread ──

#[allow(clippy::too_many_arguments)]
fn ai_thread<B: BoardEncoding>(
    nn_addr: &str,
    model_id: u32,
    gamma: f32,
    sigma: SigmaParams,
    gumbel: GumbelParams,
    n_threads: usize,
    rollout_depth: usize,
    pruning_tree: bool,
    shared: Arc<SharedState<B>>,
    update_tx: std::sync::mpsc::Sender<AiUpdate>,
) {
    let n = n_threads.max(1);
    let services: Vec<_> = (0..n)
        .map(|_| alpha_cc_nn_service::NNRemote::<B>::connect(nn_addr))
        .collect();
    let mcts = HalvingScheduler::build_improved_halving(services, model_id, gamma, sigma, gumbel, pruning_tree, false);
    let rollouts_per_batch = n_threads.max(1);
    let mut last_board_hash: u64 = 0;
    let mut total_rollouts: usize = 0;

    loop {
        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }

        if !shared.should_think.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        let board = shared.board.lock().unwrap().clone();
        let board_hash = board.compute_hash();

        if board_hash != last_board_hash {
            mcts.mcts().notify_move_applied(&board);
            last_board_hash = board_hash;
            total_rollouts = 0;
        }

        let result = mcts.run(&board, rollouts_per_batch, rollout_depth);
        total_rollouts += rollouts_per_batch;

        let nn = match mcts.mcts().get_node_snapshot(&board) {
            Some(node) => NNData {
                pi: alpha_cc_nn::softmax(&node.pi_logits),
                value: node.v.dequantize(),
            },
            None => NNData::default(),
        };

        let _ = update_tx.send(AiUpdate::Progress(AiProgress {
            board_hash,
            mcts_pi: result.pi.clone(),
            mcts_value: result.value,
            nn,
            total_rollouts,
        }));

        if shared.should_move.load(Ordering::Acquire) {
            let sample = shared.sample.load(Ordering::Relaxed);
            let action_index = if sample { sample_from_pi(&result.pi) } else { argmax(&result.pi) };
            let _ = update_tx.send(AiUpdate::Move(AiMove { board_hash, action_index }));
            shared.should_move.store(false, Ordering::Relaxed);
        }
    }
}

pub fn argmax(pi: &[f32]) -> usize {
    pi.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub fn sample_from_pi(pi: &[f32]) -> usize {
    use rand::RngExt;
    let r: f32 = rand::rng().random_range(0.0..1.0);
    let mut cumulative = 0.0;
    for (i, &p) in pi.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    pi.len().saturating_sub(1)
}
