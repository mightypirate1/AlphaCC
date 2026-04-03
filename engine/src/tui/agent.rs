use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crate::cc::game::board::Board;
use crate::cc::rollouts::mcts::{MCTS, MCTSParams};

// ── Channel types ──

pub enum AiCommand {
    /// temperature=None means argmax, Some(t) means sample with that temperature
    Think { board: Board, time_budget: Duration, temperature: Option<f32> },
    MoveApplied { board: Board },
    Cancel,
    Shutdown,
}

pub struct AiProgress {
    pub pi: Vec<f32>,
    pub value: f32,
    pub total_rollouts: usize,
}

pub struct AiResult {
    pub action_index: usize,
    pub pi: Vec<f32>,
    pub value: f32,
    pub total_rollouts: usize,
}

pub enum AiUpdate {
    Progress(AiProgress),
    Done(AiResult),
}

// ── Handle held by the main (TUI) thread ──

pub struct AiHandle {
    cmd_tx: mpsc::Sender<AiCommand>,
    update_rx: mpsc::Receiver<AiUpdate>,
    _thread: thread::JoinHandle<()>,
}

impl AiHandle {
    pub fn new(
        nn_addr: String,
        model_id: u32,
        params: MCTSParams,
        n_threads: usize,
        rollout_depth: usize,
        pruning_tree: bool,
    ) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (update_tx, update_rx) = mpsc::channel();

        let handle = thread::spawn(move || {
            ai_thread_main(&nn_addr, model_id, params, n_threads, rollout_depth, pruning_tree, cmd_rx, update_tx);
        });

        Self {
            cmd_tx,
            update_rx,
            _thread: handle,
        }
    }

    pub fn start_thinking(&self, board: Board, time_budget: Duration, temperature: Option<f32>) {
        let _ = self.cmd_tx.send(AiCommand::Think { board, time_budget, temperature });
    }

    pub fn notify_move_applied(&self, board: &Board) {
        let _ = self.cmd_tx.send(AiCommand::MoveApplied { board: board.clone() });
    }

    pub fn cancel(&self) {
        let _ = self.cmd_tx.send(AiCommand::Cancel);
    }

    pub fn try_recv(&self) -> Option<AiUpdate> {
        self.update_rx.try_recv().ok()
    }

    pub fn shutdown(self) {
        let _ = self.cmd_tx.send(AiCommand::Shutdown);
    }
}

// ── Background thread ──

fn ai_thread_main(
    nn_addr: &str,
    model_id: u32,
    params: MCTSParams,
    n_threads: usize,
    rollout_depth: usize,
    pruning_tree: bool,
    cmd_rx: mpsc::Receiver<AiCommand>,
    update_tx: mpsc::Sender<AiUpdate>,
) {
    let mcts = MCTS::new(nn_addr, model_id, params, n_threads, pruning_tree, false);
    let rollouts_per_batch = n_threads.max(1);

    loop {
        let cmd = match cmd_rx.recv() {
            Ok(cmd) => cmd,
            Err(_) => return,
        };

        match cmd {
            AiCommand::Think { board, time_budget, temperature } => {
                think(&mcts, &board, time_budget, rollouts_per_batch, rollout_depth, temperature, &cmd_rx, &update_tx);
            }
            AiCommand::MoveApplied { board } => {
                mcts.notify_move_applied(&board);
            }
            AiCommand::Cancel => {}
            AiCommand::Shutdown => return,
        }
    }
}

fn think(
    mcts: &MCTS,
    board: &Board,
    time_budget: Duration,
    rollouts_per_batch: usize,
    rollout_depth: usize,
    temperature: Option<f32>,
    cmd_rx: &mpsc::Receiver<AiCommand>,
    update_tx: &mpsc::Sender<AiUpdate>,
) {
    let start = Instant::now();
    let mut total_rollouts = 0;
    // For the MCTS pi computation: use the temperature if sampling, otherwise 1.0
    // (the overflow-safe normalization in run_rollouts_inner handles any value now)
    let mcts_temperature = temperature.unwrap_or(1.0);

    loop {
        if let Ok(AiCommand::Cancel | AiCommand::Shutdown) = cmd_rx.try_recv() {
            break;
        }

        let (pi, value) = mcts.run_rollouts_inner(board, rollouts_per_batch, rollout_depth, mcts_temperature);
        total_rollouts += rollouts_per_batch;

        let _ = update_tx.send(AiUpdate::Progress(AiProgress {
            pi: pi.clone(),
            value,
            total_rollouts,
        }));

        if start.elapsed() >= time_budget {
            let action_index = match temperature {
                Some(_) => sample_from_pi(&pi),
                None => argmax(&pi),
            };

            let _ = update_tx.send(AiUpdate::Done(AiResult {
                action_index,
                pi,
                value,
                total_rollouts,
            }));
            return;
        }
    }
}

fn argmax(pi: &[f32]) -> usize {
    pi.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn sample_from_pi(pi: &[f32]) -> usize {
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
