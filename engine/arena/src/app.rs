use std::io;
use std::time::{Duration, Instant};

use crossterm::event::{EnableMouseCapture, DisableMouseCapture};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::widgets::{Block, Borders};
use ratatui::Terminal;

use alpha_cc_core::board::{CellContent, Coord};
use alpha_cc_nn::BoardEncoding;
use crate::agent::{AiHandle, AiUpdate};
use crate::game::GameState;
use crate::input::{self, AppEvent};
use crate::modal::{TemperatureModal, TemperatureModalWidget};
use crate::renderer::GameRenderer;
use crate::sidebar::{EvalBarWidget, MoveListWidget, PVWidget};
use crate::status_bar::{
    CurrentPlayerWidget, PolicyBarsWidget, ToggleState, TogglesPanelWidget,
};
use crate::styling;
use crate::theme;
use crate::visual::{BoardView, GameVisual};

// ── Player config ──

#[derive(Clone, PartialEq)]
pub enum PlayerConfig {
    Human,
    Ai { channel: u32 },
}

impl PlayerConfig {
    pub fn is_human(&self) -> bool {
        matches!(self, PlayerConfig::Human)
    }
}

// ── Drag resize targets ──

#[derive(Clone, Copy)]
enum DragTarget {
    SidebarLeft,
    PvTop,
    PolicyBarsTop,
}

// ── App state machine ──

#[derive(Clone)]
enum Phase<C: Coord> {
    HumanTurn,
    PieceSelected {
        piece: C,
        /// (action_index, display_destination)
        legal: Vec<(usize, C)>,
    },
    AiThinking { deadline: Instant },
    GameOver,
}

// ── Config passed from CLI ──

pub struct AppConfig {
    pub p1: PlayerConfig,
    pub p2: PlayerConfig,
    pub board_size: u8,
    pub nn_addr: String,
    pub think_time: Duration,
    pub n_threads: usize,
    pub rollout_depth: usize,
    pub gamma: f32,
    pub c_visit: f32,
    pub c_scale: f32,
    pub pruning_tree: bool,
}

// ── Main App ──

#[derive(Clone, Default)]
struct LatestAnalysis {
    board_hash: u64,
    current_player: i8,
    mcts_pi: Vec<f32>,
    mcts_value: f32,
    nn_pi: Vec<f32>,
    nn_value: f32,
    rollouts: usize,
}

pub struct App<B: BoardEncoding + GameVisual, R: GameRenderer<Coord = B::Coord>> {
    config: AppConfig,
    game: GameState<B>,
    phase: Phase<B::Coord>,
    toggles: ToggleState,
    renderer: R,
    view_index: usize,
    ai_p1: Option<AiHandle<B>>,
    ai_p2: Option<AiHandle<B>>,
    analysis_p1: LatestAnalysis,
    analysis_p2: LatestAnalysis,
    pv_moves: Vec<String>,
    board_area: Rect,
    last_screen_size: (u16, u16),
    temperature_modal: Option<TemperatureModal>,
    sidebar_width: u16,
    pv_height: u16,
    policy_bars_height: u16,
    sidebar_area: Rect,
    move_list_area: Rect,
    toggles_area: Rect,
    pv_area: Rect,
    policy_bars_area: Rect,
    hovered_move: Option<usize>,
    move_list_scroll: usize,
    move_list_latched: bool,
    dragging: Option<DragTarget>,
    should_quit: bool,
}

impl<B: BoardEncoding + GameVisual + 'static, R: GameRenderer<Coord = B::Coord>> App<B, R> {
    pub fn new(config: AppConfig, initial_board: B, renderer: R) -> Self {
        let game = GameState::new(initial_board);
        use crate::status_bar::EvalMode;
        let eval_mode = match (&config.p1, &config.p2) {
            (PlayerConfig::Ai { .. }, PlayerConfig::Ai { .. }) => EvalMode::Both,
            (PlayerConfig::Ai { .. }, _) => EvalMode::P1,
            (_, PlayerConfig::Ai { .. }) => EvalMode::P2,
            _ => EvalMode::Off,
        };
        let toggles = ToggleState {
            eval_mode,
            ..Default::default()
        };
        Self {
            config,
            game,
            phase: Phase::HumanTurn,
            toggles,
            renderer,
            view_index: 0,
            ai_p1: None,
            ai_p2: None,
            analysis_p1: LatestAnalysis::default(),
            analysis_p2: LatestAnalysis::default(),
            pv_moves: Vec::new(),
            board_area: Rect::default(),
            last_screen_size: (80, 24),
            temperature_modal: None,
            sidebar_width: 22,
            pv_height: 8,
            policy_bars_height: 3,
            sidebar_area: Rect::default(),
            move_list_area: Rect::default(),
            pv_area: Rect::default(),
            toggles_area: Rect::default(),
            policy_bars_area: Rect::default(),
            hovered_move: None,
            move_list_scroll: 0,
            move_list_latched: true,
            dragging: None,
            should_quit: false,
        }
    }

    fn player_config(&self, player: i8) -> &PlayerConfig {
        if player == 1 { &self.config.p1 } else { &self.config.p2 }
    }

    fn mcts_params(&self) -> alpha_cc_mcts::MCTSParams {
        alpha_cc_mcts::MCTSParams {
            gamma: self.config.gamma,
            c_visit: self.config.c_visit,
            c_scale: self.config.c_scale,
            gumbel: alpha_cc_mcts::GumbelParams {
                all_at_least_once: false,
                base_count: 16,
                floor_count: 5,
                keep_frac: 0.5,
            },
        }
    }

    fn ai_for(&self, player: i8) -> Option<&AiHandle<B>> {
        if player == 1 { self.ai_p1.as_ref() } else { self.ai_p2.as_ref() }
    }

    fn analysis(&self, player: i8) -> &LatestAnalysis {
        if player == 1 { &self.analysis_p1 } else { &self.analysis_p2 }
    }

    fn analysis_mut(&mut self, player: i8) -> &mut LatestAnalysis {
        if player == 1 { &mut self.analysis_p1 } else { &mut self.analysis_p2 }
    }

    fn ensure_ai(&mut self, player: i8) {
        let slot = if player == 1 { &self.ai_p1 } else { &self.ai_p2 };
        if slot.is_some() { return; }
        let channel = match self.player_config(player) {
            PlayerConfig::Ai { channel } => *channel,
            PlayerConfig::Human => return,
        };
        let ai = AiHandle::new(
            self.config.nn_addr.clone(),
            channel,
            self.mcts_params(),
            self.config.n_threads,
            self.config.rollout_depth,
            self.config.pruning_tree,
            self.game.current_board().clone(),
        );
        if player == 1 { self.ai_p1 = Some(ai); } else { self.ai_p2 = Some(ai); }
    }

    fn sync_ai_state(&mut self) {
        let board = self.game.current_board();
        let cp = board.get_info().current_player;

        for player in [1i8, 2] {
            if let Some(ai) = self.ai_for(player) {
                ai.set_board(board);
                let should_think = match self.player_config(player) {
                    PlayerConfig::Ai { .. } => {
                        cp == player || self.toggles.pondering
                    }
                    PlayerConfig::Human => false,
                };
                ai.set_thinking(should_think);
            }
        }
    }

    fn apply_move(&mut self, action_index: usize) {
        self.game.apply_move(action_index);
        self.view_index = self.game.len() - 1;
        self.pv_moves.clear();
        self.hovered_move = None;

        if self.move_list_latched {
            let visible = self.move_list_visible_rows();
            self.move_list_scroll = self.game.ply().saturating_sub(visible);
        }

        if self.game.is_game_over() {
            if let Some(ai) = &self.ai_p1 { ai.set_thinking(false); }
            if let Some(ai) = &self.ai_p2 { ai.set_thinking(false); }
            self.phase = Phase::GameOver;
        } else {
            self.advance_turn();
        }
    }

    fn advance_turn(&mut self) {
        let player = self.game.current_board().get_info().current_player;
        self.sync_ai_state();
        match self.player_config(player) {
            PlayerConfig::Human => {
                self.phase = Phase::HumanTurn;
            }
            PlayerConfig::Ai { .. } => {
                self.phase = Phase::AiThinking { deadline: Instant::now() + self.config.think_time };
            }
        }
    }

    fn viewing_live(&self) -> bool {
        self.view_index == self.game.len() - 1
    }

    fn displayed_current_player(&self) -> i8 {
        self.game.board_at(self.view_index).get_info().current_player
    }

    // ── Event handling ──

    fn handle_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::Quit => self.should_quit = true,
            AppEvent::ToggleEval => {
                let p1_ai = !self.config.p1.is_human();
                let p2_ai = !self.config.p2.is_human();
                self.toggles.eval_mode = self.toggles.eval_mode.cycle(p1_ai, p2_ai);
            }
            AppEvent::ToggleDataSource => self.toggles.data_source = self.toggles.data_source.toggle(),
            AppEvent::TogglePolicy => {
                self.toggles.policy_mode = self.toggles.policy_mode.cycle(true, true);
            }
            AppEvent::TogglePolicyScale => self.toggles.policy_scale = self.toggles.policy_scale.cycle(),
            AppEvent::TogglePV => self.toggles.show_pv = !self.toggles.show_pv,
            AppEvent::ToggleMoves => self.toggles.show_moves = !self.toggles.show_moves,
            AppEvent::ToggleSampling => self.toggles.toggle_sampling(),
            AppEvent::TogglePonder => {
                self.toggles.pondering = !self.toggles.pondering;
                self.sync_ai_state();
            }
            AppEvent::ToggleRenderer => {}
            AppEvent::ToggleBrailleBars => self.toggles.braille_bars = !self.toggles.braille_bars,
            AppEvent::OpenTemperatureModal => {
                let modal = TemperatureModal {
                    value: self.toggles.sampling_temperature,
                    ..Default::default()
                };
                self.temperature_modal = Some(modal);
            }
            AppEvent::HistoryBack => {
                if self.view_index > 0 {
                    self.view_index -= 1;
                    self.sync_move_list_to_view();
                }
            }
            AppEvent::HistoryForward => {
                if self.view_index < self.game.len() - 1 {
                    self.view_index += 1;
                    self.sync_move_list_to_view();
                }
            }
            AppEvent::HistoryStart => {
                self.view_index = 0;
                self.sync_move_list_to_view();
            }
            AppEvent::HistoryEnd => {
                self.view_index = self.game.len() - 1;
                self.sync_move_list_to_view();
            }
            AppEvent::NewGame => {
                if let Some(ai) = &self.ai_p1 { ai.set_thinking(false); }
                if let Some(ai) = &self.ai_p2 { ai.set_thinking(false); }
                self.game.reset();
                self.view_index = 0;
                self.pv_moves.clear();
                self.analysis_p1 = LatestAnalysis::default();
                self.analysis_p2 = LatestAnalysis::default();
                self.advance_turn();
            }
            AppEvent::MouseDown(col, row) => self.handle_mouse_down(col, row),
            AppEvent::MouseDrag(col, row) => self.handle_mouse_drag(col, row),
            AppEvent::MouseUp => self.dragging = None,
            AppEvent::MouseMove(col, row) => self.handle_mouse_move(col, row),
            AppEvent::ScrollUp(col, row) => self.handle_scroll(-3, col, row),
            AppEvent::ScrollDown(col, row) => self.handle_scroll(3, col, row),
            AppEvent::PauseResume | AppEvent::Tick | AppEvent::Resize(_, _) => {}
            AppEvent::ModalLeft | AppEvent::ModalRight | AppEvent::ModalConfirm
            | AppEvent::ModalCancel | AppEvent::ModalClick(_, _) => {}
        }
    }

    fn handle_mouse_down(&mut self, col: u16, row: u16) {
        if let Some(coord) = self.renderer.screen_to_coord(col, row, self.board_area) {
            self.handle_click(coord);
            return;
        }

        let pb = self.policy_bars_area;
        if pb.width > 0 && row.abs_diff(pb.y) <= 1 && col >= pb.x && col < pb.x + pb.width {
            self.dragging = Some(DragTarget::PolicyBarsTop);
            return;
        }
        let sb = self.sidebar_area;
        if sb.width > 0 && col.abs_diff(sb.x) <= 1 && row >= sb.y && row < sb.y + sb.height {
            self.dragging = Some(DragTarget::SidebarLeft);
            return;
        }
        let pv = self.pv_area;
        if pv.width > 0 && row.abs_diff(pv.y) <= 1 && col >= pv.x && col < pv.x + pv.width {
            self.dragging = Some(DragTarget::PvTop);
        }
    }

    fn handle_scroll(&mut self, delta: i32, col: u16, row: u16) {
        let sb = self.sidebar_area;
        if sb.width > 0 && col >= sb.x && col < sb.x + sb.width && row >= sb.y && row < sb.y + sb.height {
            let n_moves = self.game.ply();
            let visible = self.move_list_visible_rows();
            let max_scroll = n_moves.saturating_sub(visible);

            let new_scroll = (self.move_list_scroll as i32 + delta).clamp(0, max_scroll as i32) as usize;
            self.move_list_scroll = new_scroll;
            self.move_list_latched = self.move_list_scroll >= max_scroll;
        }
    }

    fn move_list_visible_rows(&self) -> usize {
        self.move_list_area.height.saturating_sub(2).max(1) as usize
    }

    fn sync_move_list_to_view(&mut self) {
        if self.view_index == 0 {
            self.move_list_scroll = 0;
            self.move_list_latched = false;
            return;
        }
        let move_idx = self.view_index - 1;
        let visible = self.move_list_visible_rows();

        if move_idx < self.move_list_scroll {
            self.move_list_scroll = move_idx;
        } else if move_idx >= self.move_list_scroll + visible {
            self.move_list_scroll = move_idx.saturating_sub(visible) + 1;
        }

        let n_moves = self.game.ply();
        let max_scroll = n_moves.saturating_sub(visible);
        self.move_list_latched = self.view_index == self.game.len() - 1 && self.move_list_scroll >= max_scroll;
    }

    fn handle_mouse_move(&mut self, col: u16, row: u16) {
        let pb = self.policy_bars_area;
        if pb.width > 0 && col >= pb.x && col < pb.x + pb.width && row >= pb.y && row < pb.y + pb.height {
            let pi = self.active_pi();
            if !pi.is_empty() {
                let inner_x = pb.x + 1;
                let inner_w = pb.width.saturating_sub(2);
                let n = pi.len() as u16;
                let start_x = inner_x + inner_w.saturating_sub(n) / 2;
                if col >= start_x && col < start_x + n {
                    self.hovered_move = Some((col - start_x) as usize);
                    return;
                }
            }
        }
        self.hovered_move = None;
    }

    fn handle_mouse_drag(&mut self, col: u16, row: u16) {
        let Some(target) = self.dragging else { return };
        match target {
            DragTarget::SidebarLeft => {
                let toggles_left = self.toggles_area.x;
                let new_width = toggles_left.saturating_sub(col);
                let max_w = self.last_screen_size.0 / 2;
                self.sidebar_width = new_width.clamp(14, max_w);
            }
            DragTarget::PvTop => {
                let sb = self.sidebar_area;
                let sb_bottom = sb.y + sb.height;
                let new_pv = sb_bottom.saturating_sub(row);
                self.pv_height = new_pv.clamp(3, sb.height.saturating_sub(4));
            }
            DragTarget::PolicyBarsTop => {
                let screen_h = self.last_screen_size.1;
                let new_h = screen_h.saturating_sub(row);
                self.policy_bars_height = new_h.clamp(3, screen_h / 3);
            }
        }
    }

    fn handle_modal_event(&mut self, event: AppEvent) {
        let modal = match &mut self.temperature_modal {
            Some(m) => m,
            None => return,
        };
        match event {
            AppEvent::ModalLeft => modal.nudge_left(),
            AppEvent::ModalRight => modal.nudge_right(),
            AppEvent::ModalConfirm => {
                let temp = self.temperature_modal.as_ref().unwrap().as_temperature();
                self.toggles.set_temperature(temp);
                self.temperature_modal = None;
            }
            AppEvent::ModalCancel => {
                self.temperature_modal = None;
            }
            AppEvent::ModalClick(col, row) => {
                let modal_area = TemperatureModal::area(Rect {
                    x: 0, y: 0,
                    width: self.last_screen_size.0,
                    height: self.last_screen_size.1,
                });
                let block = ratatui::widgets::Block::default().borders(ratatui::widgets::Borders::ALL).border_set(theme::BORDER);
                let inner = block.inner(modal_area);
                let (track_start, track_width, track_row) = TemperatureModal::track_rect(inner);
                if row == track_row && col >= track_start && col < track_start + track_width {
                    modal.click_at(col, track_start, track_width);
                }
            }
            _ => {}
        }
    }

    fn handle_click(&mut self, display_coord: B::Coord) {
        if !self.viewing_live() { return; }

        match &self.phase {
            Phase::HumanTurn => {
                let board = self.game.current_board();
                let info = board.get_info();
                let content = board.get_cell_unflipped(&display_coord).player();
                if content == info.current_player {
                    self.select_piece(display_coord);
                }
            }
            Phase::PieceSelected { piece, legal } => {
                let piece = *piece;
                let legal = legal.clone();
                if display_coord == piece {
                    self.phase = Phase::HumanTurn;
                } else if let Some((action_index, _)) = legal.iter().find(|(_, dest)| *dest == display_coord) {
                    self.apply_move(*action_index);
                } else {
                    let board = self.game.current_board();
                    let info = board.get_info();
                    let content = board.get_cell_unflipped(&display_coord).player();
                    if content == info.current_player {
                        self.select_piece(display_coord);
                    } else {
                        self.phase = Phase::HumanTurn;
                    }
                }
            }
            _ => {}
        }
    }

    fn select_piece(&mut self, display_coord: B::Coord) {
        let board = self.game.current_board();
        let info = board.get_info();
        let moves = board.legal_moves();

        let legal: Vec<(usize, B::Coord)> = moves.iter().enumerate().filter_map(|(i, mv)| {
            let from_display = if info.current_player == 1 { mv.from_coord } else { mv.from_coord.flip() };
            if from_display == display_coord {
                let to_display = if info.current_player == 1 { mv.to_coord } else { mv.to_coord.flip() };
                Some((i, to_display))
            } else {
                None
            }
        }).collect();

        if !legal.is_empty() {
            self.phase = Phase::PieceSelected { piece: display_coord, legal };
        }
    }

    fn poll_ai_updates(&mut self) {
        let board_hash = self.game.current_board().compute_hash();
        let cp = self.game.current_board().get_info().current_player;

        for player in [1i8, 2] {
            loop {
                let update = match self.ai_for(player) {
                    Some(ai) => ai.try_recv(),
                    None => None,
                };
                match update {
                    Some(AiUpdate::Progress(p)) => {
                        if p.board_hash == board_hash {
                            let a = self.analysis_mut(player);
                            a.board_hash = p.board_hash;
                            a.current_player = cp;
                            a.mcts_value = p.mcts_value;
                            a.mcts_pi = p.mcts_pi;
                            a.nn_value = p.nn.value;
                            a.nn_pi = p.nn.pi;
                            a.rollouts = p.total_rollouts;
                        }
                    }
                    Some(AiUpdate::Move(m)) => {
                        if m.board_hash == board_hash && matches!(self.phase, Phase::AiThinking { .. }) {
                            self.apply_move(m.action_index);
                            return;
                        }
                    }
                    None => break,
                }
            }
        }

        if let Phase::AiThinking { deadline } = self.phase {
            if Instant::now() >= deadline {
                let player = self.game.current_board().get_info().current_player;
                if let Some(ai) = self.ai_for(player) {
                    ai.request_move(self.toggles.temperature().is_some());
                }
            }
        }
    }

    // ── Build board view for the styling engine ──

    fn build_board_view(&self) -> BoardView<B::Coord> {
        let board = self.game.board_at(self.view_index);
        let (s, _) = board.get_sizes();

        // Build cells using GameVisual trait
        let mut cells = Vec::with_capacity(s as usize * s as usize);
        for x in 0..s {
            for y in 0..s {
                let coord = B::Coord::new(x, y, s);
                let cell = board.get_cell_unflipped(&coord);
                let visual = B::cell_visual(cell, &coord, s);
                cells.push((coord, visual));
            }
        }

        let mut view = BoardView {
            board_size: s,
            current_player: self.displayed_current_player(),
            cells,
            selected: None,
            legal_destinations: Vec::new(),
            last_move: None,
            policy: Vec::new(),
            hovered_move: None,
        };

        if let Phase::PieceSelected { piece, legal } = &self.phase {
            if self.viewing_live() {
                view.selected = Some(*piece);
                view.legal_destinations = legal.iter().map(|(_, dest)| *dest).collect();
            }
        }

        if self.view_index > 0 {
            let rec = &self.game.move_records()[self.view_index - 1];
            let prev_board = self.game.board_at(self.view_index - 1);
            let info = prev_board.get_info();
            let player = info.current_player;
            let from = if player == 1 { rec.mv.from_coord } else { rec.mv.from_coord.flip() };
            let to = if player == 1 { rec.mv.to_coord } else { rec.mv.to_coord.flip() };
            view.last_move = Some((from, to, player));
        }

        let pi = self.active_pi();
        if self.viewing_live() && self.toggles.policy_mode.any() && !pi.is_empty() {
            let current_board = self.game.current_board();
            let info = current_board.get_info();
            let moves = current_board.legal_moves();
            let weights = self.scale_pi(pi);
            view.policy = moves.iter().enumerate()
                .filter(|(i, _)| *i < weights.len())
                .map(|(i, mv)| {
                    let dest = if info.current_player == 1 { mv.to_coord } else { mv.to_coord.flip() };
                    (dest, weights[i])
                })
                .collect();
        }

        if let Some(move_idx) = self.hovered_move {
            if self.viewing_live() {
                let current_board = self.game.current_board();
                let info = current_board.get_info();
                let moves = current_board.legal_moves();
                if move_idx < moves.len() {
                    let mv = &moves[move_idx];
                    let from = if info.current_player == 1 { mv.from_coord } else { mv.from_coord.flip() };
                    let to = if info.current_player == 1 { mv.to_coord } else { mv.to_coord.flip() };
                    view.hovered_move = Some((from, to));
                }
            }
        }

        view
    }

    fn scale_pi(&self, pi: &[f32]) -> Vec<f32> {
        use crate::status_bar::PolicyScale;
        if pi.is_empty() { return Vec::new(); }

        match self.toggles.policy_scale {
            PolicyScale::Raw => {
                let max = pi.iter().cloned().fold(0.0f32, f32::max);
                if max > 0.0 {
                    pi.iter().map(|&v| v / max).collect()
                } else {
                    pi.to_vec()
                }
            }
            PolicyScale::Rank => {
                let n = pi.len();
                if n == 0 { return Vec::new(); }
                let mut indexed: Vec<(usize, f32)> = pi.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut weights = vec![0.0f32; n];
                for (rank, &(idx, _)) in indexed.iter().enumerate() {
                    weights[idx] = 1.0 - (rank as f32 / n as f32);
                }
                weights
            }
            PolicyScale::Spread => {
                let min = pi.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = pi.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let range = max - min;
                if range > 1e-9 {
                    pi.iter().map(|&v| (v - min) / range).collect()
                } else {
                    vec![0.5; pi.len()]
                }
            }
        }
    }

    fn active_pi(&self) -> &[f32] {
        use crate::status_bar::DataSource;
        if !self.toggles.policy_mode.any() { return &[]; }

        let bh = self.game.current_board().compute_hash();
        let cp = self.game.current_board().get_info().current_player;
        let opponent = if cp == 1 { 2 } else { 1 };

        let candidates: Vec<i8> = [cp, opponent].into_iter().filter(|&p| {
            match p {
                1 => self.toggles.policy_mode.show_p1(),
                _ => self.toggles.policy_mode.show_p2(),
            }
        }).collect();

        for player in candidates {
            let a = self.analysis(player);
            if a.board_hash != bh { continue; }
            let pi = match self.toggles.data_source {
                DataSource::Mcts => &a.mcts_pi,
                DataSource::Nn => &a.nn_pi,
            };
            if !pi.is_empty() {
                return pi;
            }
        }
        &[]
    }

    fn policy_bars_data(&self) -> (Vec<f32>, ratatui::style::Color) {
        let pi = self.active_pi();
        if pi.is_empty() {
            return (Vec::new(), theme::UI_DIM);
        }
        let bars = self.scale_pi(pi);
        let cp = self.displayed_current_player();
        let color = if cp == 1 { theme::P1 } else { theme::P2 };
        (bars, color)
    }

    fn value_for_player_p1_perspective(&self, player: i8) -> f32 {
        use crate::status_bar::DataSource;
        let a = self.analysis(player);
        let raw = match self.toggles.data_source {
            DataSource::Mcts => a.mcts_value,
            DataSource::Nn => a.nn_value,
        };
        if a.current_player == 1 { raw } else { -raw }
    }

    fn rollouts_for_player(&self, player: i8) -> usize {
        self.analysis(player).rollouts
    }

    // ── Rendering ──

    fn render(&mut self, frame: &mut ratatui::Frame) {
        let size = frame.area();
        self.last_screen_size = (size.width, size.height);

        let show_p1_bar = self.toggles.eval_mode.show_p1();
        let show_p2_bar = self.toggles.eval_mode.show_p2();
        let n_bars = show_p1_bar as u16 + show_p2_bar as u16;
        let eval_col_width = if n_bars > 0 { 3 + n_bars * 5 } else { 0 };
        let toggles_width: u16 = 18;

        let mut h_constraints = vec![];
        if eval_col_width > 0 {
            h_constraints.push(Constraint::Length(eval_col_width));
        }
        h_constraints.push(Constraint::Min(20));
        if self.toggles.show_moves {
            h_constraints.push(Constraint::Length(self.sidebar_width));
        }
        h_constraints.push(Constraint::Length(toggles_width));

        let h_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(h_constraints)
            .split(size);

        let mut col_idx = 0;
        let eval_area = if eval_col_width > 0 { col_idx += 1; Some(h_chunks[col_idx - 1]) } else { None };
        let board_col = h_chunks[col_idx]; col_idx += 1;
        let moves_col = if self.toggles.show_moves { col_idx += 1; Some(h_chunks[col_idx - 1]) } else { None };
        let toggles_area = h_chunks[col_idx];

        if let Some(eval_area) = eval_area {
            let player_box_h: u16 = 4;
            let eval_vert = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(player_box_h), Constraint::Min(3)])
                .split(eval_area);

            frame.render_widget(CurrentPlayerWidget {
                current_player: self.displayed_current_player(),
                game_over: self.game.is_game_over(),
            }, eval_vert[0]);

            let bar_constraints: Vec<Constraint> = (0..n_bars).map(|_| Constraint::Ratio(1, n_bars as u32)).collect();
            let bar_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(bar_constraints)
                .split(eval_vert[1]);

            let mut bar_idx = 0;
            for &(player, show) in &[(1i8, show_p1_bar), (2i8, show_p2_bar)] {
                if show {
                    frame.render_widget(EvalBarWidget {
                        value: self.value_for_player_p1_perspective(player),
                        rollouts: self.rollouts_for_player(player),
                        visible: true,
                        braille: self.toggles.braille_bars,
                    }, bar_chunks[bar_idx]);
                    bar_idx += 1;
                }
            }
        }

        let board_vert = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(5), Constraint::Length(self.policy_bars_height)])
            .split(board_col);

        let board_area = board_vert[0];
        self.policy_bars_area = board_vert[1];

        let board_block = Block::default()
            .borders(Borders::ALL)
            .border_set(theme::BORDER)
            .border_style(Style::default().fg(theme::UI));
        let board_inner = board_block.inner(board_area);
        frame.render_widget(board_block, board_area);
        self.board_area = board_inner;

        // Build view and resolve styles using the shared engine
        let view = self.build_board_view();
        let styled_cells = styling::resolve_styles(&view);
        self.renderer.fit(self.config.board_size, board_inner);
        self.renderer.render(&styled_cells, board_inner, frame.buffer_mut());

        // Policy bars
        let (bars, bar_color) = self.policy_bars_data();
        frame.render_widget(PolicyBarsWidget {
            bars: &bars,
            player_color: bar_color,
            visible: true,
        }, self.policy_bars_area);

        if let Some(moves_col) = moves_col {
            self.sidebar_area = moves_col;

            let sidebar_constraints = if self.toggles.show_pv {
                vec![Constraint::Min(5), Constraint::Length(self.pv_height)]
            } else {
                vec![Constraint::Min(5)]
            };
            let sidebar_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(sidebar_constraints)
                .split(moves_col);

            self.move_list_area = sidebar_chunks[0];
            frame.render_widget(MoveListWidget {
                moves: self.game.move_records(),
                view_index: self.view_index,
                scroll_offset: self.move_list_scroll,
            }, sidebar_chunks[0]);

            if self.toggles.show_pv && sidebar_chunks.len() > 1 {
                self.pv_area = sidebar_chunks[1];
                frame.render_widget(PVWidget {
                    pv_moves: &self.pv_moves,
                    visible: true,
                }, sidebar_chunks[1]);
            } else {
                self.pv_area = Rect::default();
            }
        } else {
            self.sidebar_area = Rect::default();
            self.move_list_area = Rect::default();
            self.pv_area = Rect::default();
        }

        self.toggles_area = toggles_area;
        frame.render_widget(TogglesPanelWidget {
            toggles: &self.toggles,
            game_over: self.game.is_game_over(),
        }, toggles_area);

        if let Some(modal) = &self.temperature_modal {
            frame.render_widget(TemperatureModalWidget { modal }, size);
        }
    }

    // ── Main loop ──

    pub fn run(&mut self) -> anyhow::Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;

        for player in [1i8, 2] {
            self.ensure_ai(player);
        }

        self.advance_turn();

        let tick = Duration::from_millis(33);
        while !self.should_quit {
            terminal.draw(|f| self.render(f))?;

            if self.temperature_modal.is_some() {
                let event = input::poll_modal_event(tick);
                self.handle_modal_event(event);
            } else {
                let event = input::poll_event(tick, self.board_area, self.config.board_size);
                self.handle_event(event);
            }
            self.poll_ai_updates();
        }

        if let Some(ai) = self.ai_p1.take() { ai.shutdown(); }
        if let Some(ai) = self.ai_p2.take() { ai.shutdown(); }
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
        terminal.show_cursor()?;

        Ok(())
    }
}
