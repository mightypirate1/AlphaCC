use std::io;
use std::time::Duration;

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

use crate::cc::game::board::BoardMatrix;
use crate::cc::game::moves::find_all_moves;
use crate::cc::HexCoord;
use crate::tui::agent::{AiHandle, AiUpdate};
use crate::tui::board_widget::{BoardOverlays, BoardWidget, board_min_size};
use crate::tui::game::GameState;
use crate::tui::input::{self, AppEvent};
use crate::tui::modal::{TemperatureModal, TemperatureModalWidget};
use crate::tui::sidebar::{EvalBarWidget, MoveListWidget, PVWidget};
use crate::tui::status_bar::{StatusBarWidget, ToggleState};
use crate::tui::theme;

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
    SidebarLeft,  // dragging the left edge of the sidebar
    PvTop,        // dragging the top edge of the PV panel
}

// ── App state machine ──

#[derive(Clone)]
enum Phase {
    HumanTurn,
    PieceSelected {
        piece: HexCoord,
        /// (action_index, display_destination)
        legal: Vec<(usize, HexCoord)>,
    },
    AiThinking,
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
    pub c_puct_init: f32,
    pub c_puct_base: f32,
    pub pruning_tree: bool,
}

// ── Main App ──

pub struct App {
    config: AppConfig,
    game: GameState,
    phase: Phase,
    toggles: ToggleState,
    view_index: usize,
    ai_p1: Option<AiHandle>,
    ai_p2: Option<AiHandle>,
    latest_value: f32,
    latest_pi: Vec<f32>,
    latest_rollouts: usize,
    pv_moves: Vec<String>,
    board_area: Rect,
    last_screen_size: (u16, u16),
    temperature_modal: Option<TemperatureModal>,
    // Resizable panel sizes and stored rects for border hit-testing
    sidebar_width: u16,
    pv_height: u16,
    sidebar_area: Rect,
    pv_area: Rect,
    dragging: Option<DragTarget>,
    should_quit: bool,
}

impl App {
    pub fn new(config: AppConfig) -> Self {
        let game = GameState::new(config.board_size);
        let has_ai = !config.p1.is_human() || !config.p2.is_human();
        let toggles = ToggleState {
            show_eval: has_ai,
            ..Default::default()
        };
        Self {
            config,
            game,
            phase: Phase::HumanTurn,
            toggles,
            view_index: 0,
            ai_p1: None,
            ai_p2: None,
            latest_value: 0.0,
            latest_pi: Vec::new(),
            latest_rollouts: 0,
            pv_moves: Vec::new(),
            board_area: Rect::default(),
            last_screen_size: (80, 24),
            temperature_modal: None,
            sidebar_width: 22,
            pv_height: 8,
            sidebar_area: Rect::default(),
            pv_area: Rect::default(),
            dragging: None,
            should_quit: false,
        }
    }

    fn player_config(&self, player: i8) -> &PlayerConfig {
        if player == 1 { &self.config.p1 } else { &self.config.p2 }
    }

    fn mcts_params(&self) -> crate::cc::rollouts::mcts::MCTSParams {
        crate::cc::rollouts::mcts::MCTSParams {
            gamma: self.config.gamma,
            dirichlet_weight: 0.0,
            dirichlet_leaf_weight: 0.0,
            dirichlet_alpha: 0.15,
            c_puct_init: self.config.c_puct_init,
            c_puct_base: self.config.c_puct_base,
        }
    }

    fn ensure_ai(&mut self, player: i8) {
        let slot = if player == 1 { &self.ai_p1 } else { &self.ai_p2 };
        if slot.is_some() {
            return;
        }
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
        );
        if player == 1 { self.ai_p1 = Some(ai); } else { self.ai_p2 = Some(ai); }
    }

    fn ai_for(&self, player: i8) -> Option<&AiHandle> {
        if player == 1 { self.ai_p1.as_ref() } else { self.ai_p2.as_ref() }
    }

    fn start_ai_turn(&mut self) {
        let player = self.game.current_board().get_info().current_player;
        self.ensure_ai(player);
        if let Some(ai) = self.ai_for(player) {
            ai.start_thinking(self.game.current_board().clone(), self.config.think_time, self.toggles.temperature());
            self.phase = Phase::AiThinking;
            self.latest_pi.clear();
            self.latest_rollouts = 0;
        }
    }

    fn apply_move(&mut self, action_index: usize) {
        self.game.apply_move(action_index);
        self.view_index = self.game.len() - 1;
        self.latest_pi.clear();
        self.pv_moves.clear();
        self.latest_rollouts = 0;

        let board = self.game.current_board();
        if let Some(ai) = &self.ai_p1 { ai.notify_move_applied(board); }
        if let Some(ai) = &self.ai_p2 { ai.notify_move_applied(board); }

        if self.game.is_game_over() {
            self.phase = Phase::GameOver;
        } else {
            self.advance_turn();
        }
    }

    fn advance_turn(&mut self) {
        let player = self.game.current_board().get_info().current_player;
        match self.player_config(player) {
            PlayerConfig::Human => self.phase = Phase::HumanTurn,
            PlayerConfig::Ai { .. } => self.start_ai_turn(),
        }
    }

    fn viewing_live(&self) -> bool {
        self.view_index == self.game.len() - 1
    }

    fn displayed_board_matrix(&self) -> BoardMatrix {
        self.game.board_at(self.view_index).get_unflipped_matrix()
    }

    fn displayed_current_player(&self) -> i8 {
        self.game.board_at(self.view_index).get_info().current_player
    }

    // ── Event handling ──

    fn handle_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::Quit => self.should_quit = true,
            AppEvent::ToggleEval => self.toggles.show_eval = !self.toggles.show_eval,
            AppEvent::TogglePolicy => self.toggles.show_policy = !self.toggles.show_policy,
            AppEvent::TogglePV => self.toggles.show_pv = !self.toggles.show_pv,
            AppEvent::ToggleSampling => self.toggles.toggle_sampling(),
            AppEvent::OpenTemperatureModal => {
                let mut modal = TemperatureModal::default();
                modal.value = self.toggles.sampling_temperature;
                self.temperature_modal = Some(modal);
            }
            AppEvent::HistoryBack => {
                if self.view_index > 0 { self.view_index -= 1; }
            }
            AppEvent::HistoryForward => {
                if self.view_index < self.game.len() - 1 { self.view_index += 1; }
            }
            AppEvent::HistoryStart => self.view_index = 0,
            AppEvent::HistoryEnd => self.view_index = self.game.len() - 1,
            AppEvent::NewGame => {
                if let Some(ai) = &self.ai_p1 { ai.cancel(); }
                if let Some(ai) = &self.ai_p2 { ai.cancel(); }
                self.game.reset();
                self.view_index = 0;
                self.latest_pi.clear();
                self.pv_moves.clear();
                self.latest_value = 0.0;
                self.latest_rollouts = 0;
                self.advance_turn();
            }
            AppEvent::CellClicked(coord) => self.handle_click(coord),
            AppEvent::MouseDown(col, row) => self.handle_mouse_down(col, row),
            AppEvent::MouseDrag(col, row) => self.handle_mouse_drag(col, row),
            AppEvent::MouseUp => self.dragging = None,
            AppEvent::PauseResume | AppEvent::Tick | AppEvent::Resize(_, _) => {}
            // Modal events handled by handle_modal_event when modal is open
            AppEvent::ModalLeft | AppEvent::ModalRight | AppEvent::ModalConfirm
            | AppEvent::ModalCancel | AppEvent::ModalClick(_, _) => {}
        }
    }

    fn handle_mouse_down(&mut self, col: u16, row: u16) {
        // Check if click is on the left border of the sidebar (±1 col tolerance)
        let sb = self.sidebar_area;
        if col.abs_diff(sb.x) <= 1 && row >= sb.y && row < sb.y + sb.height {
            self.dragging = Some(DragTarget::SidebarLeft);
            return;
        }
        // Check if click is on the top border of the PV panel
        let pv = self.pv_area;
        if pv.width > 0 && row.abs_diff(pv.y) <= 1 && col >= pv.x && col < pv.x + pv.width {
            self.dragging = Some(DragTarget::PvTop);
            return;
        }
    }

    fn handle_mouse_drag(&mut self, col: u16, _row: u16) {
        let Some(target) = self.dragging else { return };
        match target {
            DragTarget::SidebarLeft => {
                let screen_w = self.last_screen_size.0;
                let new_width = screen_w.saturating_sub(col);
                self.sidebar_width = new_width.clamp(14, screen_w / 2);
            }
            DragTarget::PvTop => {
                let sb = self.sidebar_area;
                let sb_bottom = sb.y + sb.height;
                let new_pv = sb_bottom.saturating_sub(_row);
                self.pv_height = new_pv.clamp(3, sb.height.saturating_sub(4));
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
                // Check if click is on the slider track
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

    fn handle_click(&mut self, display_coord: HexCoord) {
        if !self.viewing_live() { return; }

        match &self.phase {
            Phase::HumanTurn => {
                let board = self.game.current_board();
                let info = board.get_info();
                let matrix = board.get_unflipped_matrix();
                let content = matrix[display_coord.x as usize][display_coord.y as usize];
                if content == info.current_player as i8 {
                    self.select_piece(display_coord);
                }
            }
            Phase::PieceSelected { legal, .. } => {
                let legal = legal.clone();
                if let Some((action_index, _)) = legal.iter().find(|(_, dest)| *dest == display_coord) {
                    self.apply_move(*action_index);
                } else {
                    let board = self.game.current_board();
                    let info = board.get_info();
                    let matrix = board.get_unflipped_matrix();
                    let content = matrix[display_coord.x as usize][display_coord.y as usize];
                    if content == info.current_player as i8 {
                        self.select_piece(display_coord);
                    } else {
                        self.phase = Phase::HumanTurn;
                    }
                }
            }
            _ => {}
        }
    }

    fn select_piece(&mut self, display_coord: HexCoord) {
        let board = self.game.current_board();
        let info = board.get_info();
        let moves = find_all_moves(board);

        let legal: Vec<(usize, HexCoord)> = moves.iter().enumerate().filter_map(|(i, mv)| {
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
        for player in [1i8, 2] {
            let update = match if player == 1 { &self.ai_p1 } else { &self.ai_p2 } {
                Some(ai) => ai.try_recv(),
                None => None,
            };
            if let Some(update) = update {
                match update {
                    AiUpdate::Progress(p) => {
                        self.latest_value = p.value;
                        self.latest_pi = p.pi;
                        self.latest_rollouts = p.total_rollouts;
                    }
                    AiUpdate::Done(result) => {
                        self.latest_value = result.value;
                        self.latest_pi = result.pi;
                        self.latest_rollouts = result.total_rollouts;
                        if matches!(self.phase, Phase::AiThinking) {
                            self.apply_move(result.action_index);
                        }
                    }
                }
            }
        }
    }

    // ── Build overlays ──

    fn build_overlays(&self) -> BoardOverlays {
        let mut overlays = BoardOverlays {
            current_player: self.displayed_current_player(),
            ..Default::default()
        };

        if let Phase::PieceSelected { piece, legal } = &self.phase {
            if self.viewing_live() {
                overlays.selected_piece = Some(*piece);
                overlays.legal_destinations = legal.iter().map(|(_, dest)| *dest).collect();
            }
        }

        if self.view_index > 0 {
            let rec = &self.game.move_records()[self.view_index - 1];
            let board = self.game.board_at(self.view_index - 1);
            let info = board.get_info();
            let player = info.current_player;
            let from = if player == 1 { rec.mv.from_coord } else { rec.mv.from_coord.flip() };
            let to = if player == 1 { rec.mv.to_coord } else { rec.mv.to_coord.flip() };
            overlays.last_move = Some((from, to, player));
        }

        if self.viewing_live() && self.toggles.show_policy && !self.latest_pi.is_empty() {
            let board = self.game.current_board();
            let info = board.get_info();
            let moves = find_all_moves(board);
            let max_pi = self.latest_pi.iter().cloned().fold(0.0f32, f32::max);
            if max_pi > 0.0 {
                overlays.policy = moves.iter().enumerate()
                    .filter(|(i, _)| *i < self.latest_pi.len())
                    .map(|(i, mv)| {
                        let dest = if info.current_player == 1 { mv.to_coord } else { mv.to_coord.flip() };
                        (dest, self.latest_pi[i] / max_pi)
                    })
                    .collect();
            }
        }

        overlays
    }

    // ── Rendering ──

    fn render(&mut self, frame: &mut ratatui::Frame) {
        let size = frame.area();
        self.last_screen_size = (size.width, size.height);

        let vert = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(1)])
            .split(size);

        let main_area = vert[0];
        let status_area = vert[1];

        let mut h_constraints = vec![];
        if self.toggles.show_eval {
            h_constraints.push(Constraint::Length(6));
        }
        h_constraints.push(Constraint::Min(20));
        h_constraints.push(Constraint::Length(self.sidebar_width));

        let h_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(h_constraints)
            .split(main_area);

        let (eval_area, board_area, sidebar_area) = if self.toggles.show_eval {
            (Some(h_chunks[0]), h_chunks[1], h_chunks[2])
        } else {
            (None, h_chunks[0], h_chunks[1])
        };

        let board_block = Block::default()
            .borders(Borders::ALL)
            .border_set(theme::BORDER)
            .border_style(Style::default().fg(theme::UI));
        let board_inner = board_block.inner(board_area);
        frame.render_widget(board_block, board_area);

        let (bw, bh) = board_min_size(self.config.board_size);
        let board_offset_x = (board_inner.width.saturating_sub(bw)) / 2;
        let board_offset_y = (board_inner.height.saturating_sub(bh)) / 2;
        let centered_board = Rect {
            x: board_inner.x + board_offset_x,
            y: board_inner.y + board_offset_y,
            width: bw.min(board_inner.width),
            height: bh.min(board_inner.height),
        };
        self.board_area = centered_board;

        let matrix = self.displayed_board_matrix();
        let overlays = self.build_overlays();
        let board_widget = BoardWidget::new(&matrix, self.config.board_size, &overlays);
        frame.render_widget(board_widget, centered_board);

        if let Some(eval_area) = eval_area {
            frame.render_widget(EvalBarWidget {
                value: self.latest_value,
                rollouts: self.latest_rollouts,
                visible: true,
            }, eval_area);
        }

        self.sidebar_area = sidebar_area;

        let sidebar_constraints = if self.toggles.show_pv {
            vec![Constraint::Min(5), Constraint::Length(self.pv_height)]
        } else {
            vec![Constraint::Min(5)]
        };
        let sidebar_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(sidebar_constraints)
            .split(sidebar_area);

        frame.render_widget(MoveListWidget {
            moves: self.game.move_records(),
            view_index: self.view_index,
            total_states: self.game.len(),
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

        frame.render_widget(StatusBarWidget {
            toggles: &self.toggles,
            game_over: self.game.is_game_over(),
            current_player: self.displayed_current_player(),
        }, status_area);

        // Modal overlay (rendered last, on top of everything)
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
