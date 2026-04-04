use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Widget};

use crate::theme;

/// Which eval bars to show.
#[derive(Clone, Copy, PartialEq)]
pub enum EvalMode {
    Off,
    P1,
    P2,
    Both,
}

impl EvalMode {
    pub fn cycle(&self, p1_is_ai: bool, p2_is_ai: bool) -> Self {
        let options: Vec<EvalMode> = std::iter::once(EvalMode::Off)
            .chain(p1_is_ai.then_some(EvalMode::P1))
            .chain(p2_is_ai.then_some(EvalMode::P2))
            .chain((p1_is_ai && p2_is_ai).then_some(EvalMode::Both))
            .collect();
        let current_idx = options.iter().position(|m| m == self).unwrap_or(0);
        options[(current_idx + 1) % options.len()]
    }

    pub fn label(&self) -> &'static str {
        match self {
            EvalMode::Off => "OFF",
            EvalMode::P1 => "P1",
            EvalMode::P2 => "P2",
            EvalMode::Both => "P1+P2",
        }
    }

    pub fn show_p1(&self) -> bool { matches!(self, EvalMode::P1 | EvalMode::Both) }
    pub fn show_p2(&self) -> bool { matches!(self, EvalMode::P2 | EvalMode::Both) }
    pub fn any(&self) -> bool { !matches!(self, EvalMode::Off) }
}

/// Whether to display raw NN output or MCTS search results.
#[derive(Clone, Copy, PartialEq)]
pub enum DataSource {
    Mcts,
    Nn,
}

impl DataSource {
    pub fn toggle(&self) -> Self {
        match self {
            DataSource::Mcts => DataSource::Nn,
            DataSource::Nn => DataSource::Mcts,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            DataSource::Mcts => "MCTS",
            DataSource::Nn => "NN",
        }
    }
}

/// How policy weights are mapped to colors.
#[derive(Clone, Copy, PartialEq)]
pub enum PolicyScale {
    /// Raw probabilities (normalized to max).
    Raw,
    /// Rank-based: top move = 1.0, linear decrease by rank.
    Rank,
    /// Rescaled: min maps to 0, max maps to 1 — blows up small differences.
    Spread,
}

impl PolicyScale {
    pub fn cycle(&self) -> Self {
        match self {
            PolicyScale::Raw => PolicyScale::Rank,
            PolicyScale::Rank => PolicyScale::Spread,
            PolicyScale::Spread => PolicyScale::Raw,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            PolicyScale::Raw => "raw",
            PolicyScale::Rank => "rank",
            PolicyScale::Spread => "spread",
        }
    }
}

pub struct ToggleState {
    pub eval_mode: EvalMode,
    pub policy_mode: EvalMode,
    pub policy_scale: PolicyScale,
    pub data_source: DataSource,
    pub show_pv: bool,
    pub show_moves: bool,
    pub pondering: bool,
    pub braille_bars: bool,
    pub sampling: bool,
    pub sampling_temperature: f32,
}

impl Default for ToggleState {
    fn default() -> Self {
        Self {
            eval_mode: EvalMode::P1,
            policy_mode: EvalMode::Off,
            policy_scale: PolicyScale::Raw,
            data_source: DataSource::Mcts,
            show_pv: false,
            show_moves: true,
            pondering: false,
            braille_bars: true,
            sampling: false,
            sampling_temperature: 0.5,
        }
    }
}

impl ToggleState {
    pub fn temperature(&self) -> Option<f32> {
        if self.sampling { Some(self.sampling_temperature) } else { None }
    }

    pub fn toggle_sampling(&mut self) {
        self.sampling = !self.sampling;
    }

    pub fn set_temperature(&mut self, temp: Option<f32>) {
        match temp {
            None => self.sampling = false,
            Some(t) => {
                self.sampling = true;
                self.sampling_temperature = t;
            }
        }
    }
}

// ── Toggles Panel (vertical, always visible on far right) ──

pub struct TogglesPanelWidget<'a> {
    pub toggles: &'a ToggleState,
    pub game_over: bool,
}

impl Widget for TogglesPanelWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_set(theme::BORDER)
            .border_style(Style::default().fg(theme::UI));
        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height == 0 || inner.width < 8 { return; }

        let mut row = inner.y;
        let x = inner.x;
        let w = inner.width as usize;

        let mut line = |key: &str, label: &str, value: &str, val_color| {
            if row >= inner.y + inner.height { return; }
            let line = Line::from(vec![
                Span::styled(format!("[{key}]"), Style::default().fg(theme::UI)),
                Span::styled(format!(" {label} "), Style::default().fg(theme::UI_DIM)),
                Span::styled(value.to_string(), Style::default().fg(val_color)),
            ]);
            buf.set_line(x, row, &line, w as u16);
            row += 1;
        };

        // Eval
        let eval_on = self.toggles.eval_mode.any();
        line("E", "Eval", self.toggles.eval_mode.label(),
            if eval_on { theme::TOGGLE_ON } else { theme::TOGGLE_OFF });

        // Data source
        let is_nn = self.toggles.data_source == DataSource::Nn;
        line("D", "Data", self.toggles.data_source.label(),
            if is_nn { theme::LEGAL_MOVE } else { theme::TOGGLE_OFF });

        // Policy
        let policy_on = self.toggles.policy_mode.any();
        line("P", "Policy", self.toggles.policy_mode.label(),
            if policy_on { theme::TOGGLE_ON } else { theme::TOGGLE_OFF });

        // Policy scale (only meaningful when policy is on)
        if policy_on {
            line("G", "Scale", self.toggles.policy_scale.label(),
                if self.toggles.policy_scale == PolicyScale::Rank { theme::LEGAL_MOVE } else { theme::TOGGLE_OFF });
        }

        // PV
        line("V", "PV",
            if self.toggles.show_pv { "ON" } else { "OFF" },
            if self.toggles.show_pv { theme::TOGGLE_ON } else { theme::TOGGLE_OFF });

        // Moves
        line("M", "Moves",
            if self.toggles.show_moves { "ON" } else { "OFF" },
            if self.toggles.show_moves { theme::TOGGLE_ON } else { theme::TOGGLE_OFF });

        // Temperature
        let (temp_val, temp_color) = if self.toggles.sampling {
            (format!("T={:.2}", self.toggles.sampling_temperature), theme::TOGGLE_ON)
        } else {
            ("argmax".to_string(), theme::TOGGLE_OFF)
        };
        line("S", "Play", &temp_val, temp_color);

        // Ponder
        line("T", "Ponder",
            if self.toggles.pondering { "ON" } else { "OFF" },
            if self.toggles.pondering { theme::TOGGLE_ON } else { theme::TOGGLE_OFF });

        // Renderer
        line("R", "Render", "", theme::UI_DIM);

        // Braille bars
        line("B", "Bars",
            if self.toggles.braille_bars { "braille" } else { "block" },
            if self.toggles.braille_bars { theme::TOGGLE_ON } else { theme::TOGGLE_OFF });

        // Separator
        if row < inner.y + inner.height {
            row += 1;
        }

        // History hint
        if row < inner.y + inner.height {
            let hint = Line::from(Span::styled("[←→] History", Style::default().fg(theme::UI_DIM)));
            buf.set_line(x, row, &hint, w as u16);
            row += 1;
        }

        // New / Quit
        if row < inner.y + inner.height {
            let nq = Line::from(vec![
                Span::styled("[N]", Style::default().fg(theme::UI)),
                Span::styled("ew ", Style::default().fg(theme::UI_DIM)),
                Span::styled("[Q]", Style::default().fg(theme::UI)),
                Span::styled("uit", Style::default().fg(theme::UI_DIM)),
            ]);
            buf.set_line(x, row, &nq, w as u16);
            row += 1;
        }

        // Game over indicator
        if self.game_over && row < inner.y + inner.height {
            row += 1;
            if row < inner.y + inner.height {
                let go = Line::from(Span::styled("Game Over", Style::default().fg(theme::TOGGLE_ON)));
                buf.set_line(x, row, &go, w as u16);
            }
        }
    }
}

// ── Current Player Box (small box above/below eval bars) ──

pub struct CurrentPlayerWidget {
    pub current_player: i8,
    pub game_over: bool,
}

impl Widget for CurrentPlayerWidget {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_set(theme::BORDER)
            .border_style(Style::default().fg(theme::UI));
        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height == 0 || inner.width < 2 { return; }

        let (label, color) = if self.game_over {
            ("GG", theme::TOGGLE_ON)
        } else {
            match self.current_player {
                1 => ("P1", theme::P1),
                _ => ("P2", theme::P2),
            }
        };

        let glyph_line = Line::from(Span::styled(theme::GLYPH_PIECE, Style::default().fg(color)));
        let label_line = Line::from(Span::styled(label, Style::default().fg(color)));

        let center_row = inner.y + inner.height / 2;
        if center_row > inner.y {
            let cx = inner.x + inner.width.saturating_sub(1) / 2;
            buf.set_line(cx, center_row - 1, &glyph_line, inner.width);
            buf.set_line(inner.x + inner.width.saturating_sub(label.len() as u16) / 2, center_row, &label_line, inner.width);
        }
    }
}

// ── Policy Histogram (vertical bars growing upward from bottom) ──

pub struct PolicyBarsWidget<'a> {
    /// Normalized weights per legal move (0..1)
    pub bars: &'a [f32],
    pub player_color: ratatui::style::Color,
    pub visible: bool,
}

impl Widget for PolicyBarsWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if !self.visible || self.bars.is_empty() || area.height < 2 || area.width < 2 {
            return;
        }

        let block = Block::default()
            .borders(Borders::ALL)
            .border_set(theme::BORDER)
            .border_style(Style::default().fg(theme::UI));
        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height == 0 || inner.width == 0 { return; }

        let n_bars = self.bars.len();
        // Each bar is 1 terminal cell wide. Each cell = 4 braille dots vertically.
        let total_width = n_bars as u16;
        if total_width == 0 { return; }
        let start_x = inner.x + inner.width.saturating_sub(total_width) / 2;
        let style = Style::default().fg(self.player_color);

        // Total dot height = inner.height * 4 (braille gives 4 vertical dots per cell)
        let dot_height = inner.height as f32 * 4.0;

        // Braille dot bits for left column, bottom to top: dot6, dot2, dot1, dot0
        // In the Unicode braille pattern, the bit positions are:
        //   dot0 = bit 0 (0x01)    top-left
        //   dot1 = bit 1 (0x02)    mid-upper-left
        //   dot2 = bit 2 (0x04)    mid-lower-left
        //   dot3 = bit 3 (0x08)    top-right
        //   dot4 = bit 4 (0x10)    mid-upper-right
        //   dot5 = bit 5 (0x20)    mid-lower-right
        //   dot6 = bit 6 (0x40)    bottom-left
        //   dot7 = bit 7 (0x80)    bottom-right
        //
        // For a single-width bar using both columns, the 4 rows (top to bottom) are:
        //   row 0: dot0 + dot3  (0x09)
        //   row 1: dot1 + dot4  (0x12)
        //   row 2: dot2 + dot5  (0x24)
        //   row 3: dot6 + dot7  (0xC0)
        const ROW_BITS: [u8; 4] = [0x09, 0x12, 0x24, 0xC0];

        for (i, &weight) in self.bars.iter().enumerate() {
            let bx = start_x + i as u16;
            if bx >= inner.x + inner.width { break; }

            let fill_dots = (weight * dot_height).round() as usize;

            // Fill from bottom up across terminal cells
            for cell_row in 0..inner.height {
                let term_row = inner.y + inner.height - 1 - cell_row;
                let cell_base_dot = cell_row as usize * 4; // bottom dot index for this cell
                let mut bits: u8 = 0;

                // The 4 dots in this cell, bottom to top: indices 3,2,1,0
                for dot_in_cell in 0..4u8 {
                    let dot_from_bottom = cell_base_dot + dot_in_cell as usize;
                    if dot_from_bottom < fill_dots {
                        // dot_in_cell 0 = bottom of cell = ROW_BITS[3]
                        // dot_in_cell 3 = top of cell = ROW_BITS[0]
                        bits |= ROW_BITS[3 - dot_in_cell as usize];
                    }
                }

                if bits != 0 {
                    let ch = char::from_u32(0x2800 + bits as u32).unwrap_or(' ');
                    let mut s = [0u8; 4];
                    buf.set_string(bx, term_row, ch.encode_utf8(&mut s), style);
                }
            }
        }
    }
}

// ── Keep the old StatusBarWidget for backward compat but it's no longer primary ──

pub struct StatusBarWidget<'a> {
    pub toggles: &'a ToggleState,
    pub game_over: bool,
    pub current_player: i8,
}

impl Widget for StatusBarWidget<'_> {
    fn render(self, _area: Rect, _buf: &mut Buffer) {
        // No longer used — toggles panel replaces this
    }
}
