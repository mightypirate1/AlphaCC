use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::widgets::Widget;

use alpha_cc_core::BoardMatrix;
use alpha_cc_core::HexCoord;
use crate::theme;

/// Visual overlays applied on top of the base board.
pub struct BoardOverlays {
    pub selected_piece: Option<HexCoord>,
    pub legal_destinations: Vec<HexCoord>,
    /// (from, to, player who made the move)
    pub last_move: Option<(HexCoord, HexCoord, i8)>,
    /// (destination_coord, normalized_weight) for policy heatmap
    pub policy: Vec<(HexCoord, f32)>,
    pub current_player: i8,
}

impl Default for BoardOverlays {
    fn default() -> Self {
        Self {
            selected_piece: None,
            legal_destinations: Vec::new(),
            last_move: None,
            policy: Vec::new(),
            current_player: 1,
        }
    }
}

pub struct BoardWidget<'a> {
    matrix: &'a BoardMatrix,
    size: u8,
    overlays: &'a BoardOverlays,
}

impl<'a> BoardWidget<'a> {
    pub fn new(matrix: &'a BoardMatrix, size: u8, overlays: &'a BoardOverlays) -> Self {
        Self { matrix, size, overlays }
    }

    fn cell_style(&self, x: u8, y: u8, content: i8) -> (Style, &'static str) {
        let coord = HexCoord::new(x, y, self.size);

        // Check overlays in priority order
        if self.overlays.selected_piece == Some(coord) {
            return (Style::default().fg(theme::SELECTED), theme::GLYPH_PIECE);
        }

        if self.overlays.legal_destinations.contains(&coord) {
            return (Style::default().fg(theme::LEGAL_MOVE), theme::GLYPH_PIECE);
        }

        // Policy heatmap on empty cells or destination cells
        if let Some(&(_, weight)) = self.overlays.policy.iter().find(|(c, _)| *c == coord) {
            let base = if self.overlays.current_player == 1 { theme::P1 } else { theme::P2 };
            let color = theme::policy_color(base, weight);
            let glyph = if content == 0 { theme::GLYPH_EMPTY } else { theme::GLYPH_PIECE };
            return (Style::default().fg(color), glyph);
        }

        // Last move highlight — bright for destination, dim for origin
        if let Some((from, to, player)) = self.overlays.last_move {
            if coord == to {
                let glyph = if content == 0 { theme::GLYPH_EMPTY } else { theme::GLYPH_PIECE };
                return (Style::default().fg(theme::last_move_color(player)), glyph);
            } else if coord == from {
                let base = if player == 1 { theme::P1 } else { theme::P2 };
                let dim = theme::policy_color(base, 0.3);
                let glyph = if content == 0 { theme::GLYPH_EMPTY } else { theme::GLYPH_PIECE };
                return (Style::default().fg(dim), glyph);
            }
        }

        // Base rendering
        match content {
            1 => (Style::default().fg(theme::P1), theme::GLYPH_PIECE),
            2 => (Style::default().fg(theme::P2), theme::GLYPH_PIECE),
            _ => (Style::default().fg(theme::EMPTY_CELL), theme::GLYPH_EMPTY),
        }
    }
}

impl Widget for BoardWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let s = self.size as usize;
        for x in 0..s {
            for y in 0..s {
                let col = area.x + (x as u16) + (y as u16) * 2;
                let row = area.y + x as u16;
                if col + 1 < area.x + area.width && row < area.y + area.height {
                    let content = self.matrix[x][y];
                    let (style, glyph) = self.cell_style(x as u8, y as u8, content);
                    buf.set_string(col, row, glyph, style);
                }
            }
        }
    }
}

/// Minimum area needed to render a board of the given size.
pub fn board_min_size(board_size: u8) -> (u16, u16) {
    let s = board_size as u16;
    // Last row is indented by (s-1) and has s cells of 2 chars each
    let width = (s - 1) + s * 2;
    let height = s;
    (width, height)
}
