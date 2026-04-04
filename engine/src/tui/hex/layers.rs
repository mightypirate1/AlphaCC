use ratatui::style::Color;

use crate::cc::HexCoord;
use crate::cc::game::board::BoardMatrix;
use crate::tui::theme;

/// Visual properties for a single hex cell, as determined by the layer stack.
#[derive(Clone)]
pub struct HexStyle {
    pub fill: Color,
    pub border: Color,
    /// Optional centered glyph (for glyph backend or future use).
    pub glyph: Option<&'static str>,
}

impl Default for HexStyle {
    fn default() -> Self {
        Self {
            fill: theme::BG,
            border: theme::EMPTY_CELL,
            glyph: None,
        }
    }
}

/// Input data that layers use to decide styling.
pub struct BoardView {
    pub matrix: BoardMatrix,
    pub board_size: u8,
    pub current_player: i8,
    pub selected_piece: Option<HexCoord>,
    pub legal_destinations: Vec<HexCoord>,
    pub last_move: Option<(HexCoord, HexCoord, i8)>,
    /// (destination_coord, normalized_weight 0..1)
    pub policy: Vec<(HexCoord, f32)>,
    /// Hovered move: (from_display, to_display) to highlight on the board
    pub hovered_move: Option<(HexCoord, HexCoord)>,
}

/// Compute the final style for every hex on the board.
/// Layers are applied in order — later layers override earlier ones.
pub fn resolve_styles(view: &BoardView) -> Vec<(u8, u8, HexStyle)> {
    let s = view.board_size;
    let mut result = Vec::with_capacity(s as usize * s as usize);

    for x in 0..s {
        for y in 0..s {
            let mut style = base_style(&view.matrix, x, y, view.board_size);

            // Last move — bright for destination, dim for origin
            if let Some((from, to, player)) = view.last_move {
                let coord = HexCoord::new(x, y, s);
                if coord == to {
                    let color = theme::last_move_color(player);
                    style.fill = color;
                    style.border = color;
                } else if coord == from {
                    let base = if player == 1 { theme::P1 } else { theme::P2 };
                    let color = darken(base, 0.5);
                    style.fill = color;
                    style.border = darken(color, 0.3);
                }
            }

            // Policy heatmap — lerp from current color toward player's bright color
            if let Some(&(_, weight)) = view.policy.iter().find(|(c, _)| *c == HexCoord::new(x, y, s)) {
                let target = if view.current_player == 1 { theme::P1 } else { theme::P2 };
                style.fill = lerp_color(style.fill, target, weight);
                style.border = lerp_color(style.border, target, weight);
            }

            // Legal destinations — cyan
            if view.legal_destinations.contains(&HexCoord::new(x, y, s)) {
                style.fill = theme::LEGAL_MOVE;
                style.border = theme::LEGAL_MOVE;
            }

            // Hovered move from policy bars — highlight from/to
            if let Some((from, to)) = view.hovered_move {
                let coord = HexCoord::new(x, y, s);
                if coord == to {
                    style.fill = theme::LEGAL_MOVE;
                    style.border = theme::LEGAL_MOVE;
                } else if coord == from {
                    style.fill = theme::brighten(theme::LEGAL_MOVE, 0.3);
                    style.border = theme::LEGAL_MOVE;
                }
            }

            // Selected piece — bright highlight (overrides hover)
            if view.selected_piece == Some(HexCoord::new(x, y, s)) {
                style.fill = theme::SELECTED;
                style.border = theme::SELECTED;
            }

            result.push((x, y, style));
        }
    }
    result
}

fn base_style(matrix: &BoardMatrix, x: u8, y: u8, board_size: u8) -> HexStyle {
    let content = matrix[x as usize][y as usize];
    match content {
        1 => HexStyle {
            fill: theme::P1,
            border: darken(theme::P1, 0.3),
            glyph: Some(theme::GLYPH_PIECE),
        },
        2 => HexStyle {
            fill: theme::P2,
            border: darken(theme::P2, 0.3),
            glyph: Some(theme::GLYPH_PIECE),
        },
        _ => {
            // Empty cell: tint with player color if it's in a home region
            let home = home_region(x, y, board_size);
            let (fill, border) = match home {
                1 => (darken(theme::P1, 0.75), darken(theme::P1, 0.55)),
                2 => (darken(theme::P2, 0.75), darken(theme::P2, 0.55)),
                _ => (theme::EMPTY_CELL, darken(theme::EMPTY_CELL, 0.3)),
            };
            HexStyle { fill, border, glyph: Some(theme::GLYPH_EMPTY) }
        }
    }
}

/// Returns 1 if (x,y) is in P1's home, 2 if P2's home, 0 otherwise.
/// Matches Board::xy_start_val logic.
fn home_region(x: u8, y: u8, board_size: u8) -> i8 {
    let home_size = (board_size - 1) / 2;
    if x + y < home_size {
        return 1;
    }
    if x + y >= home_size + board_size && x < board_size && y < board_size {
        return 2;
    }
    0
}

fn lerp_color(from: Color, to: Color, t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    if let (Color::Rgb(r1, g1, b1), Color::Rgb(r2, g2, b2)) = (from, to) {
        let l = |a: u8, b: u8| (a as f32 * (1.0 - t) + b as f32 * t) as u8;
        Color::Rgb(l(r1, r2), l(g1, g2), l(b1, b2))
    } else {
        to
    }
}

fn darken(color: Color, amount: f32) -> Color {
    let a = 1.0 - amount.clamp(0.0, 1.0);
    if let Color::Rgb(r, g, b) = color {
        Color::Rgb(
            (r as f32 * a) as u8,
            (g as f32 * a) as u8,
            (b as f32 * a) as u8,
        )
    } else {
        color
    }
}
