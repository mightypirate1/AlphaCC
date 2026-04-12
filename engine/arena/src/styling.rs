use ratatui::style::Color;

use crate::theme;
use crate::visual::{BoardView, CellStyle, CellVisual, StyledCell};

/// Compute the final style for every cell on the board.
/// Layers are applied in order — later layers override earlier ones.
pub fn resolve_styles<C: Copy + Eq>(view: &BoardView<C>) -> Vec<StyledCell<C>> {
    let mut result = Vec::with_capacity(view.cells.len());

    for (coord, content) in &view.cells {
        let mut style = base_style(content);

        // Last move — bright for destination, dim for origin
        if let Some((from, to, player)) = view.last_move {
            if *coord == to {
                let color = theme::last_move_color(player);
                style.fill = color;
                style.border = color;
            } else if *coord == from {
                let base = if player == 1 { theme::P1 } else { theme::P2 };
                let color = darken(base, 0.5);
                style.fill = color;
                style.border = darken(color, 0.3);
            }
        }

        // Policy heatmap — lerp from current color toward player's bright color
        if let Some(&(_, weight)) = view.policy.iter().find(|(c, _)| c == coord) {
            let target = if view.current_player == 1 { theme::P1 } else { theme::P2 };
            style.fill = lerp_color(style.fill, target, weight);
            style.border = lerp_color(style.border, target, weight);
        }

        // Legal destinations — cyan
        if view.legal_destinations.contains(coord) {
            style.fill = theme::LEGAL_MOVE;
            style.border = theme::LEGAL_MOVE;
        }

        // Hovered move from policy bars — highlight from/to
        if let Some((from, to)) = view.hovered_move {
            if *coord == to {
                style.fill = theme::LEGAL_MOVE;
                style.border = theme::LEGAL_MOVE;
            } else if *coord == from {
                style.fill = theme::brighten(theme::LEGAL_MOVE, 0.3);
                style.border = theme::LEGAL_MOVE;
            }
        }

        // Selected piece — bright highlight (overrides hover)
        if view.selected == Some(*coord) {
            style.fill = theme::SELECTED;
            style.border = theme::SELECTED;
        }

        result.push(StyledCell { coord: *coord, style });
    }
    result
}

fn base_style(content: &CellVisual) -> CellStyle {
    match content.piece {
        Some(1) => CellStyle {
            fill: theme::P1,
            border: darken(theme::P1, 0.3),
            glyph: content.glyph,
        },
        Some(2) => CellStyle {
            fill: theme::P2,
            border: darken(theme::P2, 0.3),
            glyph: content.glyph,
        },
        _ => {
            let (fill, border) = match content.region {
                1 => (darken(theme::P1, 0.75), darken(theme::P1, 0.55)),
                2 => (darken(theme::P2, 0.75), darken(theme::P2, 0.55)),
                _ => (theme::EMPTY_CELL, darken(theme::EMPTY_CELL, 0.3)),
            };
            CellStyle { fill, border, glyph: content.glyph }
        }
    }
}

pub fn lerp_color(from: Color, to: Color, t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    if let (Color::Rgb(r1, g1, b1), Color::Rgb(r2, g2, b2)) = (from, to) {
        let l = |a: u8, b: u8| (a as f32 * (1.0 - t) + b as f32 * t) as u8;
        Color::Rgb(l(r1, r2), l(g1, g2), l(b1, b2))
    } else {
        to
    }
}

pub fn darken(color: Color, amount: f32) -> Color {
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
