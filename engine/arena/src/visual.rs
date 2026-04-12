use ratatui::style::Color;

use alpha_cc_core::Board;

/// What the game tells the arena about a single cell's appearance.
pub struct CellVisual {
    pub glyph: &'static str,
    pub piece: Option<u8>,  // None=empty, 1=player1, 2=player2
    pub region: u8,         // 0=neutral, 1=region A (P1 home / light square), 2=region B
}

/// Computed style for a single cell, output of the shared styling engine.
#[derive(Clone)]
pub struct CellStyle {
    pub fill: Color,
    pub border: Color,
    pub glyph: &'static str,
}

/// A styled cell with its coordinate, ready for the renderer.
pub struct StyledCell<C> {
    pub coord: C,
    pub style: CellStyle,
}

/// Game-agnostic board view — input to the shared styling engine.
pub struct BoardView<C> {
    #[allow(dead_code)]
    pub board_size: u8,
    pub current_player: i8,
    pub cells: Vec<(C, CellVisual)>,
    pub selected: Option<C>,
    pub legal_destinations: Vec<C>,
    pub last_move: Option<(C, C, i8)>,
    pub policy: Vec<(C, f32)>,
    pub hovered_move: Option<(C, C)>,
}

/// Trait for mapping a game's typed cell content to visual properties.
/// Implemented per-game in the arena crate (e.g. `impl GameVisual for CCBoard`).
pub trait GameVisual: Board {
    fn cell_visual(content: Self::Content, coord: &Self::Coord, board_size: u8) -> CellVisual;
}
