use alpha_cc_core::cc::{CCBoard, CCContent, HexCoord};

use crate::visual::{CellVisual, GameVisual};

pub const GLYPH_PIECE: &str = "\u{2B22}";  // ⬢ filled pointy-top
pub const GLYPH_EMPTY: &str = "\u{2B21}";  // ⬡ outlined pointy-top

impl GameVisual for CCBoard {
    fn cell_visual(content: CCContent, coord: &HexCoord, board_size: u8) -> CellVisual {
        match content {
            CCContent::Player1 => CellVisual { glyph: GLYPH_PIECE, piece: Some(1), region: 0 },
            CCContent::Player2 => CellVisual { glyph: GLYPH_PIECE, piece: Some(2), region: 0 },
            CCContent::Empty => CellVisual {
                glyph: GLYPH_EMPTY,
                piece: None,
                region: home_region(coord.x, coord.y, board_size),
            },
        }
    }
}

/// Returns 1 if (x,y) is in P1's home, 2 if P2's home, 0 otherwise.
fn home_region(x: u8, y: u8, board_size: u8) -> u8 {
    let home_size = (board_size - 1) / 2;
    if x + y < home_size {
        return 1;
    }
    if x + y >= home_size + board_size && x < board_size && y < board_size {
        return 2;
    }
    0
}
