use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

use crate::visual::StyledCell;

/// Trait for game-specific spatial renderers.
///
/// Each game provides its own renderer that knows the grid geometry
/// (hex, square, etc.) and how to draw styled cells to the terminal.
pub trait GameRenderer {
    type Coord: Copy + Eq + std::hash::Hash + Send;

    /// Convert a screen position to a game coordinate, if the click hit a cell.
    fn screen_to_coord(&self, col: u16, row: u16, area: Rect) -> Option<Self::Coord>;

    /// Render styled cells to the terminal buffer.
    fn render(
        &self,
        styled_cells: &[StyledCell<Self::Coord>],
        area: Rect,
        buf: &mut Buffer,
    );

    /// Recalculate layout for a new terminal area size.
    fn fit(&mut self, board_size: u8, area: Rect);
}
