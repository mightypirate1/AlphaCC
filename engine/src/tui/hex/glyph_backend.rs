use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;

use crate::cc::HexCoord;
use crate::tui::theme;
use super::backend::RenderBackend;
use super::layers::HexStyle;
use super::layout::HexLayout;

/// Original glyph-based renderer: one character per hex cell, arranged
/// in the classic offset-row pattern. Simple, reliable, works everywhere.
pub struct GlyphBackend;

impl RenderBackend for GlyphBackend {
    fn render(
        &self,
        layout: &HexLayout,
        styles: &[(u8, u8, HexStyle)],
        area: Rect,
        buf: &mut Buffer,
    ) {
        let s = layout.board_size();
        // Glyph layout: row x is indented by x, each cell is 2 chars wide (glyph + space).
        // Center within the area.
        let grid_w = (s as u16 - 1) + s as u16 * 2;
        let grid_h = s as u16;
        let ox = area.x + area.width.saturating_sub(grid_w) / 2;
        let oy = area.y + area.height.saturating_sub(grid_h) / 2;

        for &(x, y, ref style) in styles {
            let col = ox + (x as u16) + (y as u16) * 2;
            let row = oy + x as u16;
            if col < area.x + area.width && row < area.y + area.height {
                let glyph = style.glyph.unwrap_or(theme::GLYPH_EMPTY);
                // Use fill color as foreground for the glyph
                buf.set_string(col, row, glyph, Style::default().fg(style.fill));
            }
        }
    }

    fn screen_to_hex(
        &self,
        layout: &HexLayout,
        col: u16,
        row: u16,
        area: Rect,
    ) -> Option<HexCoord> {
        let s = layout.board_size();
        let grid_w = (s as u16 - 1) + s as u16 * 2;
        let grid_h = s as u16;
        let ox = area.x + area.width.saturating_sub(grid_w) / 2;
        let oy = area.y + area.height.saturating_sub(grid_h) / 2;

        let rel_row = row.checked_sub(oy)?;
        let x = rel_row as u8;
        if x >= s { return None; }

        let indent = x as u16;
        let rel_col = col.checked_sub(ox + indent)?;
        let y = (rel_col / 2) as u8;
        if y >= s { return None; }

        Some(HexCoord::new(x, y, s))
    }
}
