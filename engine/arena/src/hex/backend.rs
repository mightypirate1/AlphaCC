use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

use alpha_cc_core::cc::HexCoord;
use super::layers::HexStyle;
use super::layout::HexLayout;

/// Trait for rendering backends. Each backend takes the same shared data
/// (layout + resolved styles) and writes to a ratatui Buffer in its own way.
pub trait RenderBackend {
    /// Render the hex board into the given terminal area.
    fn render(
        &self,
        layout: &HexLayout,
        styles: &[(u8, u8, HexStyle)],
        area: Rect,
        buf: &mut Buffer,
    );

    /// Map a terminal screen position to a hex coordinate.
    /// Each backend may have different coordinate mapping due to different scaling.
    fn screen_to_hex(
        &self,
        layout: &HexLayout,
        col: u16,
        row: u16,
        area: Rect,
    ) -> Option<HexCoord>;
}
