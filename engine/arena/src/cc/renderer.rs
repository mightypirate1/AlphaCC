#![allow(dead_code)]
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;

use alpha_cc_core::cc::HexCoord;
use alpha_cc_core::board::Coord;

use crate::renderer::GameRenderer;
use crate::visual::StyledCell;

use super::visual::GLYPH_EMPTY;

// ── Hex layout geometry ──

/// Hex grid geometry in floating-point "world" coordinates.
/// Uses pointy-top hexagons with offset-row layout.
pub struct HexLayout {
    board_size: u8,
    pub hex_radius: f32,
    pub origin: (f32, f32),
}

impl HexLayout {
    fn col_spacing(&self) -> f32 {
        self.hex_radius * 3.0_f32.sqrt()
    }

    fn row_spacing(&self) -> f32 {
        self.hex_radius * 1.5
    }

    fn half_col(&self) -> f32 {
        self.col_spacing() / 2.0
    }

    pub fn grid_bounds(&self) -> (f32, f32) {
        let s = self.board_size as f32;
        let max_indent = (self.board_size - 1) as f32 * self.half_col();
        let grid_w = max_indent + self.col_spacing() * (s - 1.0) + self.hex_radius * 3.0_f32.sqrt();
        let grid_h = self.row_spacing() * (s - 1.0) + self.hex_radius * 2.0;
        (grid_w, grid_h)
    }

    pub fn fit(board_size: u8, world_width: f32, world_height: f32) -> Self {
        let s = board_size as f32;
        let sqrt3 = 3.0_f32.sqrt();
        let width_factor = sqrt3 * (3.0 * s - 1.0) / 2.0;
        let height_factor = 1.5 * s + 0.5;
        let r_from_w = world_width / width_factor;
        let r_from_h = world_height / height_factor;
        let hex_radius = r_from_w.min(r_from_h).max(2.0);

        let layout = HexLayout { board_size, hex_radius, origin: (0.0, 0.0) };
        let (grid_w, grid_h) = layout.grid_bounds();
        let ox = (world_width - grid_w) / 2.0;
        let oy = (world_height - grid_h) / 2.0;

        HexLayout {
            board_size,
            hex_radius,
            origin: (ox.max(0.0), oy.max(0.0)),
        }
    }

    pub fn hex_center(&self, x: u8, y: u8) -> (f32, f32) {
        let cx = self.origin.0
            + (x as f32) * self.half_col()
            + (y as f32) * self.col_spacing()
            + self.col_spacing() / 2.0;
        let cy = self.origin.1
            + (x as f32) * self.row_spacing()
            + self.hex_radius;
        (cx, cy)
    }

    pub fn hex_vertices_at(&self, cx: f32, cy: f32) -> [(f32, f32); 6] {
        let r = self.hex_radius * crate::theme::HEX_DRAW_SCALE;
        let sqrt3_2 = 3.0_f32.sqrt() / 2.0;
        [
            (cx, cy - r),
            (cx + r * sqrt3_2, cy - r * 0.5),
            (cx + r * sqrt3_2, cy + r * 0.5),
            (cx, cy + r),
            (cx - r * sqrt3_2, cy + r * 0.5),
            (cx - r * sqrt3_2, cy - r * 0.5),
        ]
    }

    pub fn hex_vertices(&self, x: u8, y: u8) -> [(f32, f32); 6] {
        let (cx, cy) = self.hex_center(x, y);
        self.hex_vertices_at(cx, cy)
    }

    pub fn world_to_hex(&self, wx: f32, wy: f32) -> Option<HexCoord> {
        let s = self.board_size;
        let mut best: Option<(HexCoord, f32)> = None;
        let max_dist_sq = self.hex_radius * self.hex_radius;

        for x in 0..s {
            for y in 0..s {
                let (cx, cy) = self.hex_center(x, y);
                let dx = wx - cx;
                let dy = wy - cy;
                let dist_sq = dx * dx + dy * dy;
                if dist_sq > max_dist_sq {
                    continue;
                }
                match &best {
                    None => best = Some((HexCoord::new(x, y, s), dist_sq)),
                    Some((_, d)) if dist_sq < *d => best = Some((HexCoord::new(x, y, s), dist_sq)),
                    _ => {}
                }
            }
        }
        best.map(|(c, _)| c)
    }

    pub fn board_size(&self) -> u8 {
        self.board_size
    }
}

// ── Hex renderer ──

pub struct HexRenderer {
    board_size: u8,
    layout: HexLayout,
}

impl HexRenderer {
    pub fn new(board_size: u8) -> Self {
        Self {
            board_size,
            layout: HexLayout::fit(board_size, 80.0, 40.0), // dummy initial size
        }
    }

    pub fn layout(&self) -> &HexLayout {
        &self.layout
    }
}

impl GameRenderer for HexRenderer {
    type Coord = HexCoord;

    fn screen_to_coord(&self, col: u16, row: u16, area: Rect) -> Option<HexCoord> {
        let s = self.board_size;
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

    fn render(
        &self,
        styled_cells: &[StyledCell<HexCoord>],
        area: Rect,
        buf: &mut Buffer,
    ) {
        let s = self.board_size;
        let grid_w = (s as u16 - 1) + s as u16 * 2;
        let grid_h = s as u16;
        let ox = area.x + area.width.saturating_sub(grid_w) / 2;
        let oy = area.y + area.height.saturating_sub(grid_h) / 2;

        for cell in styled_cells {
            let (x, y) = cell.coord.xy();
            let col = ox + (x as u16) + (y as u16) * 2;
            let row = oy + x as u16;
            if col < area.x + area.width && row < area.y + area.height {
                let glyph = cell.style.glyph;
                let glyph = if glyph.is_empty() { GLYPH_EMPTY } else { glyph };
                buf.set_string(col, row, glyph, Style::default().fg(cell.style.fill));
            }
        }
    }

    fn fit(&mut self, board_size: u8, area: Rect) {
        self.board_size = board_size;
        self.layout = HexLayout::fit(board_size, area.width as f32, area.height as f32);
    }
}
