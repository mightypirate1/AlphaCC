use crate::cc::HexCoord;

/// Hex grid geometry in floating-point "world" coordinates.
/// Backends map these to their own coordinate systems (terminal cells, pixels, etc.).
///
/// Uses pointy-top hexagons with offset-row layout matching the board's render convention:
/// row x is indented by x half-widths.
pub struct HexLayout {
    board_size: u8,
    /// Hex outer radius (center to vertex) in world units.
    pub hex_radius: f32,
    /// Top-left origin offset in world units.
    pub origin: (f32, f32),
}

impl HexLayout {
    /// Horizontal distance between hex centers in the same row.
    fn col_spacing(&self) -> f32 {
        self.hex_radius * 3.0_f32.sqrt()
    }

    /// Vertical distance between hex row centers.
    fn row_spacing(&self) -> f32 {
        self.hex_radius * 1.5
    }

    /// Half the column spacing (the per-row indent).
    fn half_col(&self) -> f32 {
        self.col_spacing() / 2.0
    }

    /// Total bounding box of the hex grid in world units.
    pub fn grid_bounds(&self) -> (f32, f32) {
        let s = self.board_size as f32;
        let max_indent = (self.board_size - 1) as f32 * self.half_col();
        let grid_w = max_indent + self.col_spacing() * (s - 1.0) + self.hex_radius * 3.0_f32.sqrt();
        let grid_h = self.row_spacing() * (s - 1.0) + self.hex_radius * 2.0;
        (grid_w, grid_h)
    }

    /// Fit the hex grid into the given world-space dimensions, maximizing hex size.
    pub fn fit(board_size: u8, world_width: f32, world_height: f32) -> Self {
        let s = board_size as f32;
        let sqrt3 = 3.0_f32.sqrt();

        // The grid width as a function of radius r:
        //   max_indent + (s-1)*col_spacing + sqrt3*r
        //   = (s-1) * sqrt3*r/2 + (s-1)*sqrt3*r + sqrt3*r
        //   = sqrt3*r * ((s-1)/2 + (s-1) + 1)
        //   = sqrt3*r * ((3s-1)/2)
        let width_factor = sqrt3 * (3.0 * s - 1.0) / 2.0;

        // The grid height as a function of radius r:
        //   (s-1)*1.5*r + 2*r = r*(1.5s + 0.5)
        let height_factor = 1.5 * s + 0.5;

        let r_from_w = world_width / width_factor;
        let r_from_h = world_height / height_factor;
        let hex_radius = r_from_w.min(r_from_h).max(2.0);

        let layout = HexLayout {
            board_size,
            hex_radius,
            origin: (0.0, 0.0),
        };

        // Center the grid
        let (grid_w, grid_h) = layout.grid_bounds();
        let ox = (world_width - grid_w) / 2.0;
        let oy = (world_height - grid_h) / 2.0;

        HexLayout {
            board_size,
            hex_radius,
            origin: (ox.max(0.0), oy.max(0.0)),
        }
    }

    /// Center of a hex in world coordinates.
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

    /// The 6 vertices of a pointy-top hexagon centered at (cx, cy),
    /// scaled down by HEX_DRAW_SCALE to create gaps between hexes.
    pub fn hex_vertices_at(&self, cx: f32, cy: f32) -> [(f32, f32); 6] {
        let r = self.hex_radius * crate::tui::theme::HEX_DRAW_SCALE;
        let sqrt3_2 = 3.0_f32.sqrt() / 2.0;
        [
            (cx, cy - r),                       // top
            (cx + r * sqrt3_2, cy - r * 0.5),   // top-right
            (cx + r * sqrt3_2, cy + r * 0.5),   // bottom-right
            (cx, cy + r),                        // bottom
            (cx - r * sqrt3_2, cy + r * 0.5),   // bottom-left
            (cx - r * sqrt3_2, cy - r * 0.5),   // top-left
        ]
    }

    /// The 6 vertices for hex at grid position (x, y).
    pub fn hex_vertices(&self, x: u8, y: u8) -> [(f32, f32); 6] {
        let (cx, cy) = self.hex_center(x, y);
        self.hex_vertices_at(cx, cy)
    }

    /// Full-radius vertices for hit testing (not scaled by HEX_DRAW_SCALE).
    #[allow(dead_code)]
    fn hit_vertices_at(&self, cx: f32, cy: f32) -> [(f32, f32); 6] {
        let r = self.hex_radius; // full radius, no draw scale
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

    /// Find which hex contains the given world coordinate, if any.
    /// Uses nearest-center with a radius threshold for robust hit detection.
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

/// Point-in-convex-polygon test using cross products.
#[allow(dead_code)]
fn point_in_hex(px: f32, py: f32, verts: &[(f32, f32); 6]) -> bool {
    let n = verts.len();
    for i in 0..n {
        let (x1, y1) = verts[i];
        let (x2, y2) = verts[(i + 1) % n];
        let cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1);
        if cross > 0.0 {
            return false;
        }
    }
    true
}
