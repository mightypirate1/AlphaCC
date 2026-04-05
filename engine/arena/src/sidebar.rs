use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget};

use crate::game::MoveRecord;
use crate::theme;

// ── Move List ──

pub struct MoveListWidget<'a> {
    pub moves: &'a [MoveRecord],
    pub view_index: usize,
    pub scroll_offset: usize,
}

impl Widget for MoveListWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Moves ")
            .borders(Borders::ALL)
            .border_set(theme::BORDER)
            .border_style(Style::default().fg(theme::UI));
        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height == 0 || inner.width == 0 { return; }

        let visible_rows = inner.height as usize;

        let items: Vec<(Style, String)> = self.moves.iter().enumerate().map(|(i, rec)| {
            let player = if i % 2 == 0 { 1 } else { 2 };
            let color = if player == 1 { theme::P1 } else { theme::P2 };
            let highlight = if i + 1 == self.view_index { ">" } else { " " };
            let text = format!(
                "{}{:>3}. ({},{})→({},{})",
                highlight,
                i + 1,
                rec.mv.from_coord.x, rec.mv.from_coord.y,
                rec.mv.to_coord.x, rec.mv.to_coord.y,
            );
            (Style::default().fg(color), text)
        }).collect();

        for (i, (style, text)) in items.iter().enumerate().skip(self.scroll_offset).take(visible_rows) {
            let row = inner.y + (i - self.scroll_offset) as u16;
            buf.set_string(inner.x, row, text, *style);
        }
    }
}

// ── Eval Bar (vertical) ──

pub struct EvalBarWidget {
    pub value: f32,
    #[allow(dead_code)]
    pub rollouts: usize,
    pub visible: bool,
    pub braille: bool,
}

impl Widget for EvalBarWidget {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if !self.visible || area.height < 4 || area.width < 3 {
            return;
        }

        let block = Block::default()
            .borders(Borders::ALL)
            .border_set(theme::BORDER)
            .border_style(Style::default().fg(theme::UI));
        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height == 0 || inner.width == 0 {
            return;
        }

        let bar_height = inner.height.saturating_sub(1); // reserve 1 row for value label
        if bar_height == 0 {
            return;
        }

        // value in [-1, 1] → fill_ratio in [0, 1] where 1.0 = full P1 advantage (top)
        let fill_ratio = ((self.value + 1.0) / 2.0).clamp(0.0, 1.0);
        let zero_row = inner.y + bar_height / 2;

        if self.braille {
            self.render_braille(inner, bar_height, fill_ratio, buf);
        } else {
            self.render_block(inner, bar_height, fill_ratio, buf);
        }

        // Zero marker: small ticks on the left and right borders
        if zero_row >= area.y && zero_row < area.y + area.height {
            buf.set_string(area.x, zero_row, "├", Style::default().fg(theme::UI));
            let right_col = area.x + area.width - 1;
            buf.set_string(right_col, zero_row, "┤", Style::default().fg(theme::UI));
        }

        // Value label at bottom
        let label = format!("{:+.2}", self.value);
        let label_row = inner.y + bar_height;
        let label_col = inner.x + inner.width.saturating_sub(label.len() as u16) / 2;
        buf.set_string(label_col, label_row, &label, Style::default().fg(theme::UI));
    }
}

impl EvalBarWidget {
    /// Block mode: half-block sub-resolution on the boundary row.
    /// Each cell is split left/right via ▌, giving inner.width * 2 horizontal positions per row.
    fn render_block(&self, inner: Rect, bar_height: u16, fill_ratio: f32, buf: &mut Buffer) {
        let p1_rows_exact = fill_ratio * bar_height as f32;
        let p1_full_rows = p1_rows_exact as u16;
        let sub_fill = p1_rows_exact - p1_full_rows as f32;

        for row_offset in 0..bar_height {
            let row = inner.y + row_offset;

            if row_offset < p1_full_rows {
                for col in inner.x..inner.x + inner.width {
                    buf.set_string(col, row, "█", Style::default().fg(theme::P1));
                }
            } else if row_offset == p1_full_rows {
                let total_dots = inner.width as f32 * 2.0;
                let filled_dots = (sub_fill * total_dots).round() as usize;

                for col_idx in 0..inner.width {
                    let col = inner.x + col_idx;
                    let dot_start = col_idx as usize * 2;
                    let left_on = dot_start < filled_dots;
                    let right_on = dot_start + 1 < filled_dots;

                    if left_on && right_on {
                        buf.set_string(col, row, "█", Style::default().fg(theme::P1));
                    } else if left_on {
                        buf.set_string(col, row, "▌", Style::default().fg(theme::P1).bg(theme::P2));
                    } else {
                        buf.set_string(col, row, "█", Style::default().fg(theme::P2));
                    }
                }
            } else {
                for col in inner.x..inner.x + inner.width {
                    buf.set_string(col, row, "█", Style::default().fg(theme::P2));
                }
            }
        }
    }

    /// Braille mode: entire bar rendered with braille characters.
    /// Each cell = 2 dots wide × 4 dots tall, giving 4x vertical and 2x horizontal resolution.
    /// P1 dots are drawn in P1 color on P2 background. The boundary between P1/P2
    /// is smooth both vertically (4 sub-rows per cell) and horizontally (2 sub-cols).
    fn render_braille(&self, inner: Rect, bar_height: u16, fill_ratio: f32, buf: &mut Buffer) {
        // Total dot resolution
        let dot_height = bar_height as f32 * 4.0;
        let dot_width = inner.width as f32 * 2.0;

        // How many dot-rows from the top should be P1
        let p1_dot_rows = (fill_ratio * dot_height).round() as usize;

        // Within the boundary dot-row, how many dot-columns from the left should be P1
        // This gives the horizontal sub-resolution at the transition
        let p1_rows_exact = fill_ratio * dot_height;
        let sub_fill = p1_rows_exact - (p1_rows_exact as usize) as f32;
        let boundary_dots_x = (sub_fill * dot_width).round() as usize;

        // Braille dot positions (row, col) → bit:
        //   (0, 0)=0x01  (0, 1)=0x08
        //   (1, 0)=0x02  (1, 1)=0x10
        //   (2, 0)=0x04  (2, 1)=0x20
        //   (3, 0)=0x40  (3, 1)=0x80
        const DOT_BITS: [[u8; 2]; 4] = [
            [0x01, 0x08],
            [0x02, 0x10],
            [0x04, 0x20],
            [0x40, 0x80],
        ];

        for cell_row in 0..bar_height {
            let term_row = inner.y + cell_row;
            let base_dot_row = cell_row as usize * 4;

            for col_idx in 0..inner.width {
                let col = inner.x + col_idx;
                let base_dot_col = col_idx as usize * 2;
                let mut bits: u8 = 0;

                for (dot_row, row_bits) in DOT_BITS.iter().enumerate() {
                    let dy = base_dot_row + dot_row;
                    for (dot_col, &bit) in row_bits.iter().enumerate() {
                        let dx = base_dot_col + dot_col;
                        let is_p1 = if dy < p1_dot_rows {
                            true
                        } else if dy == p1_dot_rows {
                            dx < boundary_dots_x
                        } else {
                            false
                        };

                        if is_p1 {
                            bits |= bit;
                        }
                    }
                }

                // bits = P1 dots. Render P1 portion as braille fg, no bg.
                // Fully P1 or fully P2 cells use full braille (⣿), mixed cells
                // show partial dots — same style as the policy bars.
                let all_on: u8 = 0xFF;
                if bits == all_on {
                    let ch = char::from_u32(0x2800 + all_on as u32).unwrap_or('⣿');
                    let mut s = [0u8; 4];
                    buf.set_string(col, term_row, ch.encode_utf8(&mut s),
                        Style::default().fg(theme::P1));
                } else if bits == 0 {
                    let ch = char::from_u32(0x2800 + all_on as u32).unwrap_or('⣿');
                    let mut s = [0u8; 4];
                    buf.set_string(col, term_row, ch.encode_utf8(&mut s),
                        Style::default().fg(theme::P2));
                } else {
                    // Mixed: P1 dots in P1 color, gaps are black (no bg)
                    let ch = char::from_u32(0x2800 + bits as u32).unwrap_or(' ');
                    let mut s = [0u8; 4];
                    buf.set_string(col, term_row, ch.encode_utf8(&mut s),
                        Style::default().fg(theme::P1));
                }
            }
        }
    }
}

// ── Principal Variation ──

pub struct PVWidget<'a> {
    pub pv_moves: &'a [String],
    pub visible: bool,
}

impl Widget for PVWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if !self.visible {
            return;
        }

        let block = Block::default()
            .title(" PV ")
            .borders(Borders::ALL)
            .border_set(theme::BORDER)
            .border_style(Style::default().fg(theme::UI));

        let text: Vec<Line> = self.pv_moves.iter().enumerate().map(|(i, mv)| {
            let color = if i % 2 == 0 { theme::P1 } else { theme::P2 };
            Line::from(Span::styled(format!("  {}. {}", i + 1, mv), Style::default().fg(color)))
        }).collect();

        let para = Paragraph::new(text).block(block);
        Widget::render(para, area, buf);
    }
}
