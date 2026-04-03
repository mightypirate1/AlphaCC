use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Widget};

use crate::tui::game::MoveRecord;
use crate::tui::theme;

// ── Move List ──

pub struct MoveListWidget<'a> {
    pub moves: &'a [MoveRecord],
    pub view_index: usize,
    pub total_states: usize,
}

impl Widget for MoveListWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Moves ")
            .borders(Borders::ALL)
            .border_set(theme::BORDER)
            .border_style(Style::default().fg(theme::UI));

        let items: Vec<ListItem> = self.moves.iter().enumerate().map(|(i, rec)| {
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
            ListItem::new(text).style(Style::default().fg(color))
        }).collect();

        let list = List::new(items).block(block);
        Widget::render(list, area, buf);
    }
}

// ── Eval Bar (vertical) ──

pub struct EvalBarWidget {
    pub value: f32,
    pub rollouts: usize,
    pub visible: bool,
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
        let p1_rows = (fill_ratio * bar_height as f32) as u16;

        for row_offset in 0..bar_height {
            let row = inner.y + row_offset;
            let from_top = row_offset;

            let (ch, style) = if from_top < p1_rows {
                ("█", Style::default().fg(theme::P1))
            } else {
                ("█", Style::default().fg(theme::P2))
            };

            for col in inner.x..inner.x + inner.width {
                buf.set_string(col, row, ch, style);
            }
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
