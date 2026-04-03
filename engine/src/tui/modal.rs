use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::widgets::{Block, Borders, Clear, Widget};

use crate::tui::theme;

/// A modal dialog for adjusting temperature with a visual slider.
pub struct TemperatureModal {
    /// Current temperature value (0.0 = argmax-like, up to max)
    pub value: f32,
    pub min: f32,
    pub max: f32,
    pub step: f32,
}

impl Default for TemperatureModal {
    fn default() -> Self {
        Self {
            value: 0.5,
            min: 0.0,
            max: 2.0,
            step: 0.05,
        }
    }
}

impl TemperatureModal {
    pub fn nudge_left(&mut self) {
        self.value = (self.value - self.step).max(self.min);
    }

    pub fn nudge_right(&mut self) {
        self.value = (self.value + self.step).min(self.max);
    }

    /// Map a mouse click column within the slider track to a value.
    pub fn click_at(&mut self, col: u16, track_start: u16, track_width: u16) {
        if track_width == 0 {
            return;
        }
        let rel = col.saturating_sub(track_start) as f32;
        let ratio = (rel / (track_width - 1) as f32).clamp(0.0, 1.0);
        self.value = self.min + ratio * (self.max - self.min);
        // Snap to nearest step
        self.value = (self.value / self.step).round() * self.step;
        self.value = self.value.clamp(self.min, self.max);
    }

    /// Return the result as Option<f32>: None if at 0.0 (argmax), Some(t) otherwise.
    pub fn as_temperature(&self) -> Option<f32> {
        if self.value < 0.001 { None } else { Some(self.value) }
    }

    /// Centered rect for the modal.
    pub fn area(screen: Rect) -> Rect {
        let width = 30.min(screen.width.saturating_sub(4));
        let height = 7.min(screen.height.saturating_sub(2));
        let x = screen.x + (screen.width.saturating_sub(width)) / 2;
        let y = screen.y + (screen.height.saturating_sub(height)) / 2;
        Rect { x, y, width, height }
    }

    /// The slider track rect within the modal inner area.
    pub fn track_rect(inner: Rect) -> (u16, u16, u16) {
        let pad = 2;
        let start = inner.x + pad;
        let width = inner.width.saturating_sub(pad * 2);
        let row = inner.y + 1;
        (start, width, row)
    }
}

pub struct TemperatureModalWidget<'a> {
    pub modal: &'a TemperatureModal,
}

impl Widget for TemperatureModalWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let modal_area = TemperatureModal::area(area);

        // Clear the area behind the modal
        Clear.render(modal_area, buf);

        let block = Block::default()
            .title(" Temperature ")
            .borders(Borders::ALL)
            .border_set(theme::BORDER)
            .border_style(Style::default().fg(theme::UI));
        let inner = block.inner(modal_area);
        block.render(modal_area, buf);

        if inner.height < 5 || inner.width < 10 {
            return;
        }

        let (track_start, track_width, track_row) = TemperatureModal::track_rect(inner);
        if track_width < 3 {
            return;
        }

        // Draw slider track
        let ratio = ((self.modal.value - self.modal.min) / (self.modal.max - self.modal.min))
            .clamp(0.0, 1.0);
        let knob_pos = track_start + (ratio * (track_width - 1) as f32) as u16;

        for col in track_start..track_start + track_width {
            let (ch, style) = if col == knob_pos {
                ("●", Style::default().fg(theme::LEGAL_MOVE))
            } else if col < knob_pos {
                ("━", Style::default().fg(theme::P1))
            } else {
                ("━", Style::default().fg(theme::UI_DIM))
            };
            buf.set_string(col, track_row, ch, style);
        }

        // Arrows at ends
        buf.set_string(track_start - 1, track_row, "◄", Style::default().fg(theme::UI));
        buf.set_string(track_start + track_width, track_row, "►", Style::default().fg(theme::UI));

        // Value label centered below the track
        let label = if self.modal.value < 0.001 {
            "argmax".to_string()
        } else {
            format!("{:.2}", self.modal.value)
        };
        let label_col = inner.x + (inner.width.saturating_sub(label.len() as u16)) / 2;
        buf.set_string(label_col, track_row + 1, &label, Style::default().fg(theme::SELECTED));

        // Hints
        let hint1 = "[←/→] adjust  [Click] set";
        let hint2 = "[Enter] confirm  [Esc] cancel";
        let hint_row1 = track_row + 3;
        let hint_row2 = track_row + 4;
        let h1_col = inner.x + inner.width.saturating_sub(hint1.len() as u16) / 2;
        let h2_col = inner.x + inner.width.saturating_sub(hint2.len() as u16) / 2;
        if hint_row1 < inner.y + inner.height {
            buf.set_string(h1_col, hint_row1, hint1, Style::default().fg(theme::UI_DIM));
        }
        if hint_row2 < inner.y + inner.height {
            buf.set_string(h2_col, hint_row2, hint2, Style::default().fg(theme::UI_DIM));
        }
    }
}
