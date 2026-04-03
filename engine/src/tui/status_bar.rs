use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use crate::tui::theme;

pub struct ToggleState {
    pub show_eval: bool,
    pub show_policy: bool,
    pub show_pv: bool,
    /// Whether sampling is active. When false → argmax. When true → sample at `sampling_temperature`.
    pub sampling: bool,
    /// The temperature used when sampling is on. Adjusted via Shift+S modal.
    pub sampling_temperature: f32,
}

impl Default for ToggleState {
    fn default() -> Self {
        Self {
            show_eval: true,
            show_policy: false,
            show_pv: false,
            sampling: false,
            sampling_temperature: 0.5,
        }
    }
}

impl ToggleState {
    /// Returns None for argmax, Some(t) for sampling.
    pub fn temperature(&self) -> Option<f32> {
        if self.sampling { Some(self.sampling_temperature) } else { None }
    }

    /// Toggle between argmax and sampling.
    pub fn toggle_sampling(&mut self) {
        self.sampling = !self.sampling;
    }

    /// Set the sampling temperature (from the modal).
    pub fn set_temperature(&mut self, temp: Option<f32>) {
        match temp {
            None => self.sampling = false,
            Some(t) => {
                self.sampling = true;
                self.sampling_temperature = t;
            }
        }
    }
}

pub struct StatusBarWidget<'a> {
    pub toggles: &'a ToggleState,
    pub game_over: bool,
    pub current_player: i8,
}

impl Widget for StatusBarWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mut spans = vec![];

        fn toggle_span(label: &str, key: &str, on: bool) -> Vec<Span<'static>> {
            let state_str = if on { "ON " } else { "OFF" };
            let state_color = if on { theme::TOGGLE_ON } else { theme::TOGGLE_OFF };
            vec![
                Span::styled(format!(" [{key}]"), Style::default().fg(theme::UI)),
                Span::styled(format!("{label}:"), Style::default().fg(theme::UI)),
                Span::styled(state_str, Style::default().fg(state_color)),
            ]
        }

        spans.push(Span::styled(" [Q]", Style::default().fg(theme::UI)));
        spans.push(Span::styled("uit", Style::default().fg(theme::UI_DIM)));

        spans.extend(toggle_span("val", "E", self.toggles.show_eval));
        spans.extend(toggle_span("olicy", "P", self.toggles.show_policy));
        spans.extend(toggle_span("pv", "V", self.toggles.show_pv));

        // Temperature: [S] toggles argmax/sample, show current temperature value
        let (temp_label, temp_color) = if self.toggles.sampling {
            (format!("T={:.2}", self.toggles.sampling_temperature), theme::TOGGLE_ON)
        } else {
            ("argmax".to_string(), theme::TOGGLE_OFF)
        };
        spans.push(Span::styled(" [S]", Style::default().fg(theme::UI)));
        spans.push(Span::styled(temp_label, Style::default().fg(temp_color)));

        spans.push(Span::styled("  [←→]", Style::default().fg(theme::UI)));
        spans.push(Span::styled("History", Style::default().fg(theme::UI_DIM)));

        spans.push(Span::styled("  [N]", Style::default().fg(theme::UI)));
        spans.push(Span::styled("ew", Style::default().fg(theme::UI_DIM)));

        // Current player / game over indicator
        spans.push(Span::raw("  "));
        if self.game_over {
            spans.push(Span::styled("Game Over", Style::default().fg(theme::TOGGLE_ON)));
        } else {
            let (label, color) = match self.current_player {
                1 => ("P1", theme::P1),
                _ => ("P2", theme::P2),
            };
            spans.push(Span::styled(
                format!("{} {}", theme::GLYPH_PIECE, label),
                Style::default().fg(color),
            ));
        }

        let line = Line::from(spans);
        let para = Paragraph::new(line);
        Widget::render(para, area, buf);
    }
}
