use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::layout::Rect;

use crate::cc::HexCoord;

#[derive(Debug)]
pub enum AppEvent {
    // Mouse
    CellClicked(HexCoord),

    // Navigation
    Quit,
    HistoryBack,
    HistoryForward,
    HistoryStart,
    HistoryEnd,

    // Toggles
    ToggleEval,
    TogglePolicy,
    TogglePV,
    ToggleSampling,
    OpenTemperatureModal,

    // Modal interaction
    ModalLeft,
    ModalRight,
    ModalConfirm,
    ModalCancel,
    ModalClick(u16, u16), // raw screen (col, row) for slider click

    // Mouse drag
    MouseDown(u16, u16),   // raw (col, row)
    MouseDrag(u16, u16),   // raw (col, row)
    MouseUp,

    // Game control
    PauseResume,
    NewGame,

    // No-op
    Tick,
    Resize(u16, u16),
}

/// Poll for an input event, returning Tick if nothing happened within `timeout`.
pub fn poll_event(timeout: Duration, board_area: Rect, board_size: u8) -> AppEvent {
    if !event::poll(timeout).unwrap_or(false) {
        return AppEvent::Tick;
    }

    match event::read() {
        Ok(Event::Key(key)) => translate_key(key),
        Ok(Event::Mouse(mouse)) => translate_mouse(mouse, board_area, board_size),
        Ok(Event::Resize(w, h)) => AppEvent::Resize(w, h),
        _ => AppEvent::Tick,
    }
}

/// Poll for events when a modal dialog is open. Only modal-relevant keys/clicks are returned.
pub fn poll_modal_event(timeout: Duration) -> AppEvent {
    if !event::poll(timeout).unwrap_or(false) {
        return AppEvent::Tick;
    }

    match event::read() {
        Ok(Event::Key(key)) => match key.code {
            KeyCode::Left => AppEvent::ModalLeft,
            KeyCode::Right => AppEvent::ModalRight,
            KeyCode::Enter => AppEvent::ModalConfirm,
            KeyCode::Esc => AppEvent::ModalCancel,
            KeyCode::Char('q') => AppEvent::ModalCancel,
            _ => AppEvent::Tick,
        },
        Ok(Event::Mouse(mouse)) => {
            if let MouseEventKind::Down(MouseButton::Left) = mouse.kind {
                AppEvent::ModalClick(mouse.column, mouse.row)
            } else {
                AppEvent::Tick
            }
        }
        Ok(Event::Resize(w, h)) => AppEvent::Resize(w, h),
        _ => AppEvent::Tick,
    }
}

fn translate_key(key: KeyEvent) -> AppEvent {
    match key.code {
        KeyCode::Char('q') => AppEvent::Quit,
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => AppEvent::Quit,
        KeyCode::Char('e') => AppEvent::ToggleEval,
        KeyCode::Char('p') => AppEvent::TogglePolicy,
        KeyCode::Char('v') => AppEvent::TogglePV,
        KeyCode::Char('s') => AppEvent::ToggleSampling,
        KeyCode::Char('S') => AppEvent::OpenTemperatureModal,
        KeyCode::Char(' ') => AppEvent::PauseResume,
        KeyCode::Char('n') => AppEvent::NewGame,
        KeyCode::Left => AppEvent::HistoryBack,
        KeyCode::Right => AppEvent::HistoryForward,
        KeyCode::Home => AppEvent::HistoryStart,
        KeyCode::End => AppEvent::HistoryEnd,
        _ => AppEvent::Tick,
    }
}

fn translate_mouse(mouse: MouseEvent, board_area: Rect, board_size: u8) -> AppEvent {
    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            // Try hex cell first, fall back to raw mouse-down for drag detection
            if let Some(coord) = screen_to_hex(mouse.column, mouse.row, board_area, board_size) {
                AppEvent::CellClicked(coord)
            } else {
                AppEvent::MouseDown(mouse.column, mouse.row)
            }
        }
        MouseEventKind::Drag(MouseButton::Left) => AppEvent::MouseDrag(mouse.column, mouse.row),
        MouseEventKind::Up(MouseButton::Left) => AppEvent::MouseUp,
        _ => AppEvent::Tick,
    }
}

/// Map terminal (column, row) to hex board coordinate.
///
/// The board renders with row x indented by x spaces, each cell taking 2 columns (glyph + space).
pub fn screen_to_hex(col: u16, row: u16, board_area: Rect, board_size: u8) -> Option<HexCoord> {
    let rel_row = row.checked_sub(board_area.y)?;
    let x = rel_row as u8;
    if x >= board_size {
        return None;
    }

    let indent = x as u16;
    let rel_col = col.checked_sub(board_area.x + indent)?;
    let y = (rel_col / 2) as u8;
    if y >= board_size {
        return None;
    }

    Some(HexCoord::new(x, y, board_size))
}

/// Map hex coordinate to terminal (column, row) within the board area.
pub fn hex_to_screen(x: u8, y: u8, board_area: Rect) -> (u16, u16) {
    let col = board_area.x + (x as u16) + (y as u16) * 2;
    let row = board_area.y + x as u16;
    (col, row)
}
