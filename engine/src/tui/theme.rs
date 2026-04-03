use ratatui::style::Color;
use ratatui::symbols::border;

// Player colors
pub const P1: Color = Color::Rgb(255, 170, 30);   // vivid orange
pub const P2: Color = Color::Rgb(255, 105, 180);  // pink

// UI chrome
pub const UI: Color = Color::Rgb(150, 100, 200);      // purple — borders, chrome, labels
pub const UI_DIM: Color = Color::Rgb(90, 60, 130);     // dim purple — inactive toggles, subtle elements
pub const EMPTY_CELL: Color = Color::Rgb(100, 70, 150); // dark purple — empty hex cells
pub const BG: Color = Color::Rgb(20, 14, 30);          // near-black with purple tint

// Interaction
pub const SELECTED: Color = Color::Rgb(255, 255, 200); // bright warm white — selected piece
pub const LEGAL_MOVE: Color = Color::Rgb(0, 220, 220); // cyan — legal destinations

/// Brighten a player color for last-move highlight.
pub fn last_move_color(player: i8) -> Color {
    brighten(if player == 1 { P1 } else { P2 }, 0.5)
}

// Status bar
pub const TOGGLE_ON: Color = Color::Rgb(100, 220, 130);  // green
pub const TOGGLE_OFF: Color = UI_DIM;

// Glyphs
pub const GLYPH_PIECE: &str = "\u{2B22}";  // ⬢ filled pointy-top
pub const GLYPH_EMPTY: &str = "\u{2B21}";  // ⬡ outlined pointy-top

// Borders — thin rounded (╭─╮ / │ / ╰─╯)
pub const BORDER: border::Set = border::ROUNDED;

/// Brighten a color by lerping each channel toward 255.
pub fn brighten(color: Color, amount: f32) -> Color {
    let a = amount.clamp(0.0, 1.0);
    if let Color::Rgb(r, g, b) = color {
        let up = |c: u8| (c as f32 + (255.0 - c as f32) * a) as u8;
        Color::Rgb(up(r), up(g), up(b))
    } else {
        color
    }
}

/// Policy heatmap: lerp from dim to vivid in the given player color.
pub fn policy_color(base: Color, weight: f32) -> Color {
    let w = weight.clamp(0.0, 1.0);
    if let Color::Rgb(r, g, b) = base {
        let dim = 0.15_f32;
        let lerp = |c: u8| ((c as f32) * (dim + (1.0 - dim) * w)) as u8;
        Color::Rgb(lerp(r), lerp(g), lerp(b))
    } else {
        base
    }
}

/// Eval bar: color based on value sign.
pub fn eval_bar_color(value: f32) -> Color {
    if value >= 0.0 { P1 } else { P2 }
}
