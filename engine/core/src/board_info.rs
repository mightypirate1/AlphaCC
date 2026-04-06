use crate::dtypes;

#[derive(Clone, Copy, Debug)]
pub struct WDL {
    pub win: f32,
    pub draw: f32,
    pub loss: f32,
}

impl WDL {
    pub fn win() -> Self { Self { win: 1.0, draw: 0.0, loss: 0.0 } }
    pub fn loss() -> Self { Self { win: 0.0, draw: 0.0, loss: 1.0 } }
    pub fn draw() -> Self { Self { win: 0.0, draw: 1.0, loss: 0.0 } }

    pub fn to_value(&self) -> f32 {
        self.win - self.loss
    }

    /// Flip perspective: my win is their loss and vice versa.
    pub fn flip(&self) -> Self {
        Self { win: self.loss, draw: self.draw, loss: self.win }
    }
}

pub struct BoardInfo {
    pub current_player: i8,
    pub winner: i8,
    pub size: dtypes::BoardSize,
    pub duration: dtypes::GameDuration,
    pub game_over: bool,
    pub wdl: WDL,
}
