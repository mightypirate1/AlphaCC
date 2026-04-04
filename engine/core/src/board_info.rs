use crate::dtypes;

pub struct BoardInfo {
    pub current_player: i8,
    pub winner: i8,
    pub size: dtypes::BoardSize,
    pub duration: dtypes::GameDuration,
    pub game_over: bool,
    pub reward: f32,
}
