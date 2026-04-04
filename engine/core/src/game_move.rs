use crate::hexcoord::HexCoord;

#[derive(Clone)]
pub struct Move {
    pub from_coord: HexCoord,
    pub to_coord: HexCoord,
    pub path: Vec<HexCoord>,
}
