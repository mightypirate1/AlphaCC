use crate::board::Coord;

#[derive(Clone)]
pub struct Move<C: Coord> {
    pub from_coord: C,
    pub to_coord: C,
}
