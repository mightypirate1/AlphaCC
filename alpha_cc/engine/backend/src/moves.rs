use pyo3::prelude::*;
use crate::hexcoordinate::HexCoordinate;
use crate::board::Board;

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub enum Move {
    Place {coord: HexCoordinate},
    Walk {to_coord: HexCoordinate, from_coord: HexCoordinate},
    Jump {to_coord: HexCoordinate, from_coord: HexCoordinate},
}

impl Move {
    pub fn apply(self, mut board: Board) -> Board{
        match self {
            Move::Walk{to_coord, from_coord} => board = self.move_stone(board, to_coord, from_coord),
            Move::Jump{to_coord, from_coord} => board = self.move_stone(board, to_coord, from_coord),
            Move::Place{coord}   => board = self.place_at(board, coord),
        }
        board
    }

    fn place_at(self, mut board: Board, coord: HexCoordinate) -> Board {
        if !board.place_moves_are_allowed(){
            println!("This board does not allow Places!");
            return board;
        }
        board.place(coord);
        board
    }

    fn move_stone(self, mut board: Board, to_coord: HexCoordinate, from_coord: HexCoordinate) -> Board {
        board.clear(from_coord);
        board.place(to_coord);
        board
    }
}

