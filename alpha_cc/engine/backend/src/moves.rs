use crate::hexcoordinate::HexCoordinate;
use crate::board::Board;

#[derive(Clone, Copy, Debug)]
pub enum Move {
    Place {coord: HexCoordinate},
    Walk {to: HexCoordinate, from: HexCoordinate},
    Jump {to: HexCoordinate, from: HexCoordinate},
}

impl Move {
    pub fn apply(self, mut board: Board) -> Board{
        match self {
            Move::Walk{to, from} => board = self.move_stone(board, to, from),
            Move::Jump{to, from} => board = self.move_stone(board, to, from),
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

    fn move_stone(self, mut board: Board, to: HexCoordinate, from: HexCoordinate) -> Board {
        board.clear(from);
        board.place(to);
        board
    }
}
