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

    pub fn is_legal(&self, board: &Board) -> bool {
        match self {
            Move::Walk{to, from} => {
                from.get_all_neighbours(1).contains(to)
                    && board.coord_is_empty(to)
                    && board.coord_is_occupied_by_current_player(from)
            },
            Move::Jump{to, from} => {
                for direction in from.get_all_directions(){
                    if *to == from.get_neighbor(direction, 2)
                    && !board.coord_is_empty(&from.get_neighbor(direction, 1)) {
                        return true;
                    }
                }
                false
            },
            Move::Place{coord}   => {board.coord_is_empty(coord)},
        }
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
