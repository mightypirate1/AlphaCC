use crate::hexcoordinate::HexCoordinate;
use crate::board::Board;

#[derive(Clone, Copy, Debug)]
pub enum Move {
    PlaceMove {coord: HexCoordinate},
    WalkMove {to: HexCoordinate, from: HexCoordinate},
    JumpMove {to: HexCoordinate, from: HexCoordinate},
}

impl Move {
    pub fn apply(self, mut board: Board) -> Board{
        match self {
            Move::WalkMove{to, from} => board = self.move_stone(board, to, from),
            Move::JumpMove{to, from} => board = self.move_stone(board, to, from),
            Move::PlaceMove{coord}   => board = self.place_at(board, coord),
        }
        return board;
    }

    pub fn is_legal(&self, board: &Board) -> bool {
        match self {
            Move::WalkMove{to, from} => {
                return from.get_all_neighbours(1).contains(&to)
                    && board.coord_is_empty(&to)
                    && board.coord_is_occupied_by_current_player(&from)
            },
            Move::JumpMove{to, from} => {
                for direction in from.get_all_directions(){
                    if *to == from.get_neighbor(direction, 2)
                    && !board.coord_is_empty(&from.get_neighbor(direction, 1)) {
                        return true;
                    }
                }
                return false;
            },
            Move::PlaceMove{coord}   => {return board.coord_is_empty(&coord)},
        };
    }
    fn place_at(self, mut board: Board, coord: HexCoordinate) -> Board {
        if !board.place_moves_are_allowed(){
            println!("This board does not allow PlaceMoves!");
            return board;
        }
        board.place(coord);
        return board;
    }
    fn move_stone(self, mut board: Board, to: HexCoordinate, from: HexCoordinate) -> Board {
        board.clear(from);
        board.place(to);
        return board;
    }
}
