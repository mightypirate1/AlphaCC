mod board;
mod hexcoordinate;
mod moves;
use std::vec::Vec;
use board::Board;
use hexcoordinate::HexCoordinate;
use moves::Move;

// fn dbg_print_direction(direction: usize, board: &Board){
//     let centre: HexCoordinate = HexCoordinate::create(1, 1);
//     let neighbor: HexCoordinate;
//     let state: i32;
//     neighbor = centre.get_neighbor(direction, 1);
//     state = board.get_boardstate_by_coord(&neighbor);
//     println!("direction {} -> {},{} = {}", direction, neighbor.x, neighbor.y, state);
// }

fn extra_place_moves() -> Vec<Move> {
    return vec![
        Move::PlaceMove{coord: HexCoordinate::create(3,2)},
        Move::PlaceMove{coord: HexCoordinate::create(3,3)},
        Move::PlaceMove{coord: HexCoordinate::create(4,3)},
        Move::PlaceMove{coord: HexCoordinate::create(5,2)},
        Move::PlaceMove{coord: HexCoordinate::create(2,6)},
        Move::PlaceMove{coord: HexCoordinate::create(7,5)},
        Move::PlaceMove{coord: HexCoordinate::create(8,4)},
        Move::PlaceMove{coord: HexCoordinate::create(5,1)},
        Move::PlaceMove{coord: HexCoordinate::create(5,6)},
        Move::PlaceMove{coord: HexCoordinate::create(1,7)},
        Move::PlaceMove{coord: HexCoordinate::create(5,5)},
    ];
}

fn main() {
    // Set up board!
    let mut board = Board::create(9);

    board.allow_place_moves = true;
    for init_move in extra_place_moves() {
        board = init_move.apply(board);
    }
    board.allow_place_moves = false;

    println!("-----");
    let mut is_legal: bool;
    for legal_move in board.get_all_legal_moves(1){
        is_legal = legal_move.is_legal(&board);
        println!("{legal_move:?} -> {is_legal}")
    }
    board.print();
    for legal_move in board.get_all_legal_moves(1){
        match legal_move {
            Move::JumpMove{..} => {
                println!("APPLYING: {legal_move:?}");
                board = legal_move.apply(board);
                break;
            },
            _ => {},
        }
    }

    for _ in 0..1000 {
         board.get_all_legal_moves(1);
    }

    board.print();
}
