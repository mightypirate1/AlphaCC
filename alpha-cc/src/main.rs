use std::vec::Vec;
mod board;
use board::Board;
use board::hexcoordinate::HexCoordinate;

fn dbg_print_direction(direction: usize, board: &Board){
    let centre: HexCoordinate = HexCoordinate::create(1, 1);
    let neighbor: HexCoordinate;
    let state: i32;
    neighbor = centre.get_neighbor(direction);
    state = board.get_boardstate_by_coord(&neighbor);
    println!("direction {} -> {},{} = {}", direction, neighbor.x, neighbor.y, state);
}

fn main() {
    let mut x = Board::create();
    x.place(HexCoordinate::create(0,0));
    x.print();
    x.place(HexCoordinate::create(0,1));
    x.print();
    let legal_moves: Vec<HexCoordinate> = x.get_legal_moves_for_player(1);
    for legal_move in legal_moves.iter(){
        println!("{},{}", legal_move.x, legal_move.y);
    }

    for direction in 0..6 {
        dbg_print_direction(direction, &x);
    }
}
