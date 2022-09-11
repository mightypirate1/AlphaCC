mod hexcoordinate;
use std::vec::Vec;
use hexcoordinate::HexCoordinate;
#[derive(Debug)]
enum Move {
    WalkMove {to: HexCoordinate, from: HexCoordinate},
    JumpMove {to: HexCoordinate, from: HexCoordinate},
}

impl Move {
    pub fn apply(self){
        match self {
            Move::WalkMove{to, from} => println!("I WALK to {to:?} from {from:?}"),
            Move::JumpMove{to, from} => println!("I JUMP to {to:?} from {from:?}"),
            _ => panic!("How did you even??"),
        }
    }
}

impl Move {
    pub fn another_function(self) {
        println!("det gick!");
    }
}


fn main() {
    let x: HexCoordinate = HexCoordinate::create(0,0);
    let y: HexCoordinate = HexCoordinate::create(2,1);
    let my_move_1: Move = Move::WalkMove{to: x, from: y};
    let my_move_2: Move = Move::JumpMove{to: x, from: y};
    let move_vec = vec![my_move_1, my_move_2];
    for my_move in move_vec {
        println!("{my_move:?}");
        my_move.apply();
    }
}
