use std::vec::Vec;
use ndarray::Array2;
pub mod hexcoordinate;
use hexcoordinate::HexCoordinate;

pub struct Board{
    current_player: i32,
    n_players: i32,
    matrix: Array2::<i32>,
}

impl Board{
    pub fn create() -> Board {
        return Board {
            current_player: 1,
            n_players: 2,
            matrix: Array2::zeros((3,3)),
        };
    }

    pub fn print(&self) {
        println!("{:}", self.matrix);
        println!("Current player: {}", self.current_player);
    }

    pub fn place(&mut self, coord: HexCoordinate){
        if  self.coord_is_valid(&coord) && self.coord_is_empty(&coord) {
                println!("Player {} placing at {}, {}", self.current_player, coord.x, coord.y);
                self.set_board_state(coord, self.current_player);
                self.next_player();
        }
        else {
            panic!("Invalid play attempted: {} @ {}, {}", self.current_player, coord.x, coord.y);
        }
    }

    pub fn get_boardstate_by_coord(&self, coord: &HexCoordinate) -> i32 {
        return self.matrix[[coord.x, coord.y]];
    }

    pub fn get_legal_moves_for_player(&self, _player: usize) -> Vec::<HexCoordinate> {
        let mut legal_moves: Vec::<HexCoordinate> = Vec::<HexCoordinate>::new();
        for ((x, y), &value) in self.matrix.indexed_iter(){
            if value == 0 {
                legal_moves.push(HexCoordinate::create(x, y));
            }
        }
        return legal_moves;
    }

    fn set_board_state(&mut self, coord: HexCoordinate, new_state: i32) {
        self.matrix[[coord.x, coord.y]] = new_state;
    }

    fn next_player(&mut self){
        self.current_player += 1;
        if self.current_player > self.n_players {
            self.current_player -= self.n_players;
        }
    }

    fn coord_is_valid(&self, coord: &HexCoordinate) -> bool {
        let x_max: usize = self.matrix.shape()[0];
        let y_max: usize = self.matrix.shape()[1];
        if coord.x < x_max &&
           coord.y < y_max {
                return true;
        }
        return false;
    }
    fn coord_is_empty(&self, coord: &HexCoordinate) -> bool {
        return self.matrix[[coord.x, coord.y]] == 0;
    }
}
