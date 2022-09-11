extern crate pyo3;
use pyo3::prelude::*;
use crate::moves::Move;
use std::vec::Vec;
use ndarray::{s, Array2};
use crate::hexcoordinate::HexCoordinate;

#[pyclass]
pub struct Board {
    current_player: usize,
    n_players: usize,
    matrix: Array2::<i32>,
    calculated_moves: Vec<Move>,
    pub allow_place_moves: bool,
}

#[pyclass]
pub struct BoardInfo {
    #[pyo3(get)]
    current_player: usize,
    #[pyo3(get)]
    winner: usize,
}

impl Default for Board {
    fn default() -> Board {
        return Board {
            current_player: 1,
            n_players: 2,
            matrix: Array2::zeros((9, 9)),
            calculated_moves: Vec::new(),
            allow_place_moves: false,
        }
    }
}

impl Board {
    pub fn create(size: usize) -> Board {
        let mut board = Board {
            matrix: Array2::zeros((size, size)),
            ..Default::default()
        };
        board.initialize_board();
        return board;
    }

    // Copy functionality
    pub fn copy(&self) -> Board{
        return Board {
            current_player: self.current_player,
            n_players: self.n_players,
            matrix: self.matrix.to_owned(),
            calculated_moves: Vec::new(),
            allow_place_moves: self.allow_place_moves,
        };
    }

    // Rendering
    pub fn print(&self) {
        let mut row;
        let width = self.matrix.shape()[0];
        let height = self.matrix.shape()[1];
        for row_idx in 0..height {
            row = self.matrix.slice(s![row_idx, ..]);
            print!("{}", (0..row_idx+1).map( |_| "  ").collect::<String>() );
            print!("{row}");
            print!("{}\n", (0..width-row_idx).map( |_| "  ").collect::<String>() );
        }
        println!("Current player: {}", self.current_player);
    }

    /////////////////////
    // Initialization: //
    /////////////////////

    fn initialize_board(& mut self) {
        let board_size: usize = self.get_board_size();
        self.matrix = Array2::zeros((board_size, board_size));
        for x in 0..board_size {
            for y in 0..board_size {
                if self.coord_is_in_home_of_player(1, HexCoordinate::from_usize(x, y)) {
                    self.matrix[[y, x]] = 1;
                }
                if self.coord_is_in_home_of_player(2, HexCoordinate::from_usize(x, y)) {
                    self.matrix[[y, x]] = 2;
                }
            }
        }
    }

    //////////////
    // Getters: //
    //////////////

    pub fn get_n_players(&self) -> usize {
        return self.n_players;
    }
    pub fn place_moves_are_allowed(&self) -> bool {
        return self.allow_place_moves;
    }
    fn get_board_size(&self) -> usize {
        return self.matrix.shape()[0];
    }
    fn get_home_size(&self) -> usize {
        return (self.get_board_size() - 1) / 2;
    }

    ////////////////////////////
    // Core board operations: //
    ////////////////////////////

    pub fn place(&mut self, coord: HexCoordinate) {
        if  self.coord_is_valid(&coord) && self.coord_is_empty(&coord) {
                self.set_board_state(coord, self.current_player as i32);
        }
        else {
            panic!("Invalid play attempted: {} @ {}, {}", self.current_player, coord.x, coord.y);
        }
    }
    pub fn clear(&mut self, coord: HexCoordinate) {
        if self.coord_is_valid(&coord){
            let current: i32 = self.get_boardstate_by_coord(&coord);
            match current {
                -1 => panic!("attempted to clear a dead node: ({},{})", coord.x, coord.y),
                0  => panic!("attempted to clear an empty node: ({},{})", coord.x, coord.y),
                _ => self.set_board_state(coord, 0),
            }
        }
    }
    fn set_board_state(&mut self, coord: HexCoordinate, new_state: i32) {
        if self.coord_is_valid(&coord){
            self.matrix[[coord.x as usize, coord.y as usize]] = new_state;
        }
        else{
            panic!("invalid coord for set state")
        }
    }
    fn next_player(&mut self) {
        self.current_player += 1;
        if self.current_player > self.n_players {
            self.current_player = (self.current_player as i32 - self.n_players as i32) as usize;
        }
    }

    ////////////////////////////////////////////////
    // Utils for checking board status at coords: //
    ////////////////////////////////////////////////

    pub fn coord_is_empty(&self, coord: &HexCoordinate) -> bool {
        return self.get_boardstate_by_coord(coord) == 0;
    }
    pub fn coord_is_occupied(&self, coord: &HexCoordinate) -> bool {
        return self.get_boardstate_by_coord(coord) > 0
            && self.get_boardstate_by_coord(coord) < self.n_players as i32;
    }
    pub fn coord_is_occupied_by_current_player(&self, coord: &HexCoordinate) -> bool {
        return self.get_boardstate_by_coord(coord) == self.current_player as i32;
    }
    pub fn get_boardstate_by_coord(&self, coord: &HexCoordinate) -> i32 {
        if self.coord_is_valid(&coord){
            return self.matrix[[coord.x as usize, coord.y as usize]];
        }
        return -1;
    }
    fn coord_is_valid(&self, coord: &HexCoordinate) -> bool {
        let x_max: usize = self.matrix.shape()[0];
        let y_max: usize = self.matrix.shape()[1];
        if   coord.x > -1 &&
             coord.y > -1 &&
            (coord.x as usize) < x_max &&
            (coord.y as usize) < y_max {
                return true;
        }
        return false;
    }
    fn coord_is_in_home_of_player(&self, player: usize, coord: HexCoordinate) -> bool {
        let home_size = self.get_home_size() as i32;
        let board_size = self.get_board_size() as i32;
        match player {
            1 => {return coord.x + coord.y < home_size},
            2 => {return coord.x + coord.y + 1 >= 2 * board_size - home_size},
            _ => {panic!("no such player: {player}");}
        }
    }

    //////////////////////////////////////
    // Getters for player piece coords: //
    //////////////////////////////////////

    pub fn get_player_piece_coords(&self, player: usize) -> Vec::<HexCoordinate> {
        let mut coords: Vec::<HexCoordinate> = Vec::new();
        for ((x, y), &value) in self.matrix.indexed_iter(){
            if value == player as i32 {
                coords.push(HexCoordinate::from_usize(x, y));
            }
        }
        return coords;
    }

    pub fn get_current_player_piece_coords(&self) -> Vec::<HexCoordinate>{
        return self.get_player_piece_coords(self.current_player);
    }

    ////////////////////
    // Win-condition: //
    ////////////////////

    fn current_win_status(&self) -> usize {
        if self.player_one_has_won(){
            return 1;
        }
        if self.player_two_has_won(){
            return 2;
        }
        return 0;
    }
    fn player_one_has_won(&self) -> bool {
        let home_size: usize = self.get_home_size();
        let board_size: usize = self.get_board_size();
        for ((y, x), value) in self.matrix.slice(
            s![board_size-home_size.., board_size-home_size..]  // Player2's home, i.e. player1's destination!
        ).indexed_iter() {
            if self.coord_is_in_home_of_player(2, HexCoordinate::from_usize(x+board_size-home_size, y+board_size-home_size))
                && *value != 1 {
                    return false;
                }
        }
        return true;
    }
    fn player_two_has_won(&self) -> bool {
        let home_size: usize = self.get_home_size();
        for ((y, x), value) in self.matrix.slice(
            s![0..home_size, 0..home_size] // Player1's home, i.e. player2's destination!
        ).indexed_iter() {
            if self.coord_is_in_home_of_player(1, HexCoordinate::from_usize(x, y))
                && *value != 2 {
                    return false;
                }
        }
        return true;
    }

    //////////////////////////////
    // Move finding algorithms: //
    //////////////////////////////

    pub fn get_all_legal_moves_for_current_player(&self) -> Vec<Move>{
        return self.get_all_legal_moves(self.current_player);
    }

    pub fn get_all_legal_moves(&self, player: usize) -> Vec<Move> {
        let mut moves = Vec::new();
        let mut candidate_coord: HexCoordinate;
        if self.place_moves_are_allowed() {
            for coord in self.get_player_piece_coords(0) {
                moves.push(Move::PlaceMove{coord: coord});
            }
        }
        // For each stone of the current player:
        for coord in self.get_player_piece_coords(player) {
            // For each direction that the hex-grid allows:
            for direction in coord.get_all_directions(){
                candidate_coord = coord.get_neighbor(direction, 1);
                if self.coord_is_empty(&candidate_coord){
                    // If the immediate neighbor is empty, it's legal to walk there
                    moves.push(Move::WalkMove{from: coord, to: candidate_coord});
                }
            }
            moves.extend(self.find_all_jumps_from_coord(coord));
        }
        return moves;
    }

    fn find_all_jumps_from_coord (&self, coord: HexCoordinate) -> Vec<Move> {
        let mut jump_moves: Vec<Move> = Vec::new();
        let mut final_positions: Vec<HexCoordinate> = Vec::new();
        self.recusive_exporation(coord, coord, & mut final_positions, & mut jump_moves);
        return jump_moves;
    }

    fn recusive_exporation<'a>(
        &self,
        starting_coord: HexCoordinate,
        current_coord: HexCoordinate,
        mut final_positions: &'a mut Vec<HexCoordinate>,
        mut jump_moves: &'a mut Vec<Move>,
    ) -> &'a Vec<Move> {
        let mut target_coord: HexCoordinate;
        let mut intermediate_coord: HexCoordinate;
        for direction in current_coord.get_all_directions() {
            target_coord = current_coord.get_neighbor(direction, 2);
            intermediate_coord = current_coord.get_neighbor(direction, 1);
            if self.coord_is_empty(&target_coord)
                && self.coord_is_occupied(&intermediate_coord)
                && !final_positions.contains(&target_coord) {
                    // If `current_coord` -> `target_coord` woul be a legal jump,
                    // and this is the fist time we encounter `target_coord`:
                    // - Remember that we already know we can get to `target_coord` from `starting_coord`.
                    // - Add `starting_coord` -> `target_coord` to list of JumpMoves.
                    final_positions.push(target_coord);
                    jump_moves.push(Move::JumpMove{from: starting_coord, to: target_coord});
                    self.recusive_exporation(starting_coord, target_coord, & mut final_positions, & mut jump_moves);
            }
        }
        return jump_moves;
    }
}

/////////
// RL: //
/////////

#[pymethods]
impl Board {
    // Constructor
    #[new]
    pub fn pycreate(size: usize) -> PyResult<Self> {
        return Ok(Board::create(size));
    }
    pub fn reset(mut slf: PyRefMut<'_, Self>) -> Board {
        // Resets board to starting position, randomizes starting player, and returns state
        let size: usize = slf.matrix.shape()[0];
        slf.matrix = Array2::zeros((size, size));
        slf.initialize_board();
        slf.calculated_moves = Vec::new();
        slf.current_player = if rand::random() {1} else {2};
        return slf.copy();
    }

    pub fn get_all_possible_next_states(& mut self) -> Vec<Board> {
        // IMPORTANT: This is being set here so that we can have an action space that
        // is just a usize; namely the index in this list.
        self.calculated_moves = self.get_all_legal_moves_for_current_player();
        let mut next_boards: Vec<Board> = Vec::new();
        let mut next_board: Board;
        for a_move in &self.calculated_moves {
            next_board = a_move.apply(self.copy());
            next_boards.push(next_board);
        }
        return next_boards;
    }

    pub fn perform_move(slf: PyRef<'_, Self>, move_index: usize) -> Board {
        let mut new_board_state: Board;
        if move_index < slf.calculated_moves.len() {
            new_board_state = slf.calculated_moves[move_index].apply(slf.copy());
            new_board_state.next_player();
            return new_board_state;
        }
        panic!("Invalid move index {move_index}: valid choices are {:?}", [0..slf.calculated_moves.len()]);
    }

    pub fn render(&self) {
        self.print();
    }

    pub fn get_matrix(&self) -> Vec<Vec<i32>> {
        let mut mtx: Vec<Vec<i32>> = Vec::new();
        for col in self.matrix.outer_iter() {
            mtx.push(col.to_vec());
        }
        return mtx;
    }

    pub fn get_board_info(&self) -> BoardInfo {
        return BoardInfo { current_player: self.current_player, winner: self.current_win_status()}
    }
}
