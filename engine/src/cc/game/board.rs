/*
Board is designed such that under the hood; whoever is
the current player, their pieces are always ones 1,
and the opponent's are 2.

This may lead to some confusion, but I think its faster
and less bug prone
     
*/

use std::collections::{HashMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

#[cfg(feature = "extension-module")]
extern crate pyo3;
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use pyo3::types::{PyTuple, PyBytes};

use crate::cc::{BoardInfo, HexCoord, Move};
use crate::cc::game::moves::find_all_moves;
use crate::cc::dtypes;

pub const MAX_SIZE: usize = 9;

type BoardMatrix = [[dtypes::BoardContent; MAX_SIZE]; MAX_SIZE];


#[cfg_attr(feature = "extension-module", pyo3::prelude::pyclass(module="alpha_cc_engine", from_py_object))]
#[derive(Clone, bitcode::Encode, bitcode::Decode)]
pub struct Board {
    size: dtypes::BoardSize,
    duration: dtypes::GameDuration,
    home_size: dtypes::BoardSize,
    home_capacity: dtypes::HomeCapacity,
    matrix: BoardMatrix,
    current_player: i8,
    #[bitcode(skip)]
    cached_hash: u64,
    #[bitcode(skip)]
    cached_reward: f32,
    #[bitcode(skip)]
    cached_winner: i8,
}

impl PartialEq for Board {
    fn eq(&self, other: &Self) -> bool {
        self.matrix == other.matrix
    }
}

impl Eq for Board {}

impl Hash for Board {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.cached_hash.hash(state);
    }
}

impl Board {
    fn recompute_cache(&mut self) {
        let mut hasher = DefaultHasher::new();
        self.matrix.hash(&mut hasher);
        self.cached_hash = hasher.finish();
        let (reward, winner) = self.compute_reward_and_winner();
        self.cached_reward = reward;
        self.cached_winner = winner;
    }

    pub fn create(size: usize) -> Board {
        if ![3, 5, 7, 9].contains(&size) {
            panic!("supports sizes: 3, 5, 7, 9");
        }
        let size = size as dtypes::BoardSize;
        let mut board = Board {
            size,
            duration: 0,
            home_size: Board::home_size(size),
            home_capacity: Board::home_capacity(size),
            matrix: Board::initialize_matrix(size),
            current_player: 1,
            cached_hash: 0,
            cached_reward: 0.0,
            cached_winner: 0,
        };
        board.recompute_cache();
        board
    }

    pub fn get_size(&self) -> dtypes::BoardSize {
        self.size
    }

    pub fn get_content(&self, coord: &HexCoord) -> i8 {
        self.matrix[coord.x as usize][coord.y as usize]
    }

    pub fn coord_is_empty(&self, coord: &HexCoord) -> bool {
        self.get_content(coord) == 0
    }
    
    pub fn coord_is_player1(&self, coord: &HexCoord) -> bool {
        self.get_content(coord) == 1
    }
    
    pub fn coord_is_player2(&self, coord: &HexCoord) -> bool {
        self.get_content(coord) == 2
    }

    pub fn coord_is_player1_home(&self, coord: &HexCoord) -> bool {
        Board::xy_start_val(coord.x, coord.y, self.size) == 1
    }

    pub fn coord_is_player2_home(&self, coord: &HexCoord) -> bool {
        Board::xy_start_val(coord.x, coord.y, self.size) == 2
    }

    pub fn apply_move(&self, r#move: &Move) -> Board {
        /*
        we attempt to save a copy by flipping once (copy),
        and then perform the move on the flipped matrix:
        - flipped board
        - flipped coords
        - played piece is 2 (flipped)

        most moves will leave the from_coord empty, but some moves
        swap the stone with the opponents stone. thus we will put
        whatever content is at the to_coord cell at the from_coord cell,
        and trust the move finding algorithm to have checked that the
        result is valid.
        */

        let mut matrix = self.flipped_matrix();
        let flipped_from_coord = r#move.from_coord.flip();
        let flipped_to_coord = r#move.to_coord.flip();
        let flipped_to_x = flipped_to_coord.x as usize;
        let flipped_to_y = flipped_to_coord.y as usize;
        let flipped_from_x = flipped_from_coord.x as usize;
        let flipped_from_y = flipped_from_coord.y as usize;
        let to_coord_content = matrix[flipped_to_x][flipped_to_y];  

        matrix[flipped_from_x][flipped_from_y] = to_coord_content;
        matrix[flipped_to_x][flipped_to_y] = 2;  // 2 is the current player (since the board is flipped)
        let mut board = Board {
            size: self.size,
            duration: self.duration + 1,
            home_size: self.home_size,
            home_capacity: self.home_capacity,
            matrix,
            current_player: if self.current_player == 1 {2} else {1},
            cached_hash: 0,
            cached_reward: 0.0,
            cached_winner: 0,
        };
        board.recompute_cache();
        board
    }

    pub fn compute_hash(&self) -> dtypes::BoardHash {
        self.cached_hash
    }

    pub fn compute_reward_and_winner(&self) -> (f32, i8) {
        /*
        compute the reward for a board that may or may not be terminal.
        
        for terminal states:
            the reward is 1 if the current player wins, -1 if the opponent wins.

        for non-terminal states:
            - let the score for a player be the number of stones they have in the opponents home.
            - the reward is then the difference in score divided by the number of stones that fit
                in a home region.

         */
        let s = self.size;
        let hs = self.home_size;
        
        let mut n_p1_stones_in_p2_home = 0;
        let mut goal_is_full: bool = true;
        for x in (s-hs)..s {
            for y in (s-hs)..s {
                if Board::xy_start_val(x, y, self.size) == 2 {
                    let coord = HexCoord::new(x, y, self.size);
                    let content = self.get_content(&coord);
                    if content == 0 {
                        goal_is_full = false;
                    }
                    if content == 1 {
                        n_p1_stones_in_p2_home += 1;
                    }
                }
            }
        }
        // p1 wins
        if goal_is_full && n_p1_stones_in_p2_home > 0 {
            return (1.0, self.current_player);
        }


        let mut n_p2_stones_in_p1_home = 0;
        goal_is_full = true;
        for x in 0..hs {
            for y in 0..hs {
                if Board::xy_start_val(x, y, self.size) == 1 {
                    let coord = HexCoord::new(x, y, self.size);
                    let content = self.get_content(&coord);
                    if content == 0 {
                        goal_is_full = false;
                    }
                    if content == 2 {
                        n_p2_stones_in_p1_home += 1;
                    }
                }
            }
        }
        // p2 wins
        if goal_is_full && n_p2_stones_in_p1_home > 0 {
            return (-1.0, 3 - self.current_player);
        }

        // non-terminal state        
        let reward = (n_p1_stones_in_p2_home - n_p2_stones_in_p1_home) as f32 / self.home_capacity as f32;
        (reward, 0)
    }

    #[allow(clippy::needless_range_loop)]
    fn flipped_matrix(&self) -> BoardMatrix {
        /*
        Used when playing moves to make it look to the next player like they are the main character
         */
        let mut matrix = Board::empty_matrix();
        // manual version for now
        
        for x in 0..self.size {
            for y in 0..self.size {
                let coord = HexCoord::new(x, y, self.size);
                let flipped_content = self.get_content(&coord.flip());
                matrix[x as usize][y as usize] = match flipped_content {
                    0 => 0,
                    1 => 2,
                    2 => 1,
                    _ => {
                        unreachable!("invalid value on board: {flipped_content}")
                    }
                };
            } 
        }
        matrix
    }

    fn empty_matrix() -> BoardMatrix {
        [[0; MAX_SIZE]; MAX_SIZE]
    }

    #[allow(clippy::needless_range_loop)]
    fn initialize_matrix(size: dtypes::BoardSize) -> BoardMatrix {
        let mut matrix = Board::empty_matrix();
        for x in 0..MAX_SIZE.try_into().unwrap() {
            for y in 0..MAX_SIZE.try_into().unwrap() {
                matrix[x as usize][y as usize] = Board::xy_start_val(x, y, size);
            }
        }
        matrix
    }

    fn xy_start_val(x: dtypes::BoardSize, y: dtypes::BoardSize, size: dtypes::BoardSize) -> i8 {
        // player 1 home
        let home_size = Board::home_size(size);
        if x + y < home_size {
            return 1;
        }
        // player 2 home
        if x + y >= home_size + size && x < size && y < size {
            return 2;
        }
        0
    }

    fn home_size(size: dtypes::BoardSize) -> dtypes::BoardSize {
        (size - 1) / 2
    }

    fn home_capacity(size: dtypes::BoardSize) -> dtypes::HomeCapacity {
        let hs = Board::home_size(size) as usize;
        ((hs * (hs + 1)) / 2) as dtypes::HomeCapacity
    }

    pub fn serialize_rs(&self) -> dtypes::EncBoard {
        bitcode::encode(self)
    }

    pub fn deserialize_rs(data: &[u8]) -> Board {
        let mut board: Board = bitcode::decode(data)
            .unwrap_or_else(|e| {
                panic!("Failed to deserialize board state: {}", e)
            });
        board.recompute_cache();
        board
    }
}


/// Methods used from both Rust and Python.
/// When the `extension-module` feature is active, pyo3 exposes them to Python.
#[cfg_attr(feature = "extension-module", pyo3::prelude::pymethods)]
impl Board {
    pub fn reset(&self) -> Board {
        Board::create(self.size as usize)
    }

    pub fn get_moves(&self) -> Vec<Move> {
        find_all_moves(self)
    }

    pub fn get_next_states(&self) -> Vec<Board> {
        let mut next_states: Vec<Board> = Vec::new();
        for r#move in self.get_moves() {
            next_states.push(self.apply_move(&r#move));
        }
        next_states
    }

    pub fn get_matrix(&self) -> BoardMatrix {
        self.matrix
    }

    pub fn apply(&self, r#move: &Move) -> Board {
        self.apply_move(r#move)
    }

    pub fn get_unflipped_matrix(&self) -> BoardMatrix {
        /*
        unflips the matrix if needed to make it intelligable for humans
         */
        if self.current_player == 1 {
            return self.matrix
        }
        self.flipped_matrix()
    }

    pub fn render(&self) {
        let tokens = HashMap::from([
            (0, "·"),
            (1, "⬣"),
            (2, "⎔"),
        ]);
        let matrix = self.get_unflipped_matrix();
        println!();
        for (i, row) in matrix[0..self.size as usize].iter().enumerate() {
            for _ in  0..i {
                print!(" ");
            }
            for val in row[0..self.size as usize].iter() {
                print!("{} ", tokens.get(val).unwrap());
            }
            println!();
        }
        println!();
        println!("current player: {} ({})", tokens.get(&self.current_player).unwrap(), self.current_player);
    }

}

impl Board {
    pub fn get_info(&self) -> BoardInfo {
        BoardInfo {
            current_player: self.current_player,
            winner: self.cached_winner,
            reward: self.cached_reward,
            size: self.size,
            duration: self.duration,
            game_over: self.cached_winner > 0,
        }
    }
}


#[cfg(feature = "extension-module")]
#[pymethods]
impl Board {
    #[getter]
    pub fn info(&self) -> BoardInfo {
        self.get_info()
    }

    #[new]
    #[pyo3(signature = (*py_args))]
    fn new(py_args: &Bound<'_, PyTuple>) -> Self {
        // complicated constructor to allow pickling (allows call to __new__ with empty args)
        match py_args.len() {
            1 => {
                if let Ok(size) = py_args.get_item(0).unwrap().extract::<usize>() {
                    return Board::create(size)
                }
                panic!("expected a single int as input");
            },
            0 => {
                Board::create(9)
            },
            _ => {unreachable!()}
        }

    }

    pub fn __setstate__(&mut self, py: Python, state: Py<PyAny>) -> PyResult<()> {
        let py_bytes = state.extract::<Bound<'_, PyBytes>>(py)?;
        let bytes = py_bytes.as_bytes();
        let board = Board::deserialize_rs(bytes);
        *self = board;
        Ok(())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.serialize_rs()))
    }

    pub fn __hash__(&self) -> u64 {
        self.compute_hash()
    }

    pub fn __eq__(&self, other: &Board) -> bool {
        self == other
    }
}
