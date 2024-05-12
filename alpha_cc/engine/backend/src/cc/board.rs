/*
Board is designed such that under the hood; whoever is
the current player, their pieces are always ones 1,
and the opponent's are 2.

This may lead to some confusion, but I think its faster
and less bug prone
     
*/

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
extern crate pyo3;
use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyBytes};

use bincode::{deserialize, serialize};

use crate::cc::{BoardInfo, HexCoord, Move};
use crate::cc::moves::find_all_moves;

pub const MAX_SIZE: usize = 9;
type BoardMatrix = [[i8; MAX_SIZE]; MAX_SIZE];


#[pyclass(module="alpha_cc_engine")]
#[derive(Clone)]
pub struct Board {
    size: usize,
    duration: u16,
    home_size: usize,
    matrix: BoardMatrix,
    current_player: i8,
}

impl PartialEq for Board {
    fn eq(&self, other: &Self) -> bool {
        self.matrix == other.matrix
    }
}

impl Eq for Board {}

impl Hash for Board {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.matrix.hash(state);
    }
}

impl Board {
    pub fn create(size: usize) -> Board {
        if ![3, 5, 7, 9].contains(&size) {
            panic!("supports sizes: 3, 5, 7, 9");
        }
        Board {
            size,
            duration: 0,
            home_size: Board::home_size(size),
            matrix: Board::initialize_matrix(size),
            current_player: 1,
        }
    }

    pub fn get_size(&self) -> usize {
        self.size
    }

    pub fn coord_is_empty(&self, coord: &HexCoord) -> bool {
        self.matrix[coord.x][coord.y] == 0
    }

    pub fn apply_move(&self, r#move: &Move) -> Board {
        /*
        we attempt to save a copy by flipping once (copy),
        and then perform the move on the flipped matrix:
        - reversed coords
        - played piece is 2
        */ 
        let mut matrix = self.flipped_matrix();
        matrix[self.size -1 - r#move.from_coord.x][self.size -1 - r#move.from_coord.y] = 0;
        matrix[self.size -1 - r#move.to_coord.x][self.size -1 - r#move.to_coord.y] = 2;
        Board {
            size: self.size,
            duration: self.duration + 1,
            home_size: self.home_size,
            matrix,
            current_player: if self.current_player == 1 {2} else {1},
        }
    }

    pub fn get_winner(&self) -> i8 {
        fn one_wins(board: &Board) -> bool {
            let s = board.size;
            let hs = board.home_size;
            let mut at_least_one_one_opponent_home = false;
            for x in (s-hs)..s {
                for y in (s-hs)..s {
                    if Board::xy_start_val(x, y, board.size) == 2 {
                        if board.matrix[x][y] == 0 {return false}
                        if board.matrix[x][y] == 1 {
                            at_least_one_one_opponent_home = true;
                        }
                    }
                }
            }
            at_least_one_one_opponent_home
        }
        fn two_wins(board: &Board) -> bool {
            let hs = board.home_size;
            let mut at_least_one_one_opponent_home = false;
            for x in 0..hs {
                for y in 0..hs {
                    if Board::xy_start_val(x, y, board.size) == 1{
                        if board.matrix[x][y] == 0 {return false}
                        if board.matrix[x][y] == 2 {
                            at_least_one_one_opponent_home = true;
                        }
                    }
                }
            }
            at_least_one_one_opponent_home
        }
        if one_wins(self) {return self.current_player}
        if two_wins(self) {return 3 - self.current_player}
        0
    }

    pub fn compute_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
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
                let val = self.matrix[self.size - x - 1][self.size - y - 1];
                matrix[x][y] = match val {
                    0 => 0,
                    1 => 2,
                    2 => 1,
                    _ => panic!("wtf")
                };
            } 
        }
        matrix
    }

    fn empty_matrix() -> BoardMatrix {
        [[0; MAX_SIZE]; MAX_SIZE]
    }

    #[allow(clippy::needless_range_loop)]
    fn initialize_matrix(size: usize) -> BoardMatrix {
        let mut matrix = Board::empty_matrix();
        for x in 0..MAX_SIZE {
            for y in 0..MAX_SIZE {
                matrix[x][y] = Board::xy_start_val(x, y, size);
            }
        }
        matrix
    }

    fn xy_start_val(x: usize, y: usize, size: usize) -> i8 {
        // no mans land
        if x >= size || y >= size {
            return 8;
        }
        // player 1 home
        if x + y < Board::home_size(size) {
            return 1;
        }
        // player 2 home
        if x + y >= size + Board::home_size(size) {
            return 2;
        }
        0
    }

    fn home_size(size: usize) -> usize {
        (size - 1) / 2
    }
}


#[pymethods]
impl Board {
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
                Board {
                    size: 9,
                    duration: 0,
                    home_size: 4,
                    matrix: Board::empty_matrix(),
                    current_player: 1,
                }
            },
            _ => {unreachable!()}
        }
    
    }

    pub fn reset(&self) -> Board {
        Board::create(self.size)
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
    
    #[getter]
    pub fn get_info(&self) -> BoardInfo {
        let winner = self.get_winner();
        BoardInfo {
            current_player: self.current_player,
            winner,
            size: self.size,
            duration: self.duration,
            game_over: winner > 0,
            reward: match winner {
                0 => 0,
                _ => if self.current_player == winner {1} else {-1},
            }
        }
    }

    pub fn render(&self) {
        let matrix = self.get_unflipped_matrix();
        for (i, row) in matrix[0..self.size].iter().enumerate() {
            for _ in  0..i {
                print!(" ");
            }
            for val in row[0..self.size].iter() {
                print!("{val} ");
            }
            println!();
        }
        println!("current player: {}", self.current_player);
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                (
                    self.size,
                    self.duration,
                    self.home_size,
                    self.matrix,
                    self.current_player,
                ) = deserialize(s.as_bytes()).unwrap();
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new_bound(py, &self.serialize_rs()).to_object(py))
    }

    pub fn serialize_rs(&self) -> Vec<u8> {
        let data = (
            self.size,
            self.duration,
            self.home_size,
            self.matrix,
            self.current_player,
        );
        serialize(&data).unwrap()
    }
    
    #[staticmethod]
    pub fn deserialize_rs(data: Vec<u8>) -> Board {
        let (size, duration, home_size, matrix, current_player) = deserialize(&data).unwrap();
        Board {
            size,
            duration,
            home_size,
            matrix,
            current_player,
        }
    }

    pub fn __hash__(&self) -> u64 {
        self.compute_hash()
    }
}
