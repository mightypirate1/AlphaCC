use std::vec::Vec;
use std::collections::HashMap;
use pyo3::prelude::*;

use crate::hexcoordinate::HexCoordinate;
use crate::board::Board;


#[pyclass]
#[derive(Clone, Copy, Debug)]
pub enum Move {
    Place {coord: HexCoordinate},
    Walk {from_coord: HexCoordinate, to_coord: HexCoordinate},
    Jump {from_coord: HexCoordinate, to_coord: HexCoordinate},
}

#[pyclass]
pub struct Moves {
    // Container to provide a place to offload python from some computations

    moves: Vec<Move>,
    board_size: usize,
    action_mask_indices: HashMap<usize, (HexCoordinate, HexCoordinate)>,
}

#[pyclass]
struct MoveIter {
    // helper to make Moves iterable in python
    inner: std::vec::IntoIter<Move>,
}



impl Move {
    pub fn apply(self, mut board: Board) -> Board{
        match self {
            Move::Walk{to_coord, from_coord} => board = self.move_stone(board, to_coord, from_coord),
            Move::Jump{to_coord, from_coord} => board = self.move_stone(board, to_coord, from_coord),
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

    fn move_stone(self, mut board: Board, to_coord: HexCoordinate, from_coord: HexCoordinate) -> Board {
        board.tick();
        board.clear(from_coord);
        board.place(to_coord);
        board
    }
}


impl Moves {
    pub fn create(moves: Vec<Move>, board_size: usize) -> Moves {
        Moves {
            moves,
            board_size,
            action_mask_indices: HashMap::new(),
        }
    }
}


#[pymethods]
impl MoveIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Move> {
        slf.inner.next()
    }
}


#[pymethods]
impl Moves {
    pub fn __len__(&self) -> usize {
        self.moves.len()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<MoveIter>> {
        let iter = MoveIter {
            inner: slf.moves.clone().into_iter(),
        };
        Py::new(slf.py(), iter)
    }

    pub fn __getitem__(&self, index: usize) -> Move {
        self.moves[index]
    }

    pub fn get_action_mask(& mut self) -> Vec<Vec<Vec<Vec<bool>>>> {
        let mut mask = vec![vec![vec![vec![false;self.board_size];self.board_size];self.board_size];self.board_size];
        if self.action_mask_indices.is_empty() {
            for (action_index, move_) in self.moves.iter().enumerate() {
                match move_ {
                    Move::Place {..} => { panic!("Place moves are not meant to be exposed this way") },
                    Move::Walk { from_coord, to_coord } | Move::Jump { from_coord, to_coord } => {
                        mask[from_coord.x][from_coord.y][to_coord.x][to_coord.y] = true;
                        self.action_mask_indices.insert(action_index, (*from_coord, *to_coord));
                    }
                }
            }
        }
        mask
    }
    

    pub fn get_action_mask_indices(& mut self) -> HashMap<usize, (HexCoordinate, HexCoordinate)> {
        if self.action_mask_indices.is_empty() {
            self.get_action_mask();
        }
        // TODO: question this copy
        self.action_mask_indices.clone()
    }
}
