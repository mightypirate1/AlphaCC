use std::vec::Vec;
extern crate pyo3;
use pyo3::prelude::*;




#[pyclass]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct HexCoord {
    #[pyo3(get)]
    pub x: usize,
    #[pyo3(get)]
    pub y: usize,
    board_size: usize,
}


impl HexCoord {
    pub fn create(x: usize, y: usize, board_size: usize) -> HexCoord {
        if x >= board_size || y >= board_size {
            panic!("coord created out of bounds");
        }
        HexCoord {x, y, board_size}
    }

    pub fn get_all_directions(&self) -> Vec<usize> {
        vec![0, 1, 2, 3, 4, 5]
    }

    #[allow(clippy::identity_op)]
    pub fn get_neighbor(&self, direction: usize, distance: usize) -> Option<HexCoord>{
        /*
        If I'm X, then these are the locations of my immediate neighbours:

          2   1
        3   X   0
          4   5

        In the matrix, this is expected to be:

          2 1 
        3 X 0
        4 5
        */

        let d = distance as i32;
        let x = self.x as i32;
        let y = self.y as i32;

        match direction {
            0 => HexCoord::as_option(x + 0, y + d, self.board_size),
            1 => HexCoord::as_option(x - d, y + d, self.board_size),
            2 => HexCoord::as_option(x - d, y + 0, self.board_size),
            3 => HexCoord::as_option(x + 0, y - d, self.board_size),
            4 => HexCoord::as_option(x + d, y - d, self.board_size),
            5 => HexCoord::as_option(x + d, y + 0, self.board_size),
            _ => panic!("{} is not a valid direction", direction),
        }
    }

    fn as_option(x: i32, y: i32, board_size: usize) -> Option<HexCoord> {
        if (x >= 0 && x < board_size as i32) && (y >= 0 && y < board_size as i32) {
            return Some(HexCoord{x: x as usize, y: y as usize, board_size});
        }
        None
    }
}


#[pymethods]
impl HexCoord {
    pub fn get_all_neighbours(&self, distance: usize) -> Vec<HexCoord> {
        let mb_neighbors = vec![
            self.get_neighbor(0, distance),
            self.get_neighbor(1, distance),
            self.get_neighbor(2, distance),
            self.get_neighbor(3, distance),
            self.get_neighbor(4, distance),
            self.get_neighbor(5, distance),
        ];
        mb_neighbors.into_iter().flatten().collect()
    }

    pub fn __repr__(&self) -> String {
        format!("HexCoord[{}, {}]", self.x, self.y)
    }
}