extern crate pyo3;
use pyo3::prelude::*;
use std::vec::Vec;

#[derive(Copy, Clone, Eq, Debug)]
#[pyclass]
pub struct HexCoordinate {
    pub x: i32,
    pub y: i32,
}

impl PartialEq for HexCoordinate {
    fn eq(&self, other: &Self) -> bool {
        return self.x == other.x && self.y == other.y;
    }
}

// #[pymethods]
impl HexCoordinate{
    pub fn create(x: i32, y: i32) -> HexCoordinate {
        return HexCoordinate {x: x, y: y};
    }
    pub fn from_usize(x: usize, y: usize) -> HexCoordinate {
        return HexCoordinate {x: x as i32, y: y as i32};
    }

    pub fn get_all_neighbours(&self, distance: usize) -> Vec<HexCoordinate> {
        return vec![
            self.get_neighbor(0, distance),
            self.get_neighbor(1, distance),
            self.get_neighbor(2, distance),
            self.get_neighbor(3, distance),
            self.get_neighbor(4, distance),
            self.get_neighbor(5, distance),
        ];
    }

    pub fn get_all_directions(&self) -> Vec<usize> {
        return vec![0, 1, 2, 3, 4, 5];
    }

    pub fn get_neighbor(&self, direction: usize, distance: usize) -> HexCoordinate{
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

        let d: i32 = distance as i32;

        match direction {
            0 => return HexCoordinate{ x: self.x,     y: self.y + d },
            1 => return HexCoordinate{ x: self.x - d, y: self.y + d },
            2 => return HexCoordinate{ x: self.x - d, y: self.y     },
            3 => return HexCoordinate{ x: self.x,     y: self.y - d },
            4 => return HexCoordinate{ x: self.x + d, y: self.y - d },
            5 => return HexCoordinate{ x: self.x + d, y: self.y     },
            _ => panic!("{} is not a valid direction", direction),
        }
    }
}
