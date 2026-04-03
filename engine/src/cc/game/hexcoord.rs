#[cfg(feature = "extension-module")]
extern crate pyo3;

use crate::cc::dtypes;



#[cfg_attr(feature = "extension-module", pyo3::prelude::pyclass(module="alpha_cc_engine", from_py_object))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct HexCoord {
    pub x: dtypes::BoardSize,
    pub y: dtypes::BoardSize,
    board_size: dtypes::BoardSize,
}


impl HexCoord {
    pub fn new(x: dtypes::BoardSize, y: dtypes::BoardSize, board_size: dtypes::BoardSize) -> HexCoord {
        if x >= board_size || y >= board_size {
            panic!("coord created out of bounds");
        }
        HexCoord {
            x,
            y,
            board_size,
        }
    }
    pub fn create(x: usize, y: usize, board_size: usize) -> HexCoord {
        HexCoord::new(x as dtypes::BoardSize, y as dtypes::BoardSize, board_size as dtypes::BoardSize)
    }

    pub fn get_all_neighbours_arr(&self, distance: usize) -> [Option<HexCoord>; 6] {
        [
            self.get_neighbor(0, distance),
            self.get_neighbor(1, distance),
            self.get_neighbor(2, distance),
            self.get_neighbor(3, distance),
            self.get_neighbor(4, distance),
            self.get_neighbor(5, distance),
        ]
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
            _ => unreachable!("{} is not a valid direction", direction),
        }
    }

    fn as_option(x: i32, y: i32, board_size: dtypes::BoardSize) -> Option<HexCoord> {
        if (x >= 0 && x < board_size as i32) && (y >= 0 && y < board_size as i32) {
            let coord = HexCoord{
                x: x as dtypes::BoardSize,
                y: y as dtypes::BoardSize,
                board_size,
            };
            return Some(coord);
        }
        None
    }
}


/// Methods used from both Rust and Python.
#[cfg_attr(feature = "extension-module", pyo3::prelude::pymethods)]
impl HexCoord {
    pub fn flip(&self) -> HexCoord {
        /*
        the game engine will flip boards so that player 1 is always at the top,
        so coordinates need to be flipped accordingly
         */
        HexCoord {
            x: self.board_size -1 - self.x,
            y: self.board_size -1 - self.y,
            board_size: self.board_size,
        }
    }

    pub fn repr(&self) -> String {
        format!("HexCoord[{}, {}]", self.x, self.y)
    }
}

#[cfg(feature = "extension-module")]
#[pyo3::prelude::pymethods]
impl HexCoord {
    pub fn get_all_neighbours(&self, distance: usize) -> Vec<HexCoord> {
        self.get_all_neighbours_arr(distance).into_iter().flatten().collect()
    }

    #[getter]
    fn get_x(&self) -> dtypes::BoardSize {
        self.x
    }

    #[getter]
    fn get_y(&self) -> dtypes::BoardSize {
        self.y
    }

    pub fn __repr__(&self) -> String {
        self.repr()
    }
}
