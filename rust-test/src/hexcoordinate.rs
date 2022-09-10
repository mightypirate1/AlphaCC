use std::vec::Vec;

#[derive(Copy, Clone, Eq, Debug)]
pub struct HexCoordinate {
    pub x: i32,
    pub y: i32,
}

impl PartialEq for HexCoordinate {
    fn eq(&self, other: &Self) -> bool {
        return self.x == other.x && self.y == other.y;
    }
}

impl HexCoordinate{
    pub fn create(x: i32, y: i32) -> HexCoordinate {
        return HexCoordinate {x: x, y: y};
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
        If I'm X, then these are the locations of my neighbours:

          2   1
        3   X   0
          4   5

        In the matrix, this is expected to be:

        2 _ 1
        3 X 0
        4 _ 5
        */

        if distance != 1 {
            // Don't forget the current move stuff is wrong. Probably only works for some coords...
            panic!("only distance 1 implemented")
        }

        match direction {
            0 => return HexCoordinate{x: self.x,   y: self.y+2},
            1 => return HexCoordinate{x: self.x,   y: self.y+1},
            2 => return HexCoordinate{x: self.x-1, y: self.y-1},
            3 => return HexCoordinate{x: self.x,   y: self.y-2},
            4 => return HexCoordinate{x: self.x+1, y: self.y-1},
            5 => return HexCoordinate{x: self.x+1, y: self.y+1},
            _ => panic!("{} is not a valid direction", direction),
        }
    }
}
