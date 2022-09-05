pub struct HexCoordinate {
    pub x: usize,
    pub y: usize,
}

impl HexCoordinate{
    pub fn create(x: usize, y: usize) -> HexCoordinate {
        return HexCoordinate {x: x, y: y};
    }

    pub fn get_neighbor(&self, direction: usize) -> HexCoordinate{
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
        match direction {
            0 => return HexCoordinate{x: self.x,   y: self.y+1},
            1 => return HexCoordinate{x: self.x-1, y: self.y+1},
            2 => return HexCoordinate{x: self.x-1, y: self.y-1},
            3 => return HexCoordinate{x: self.x,   y: self.y-1},
            4 => return HexCoordinate{x: self.x+1, y: self.y-1},
            5 => return HexCoordinate{x: self.x+1, y: self.y+1},
            _ => panic!("{} is not a valid direction", direction),
        }
    }
}
