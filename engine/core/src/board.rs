use std::hash::{Hash, Hasher};
use crate::{BoardInfo, dtypes::{BoardHash, BoardSize}, Move};


pub trait CellContent: Copy + From<i8> {
    /// Swap player perspective: P1↔P2, empty stays empty.
    fn flip(self) -> Self;
    /// Which player owns this cell (1 or 2), or 0 if empty.
    fn player(self) -> i8;
}

pub trait Coord: Copy + Eq + std::hash::Hash + Send {
    fn new(x: BoardSize, y: BoardSize, size: BoardSize) -> Self;
    fn flip(&self) -> Self;
    fn xy(&self) -> (BoardSize, BoardSize);
}

pub trait Board: Clone + Eq + Hash + Send + Sync {
    type Coord: Coord;
    type Content: CellContent;

    fn apply_move(&self, r#move: &Move<Self::Coord>) -> Self;
    fn legal_moves(&self) -> Vec<Move<Self::Coord>>;
    fn get_info(&self) -> BoardInfo;

    fn serialize(&self) -> Vec<u8>;
    fn deserialize(encoded: &[u8]) -> Self;

    // playable size, and stored size
    fn get_sizes(&self) -> (BoardSize, BoardSize);
    fn get_cell(&self, coord: &Self::Coord) -> Self::Content;
    fn get_content(&self, coord: &Self::Coord) -> i8;
    fn get_cell_unflipped(&self, coord: &Self::Coord) -> Self::Content {
        if self.get_info().current_player == 1 {
            self.get_cell(coord)
        } else {
            self.get_cell(&coord.flip()).flip()
        }
    }
    fn compute_hash(&self) -> BoardHash {
        let mut hasher = std::hash::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}


impl Coord for crate::cc::HexCoord {
    fn new(x: BoardSize, y: BoardSize, board_size: BoardSize) -> Self {
        crate::cc::HexCoord::new(x, y, board_size)
    }
    fn flip(&self) -> Self {
        self.flip()
    }
    fn xy(&self) -> (BoardSize, BoardSize) {
        (self.x, self.y)
    }
}

impl Board for crate::cc::CCBoard {
    type Coord = crate::cc::HexCoord;
    type Content = crate::cc::CCContent;

    fn apply_move(&self, r#move: &Move<Self::Coord>) -> Self {
        self.apply(r#move)
    }
    fn legal_moves(&self) -> Vec<Move<Self::Coord>> {
        crate::cc::moves::find_all_moves(self)
    }
    fn get_info(&self) -> BoardInfo {
        self.get_info()
    }
    fn get_cell(&self, coord: &Self::Coord) -> Self::Content {
        crate::cc::CCContent::from(self.get_content(coord))
    }
    fn get_content(&self, coord: &Self::Coord) -> i8 {
        self.get_content(coord)
    }
    fn get_sizes(&self) -> (BoardSize, BoardSize) {
        (self.get_size(), 9)
    }
    fn serialize(&self) -> Vec<u8> {
        self.serialize_rs()
    }
    fn deserialize(encoded: &[u8]) -> Self {
        Self::deserialize_rs(encoded)
    }
}
