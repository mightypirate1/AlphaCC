/*
Board is designed such that under the hood; whoever is
the current player, their pieces are always ones 1,
and the opponent's are 2.

This may lead to some confusion, but I think its faster
and less bug prone

*/

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::{BoardInfo, cc::HexCoord, Move, WDL};
use crate::cc::moves::find_all_moves;
use crate::dtypes;

pub const MAX_SIZE: usize = 9;

pub type CCBoardMatrix = [[dtypes::BoardContent; MAX_SIZE]; MAX_SIZE];

/// Typed cell content for Chinese Checkers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CCContent {
    Empty = 0,
    Player1 = 1,
    Player2 = 2,
}

impl From<i8> for CCContent {
    fn from(v: i8) -> Self {
        match v {
            1 => Self::Player1,
            2 => Self::Player2,
            _ => Self::Empty,
        }
    }
}

impl crate::board::CellContent for CCContent {
    fn flip(self) -> Self {
        match self {
            Self::Player1 => Self::Player2,
            Self::Player2 => Self::Player1,
            Self::Empty => Self::Empty,
        }
    }

    fn player(self) -> i8 {
        match self {
            Self::Player1 => 1,
            Self::Player2 => 2,
            Self::Empty => 0,
        }
    }
}


#[derive(Clone, bitcode::Encode, bitcode::Decode)]
pub struct CCBoard {
    size: dtypes::BoardSize,
    duration: dtypes::GameDuration,
    home_size: dtypes::BoardSize,
    home_capacity: dtypes::HomeCapacity,
    matrix: CCBoardMatrix,
    current_player: i8,
}

impl PartialEq for CCBoard {
    fn eq(&self, other: &Self) -> bool {
        self.matrix == other.matrix
    }
}

impl Eq for CCBoard {}

impl Hash for CCBoard {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.matrix.hash(state);
    }
}

impl CCBoard {
    pub fn create(size: usize) -> Self {
        if ![3, 5, 7, 9].contains(&size) {
            panic!("supports sizes: 3, 5, 7, 9");
        }
        let size = size as dtypes::BoardSize;
        Self {
            size,
            duration: 0,
            home_size: Self::home_size(size),
            home_capacity: Self::home_capacity(size),
            matrix: Self::initialize_matrix(size),
            current_player: 1,
        }
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
        Self::xy_start_val(coord.x, coord.y, self.size) == 1
    }

    pub fn coord_is_player2_home(&self, coord: &HexCoord) -> bool {
        Self::xy_start_val(coord.x, coord.y, self.size) == 2
    }

    pub fn apply_move(&self, r#move: &Move<HexCoord>) -> Self {
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
        Self {
            size: self.size,
            duration: self.duration + 1,
            home_size: self.home_size,
            home_capacity: self.home_capacity,
            matrix,
            current_player: if self.current_player == 1 {2} else {1},
        }
    }

    pub fn compute_hash(&self) -> dtypes::BoardHash {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    pub fn compute_wdl_and_winner(&self) -> (WDL, i8, bool) {
        /*
        Compute (WDL, winner, game_over) for a board state.

        Terminal (goal-fill rule): the player who has all goal cells filled
        with at least one own piece in the opponent's home wins. game_over=true.

        Detects stalling by ruling games as a loss for the stalling player.
         */
        let s = self.size;
        let hs = self.home_size;

        let mut n_p1_stones_in_p2_home: i32 = 0;
        let mut goal_is_full: bool = true;
        for x in (s-hs)..s {
            for y in (s-hs)..s {
                if Self::xy_start_val(x, y, self.size) == 2 {
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
        // p1 wins (current player)
        if goal_is_full && n_p1_stones_in_p2_home > 0 {
            return (WDL::win(), self.current_player, true);
        }

        let mut n_p2_stones_in_p1_home: i32 = 0;
        goal_is_full = true;
        for x in 0..hs {
            for y in 0..hs {
                if Self::xy_start_val(x, y, self.size) == 1 {
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
        // p2 wins (opponent)
        if goal_is_full && n_p2_stones_in_p1_home > 0 {
            return (WDL::loss(), 3 - self.current_player, true);
        }

        // If game is not conclusively over: apply "best straggler" rule.
        // Each player's worst piece is the one with the lowest progress toward
        // their target. Whoever's worst piece has advanced further wins; equal
        // worst-progress is a draw.
        //
        // This is one of many possible ways of addressing what seems to be a
        // flaw in the original rules for comptetitive play: walling is a legit
        // and unbeatable stalling strategy, and the agent will eventually
        // learn that. Play at that point the game becomes nonsense. In a
        // living room situation, that is a flipped table and a ruined vibe.
        // Thus we avoid that by adding a rule that makes walling a losing
        // strategy.
        let mut my_worst: i32 = i32::MAX;
        let mut their_worst: i32 = i32::MAX;
        let bound = 2 * (s as i32 - 1);
        for x in 0..s {
            for y in 0..s {
                let coord = HexCoord::new(x, y, self.size);
                match self.get_content(&coord) {
                    1 => {
                        let prog = x as i32 + y as i32;
                        if prog < my_worst { my_worst = prog; }
                    }
                    2 => {
                        let prog = bound - (x as i32 + y as i32);
                        if prog < their_worst { their_worst = prog; }
                    }
                    _ => {}
                }
            }
        }

        if my_worst > their_worst {
            (WDL::win(), self.current_player, false)
        } else if my_worst < their_worst {
            (WDL::loss(), 3 - self.current_player, false)
        } else {
            (WDL::draw(), 0, false)
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn flipped_matrix(&self) -> CCBoardMatrix {
        /*
        Used when playing moves to make it look to the next player like they are the main character
         */
        let mut matrix = Self::empty_matrix();
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

    fn empty_matrix() -> CCBoardMatrix {
        [[0; MAX_SIZE]; MAX_SIZE]
    }

    #[allow(clippy::needless_range_loop)]
    fn initialize_matrix(size: dtypes::BoardSize) -> CCBoardMatrix {
        let mut matrix = Self::empty_matrix();
        for x in 0..MAX_SIZE.try_into().unwrap() {
            for y in 0..MAX_SIZE.try_into().unwrap() {
                matrix[x as usize][y as usize] = Self::xy_start_val(x, y, size);
            }
        }
        matrix
    }

    fn xy_start_val(x: dtypes::BoardSize, y: dtypes::BoardSize, size: dtypes::BoardSize) -> i8 {
        // player 1 home
        let home_size = Self::home_size(size);
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
        let hs = Self::home_size(size) as usize;
        ((hs * (hs + 1)) / 2) as dtypes::HomeCapacity
    }

    pub fn serialize_rs(&self) -> dtypes::EncBoard {
        bitcode::encode(self)
    }

    pub fn deserialize_rs(data: &[u8]) -> Self {
        bitcode::decode(data)
            .unwrap_or_else(|e| {
                panic!("Failed to deserialize board state: {}", e)
            })
    }

    pub fn reset(&self) -> Self {
        Self::create(self.size as usize)
    }

    pub fn get_moves(&self) -> Vec<Move<HexCoord>> {
        find_all_moves(self)
    }

    pub fn get_next_states(&self) -> Vec<Self> {
        let mut next_states: Vec<Self> = Vec::new();
        for r#move in self.get_moves() {
            next_states.push(self.apply_move(&r#move));
        }
        next_states
    }

    pub fn get_matrix(&self) -> CCBoardMatrix {
        self.matrix
    }

    pub fn apply(&self, r#move: &Move<HexCoord>) -> Self {
        self.apply_move(r#move)
    }

    pub fn get_unflipped_matrix(&self) -> CCBoardMatrix {
        /*
        unflips the matrix if needed to make it intelligable for humans
         */
        if self.current_player == 1 {
            return self.matrix
        }
        self.flipped_matrix()
    }

    // Render colors (truecolor ANSI: R, G, B)
    const COLOR_EMPTY: (u8, u8, u8) = (150, 100, 200); // purple
    const COLOR_P1: (u8, u8, u8) = (255, 140, 50);     // orange
    const COLOR_P2: (u8, u8, u8) = (255, 105, 180);    // pink

    pub fn render(&self) {
        let matrix = self.get_unflipped_matrix();
        println!();
        for (i, row) in matrix[0..self.size as usize].iter().enumerate() {
            for _ in 0..i {
                print!(" ");
            }
            for val in row[0..self.size as usize].iter() {
                let (r, g, b, ch) = match val {
                    1 => (Self::COLOR_P1.0, Self::COLOR_P1.1, Self::COLOR_P1.2, "⬢"),
                    2 => (Self::COLOR_P2.0, Self::COLOR_P2.1, Self::COLOR_P2.2, "⬢"),
                    _ => (Self::COLOR_EMPTY.0, Self::COLOR_EMPTY.1, Self::COLOR_EMPTY.2, "⬡"),
                };
                print!("\x1b[38;2;{r};{g};{b}m{ch}\x1b[0m ");
            }
            println!();
        }
        println!();
        let (cr, cg, cb) = match self.current_player {
            1 => Self::COLOR_P1,
            _ => Self::COLOR_P2,
        };
        println!("current player: \x1b[38;2;{cr};{cg};{cb}m⬢\x1b[0m ({})", self.current_player);
    }

    pub fn get_info(&self) -> BoardInfo {
        let (wdl, winner, game_over) = self.compute_wdl_and_winner();
        BoardInfo {
            current_player: self.current_player,
            winner,
            wdl,
            size: self.size,
            duration: self.duration,
            game_over,
        }
    }
}
