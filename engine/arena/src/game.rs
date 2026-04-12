use alpha_cc_core::{Board, cc::CCBoard, cc::HexCoord, Move};

pub struct GameState {
    history: Vec<CCBoard>,
    moves: Vec<MoveRecord>,
    board_size: u8,
}

#[allow(dead_code)]
pub struct MoveRecord {
    pub action_index: usize,
    pub mv: Move<HexCoord>,
}

impl GameState {
    pub fn new(board_size: u8) -> Self {
        Self {
            history: vec![CCBoard::create(board_size as usize)],
            moves: Vec::new(),
            board_size,
        }
    }

    pub fn current_board(&self) -> &CCBoard {
        self.history.last().unwrap()
    }

    pub fn board_at(&self, index: usize) -> &CCBoard {
        &self.history[index]
    }

    pub fn apply_move(&mut self, action_index: usize) {
        let board = self.current_board();
        let moves = board.legal_moves();
        let mv = moves[action_index].clone();
        let new_board = board.apply(&mv);
        self.moves.push(MoveRecord { action_index, mv });
        self.history.push(new_board);
    }

    pub fn len(&self) -> usize {
        self.history.len()
    }

    pub fn ply(&self) -> usize {
        self.moves.len()
    }

    pub fn move_records(&self) -> &[MoveRecord] {
        &self.moves
    }

    pub fn is_game_over(&self) -> bool {
        self.current_board().get_info().game_over
    }

    #[allow(dead_code)]
    pub fn winner(&self) -> i8 {
        self.current_board().get_info().winner
    }

    #[allow(dead_code)]
    pub fn board_size(&self) -> u8 {
        self.board_size
    }

    pub fn reset(&mut self) {
        self.history.truncate(1);
        self.history[0] = CCBoard::create(self.board_size as usize);
        self.moves.clear();
    }
}
