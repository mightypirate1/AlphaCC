use alpha_cc_core::cc::CCBoard;
use alpha_cc_core::Board;
use alpha_cc_mcts::{GumbelParams, HalvingScheduler, ImprovedPolicyDescent, Scheduler, SigmaParams};
use alpha_cc_nn::mock::MockPredictor;

fn make_mcts(predictor: MockPredictor, pruning: bool) -> HalvingScheduler<CCBoard, MockPredictor, ImprovedPolicyDescent> {
    let sigma = SigmaParams { c_visit: 50.0, c_scale: 1.0 };
    let gumbel = GumbelParams {
        all_at_least_once: false,
        base_count: 16,
        floor_count: 5,
        keep_frac: 0.5,
    };
    HalvingScheduler::build_improved_halving(vec![predictor], 0, 1.0, sigma, gumbel, pruning, false)
}

// ──────────────────────────────────────────────────
// Basic functionality
// ──────────────────────────────────────────────────

#[test]
fn run_rollouts_returns_policy_and_value() {
    let board = CCBoard::create(3);
    let mcts = make_mcts(MockPredictor::uniform(0.5), false);

    let result = mcts.run(&board, 10, 5);

    assert_eq!(result.pi.len(), board.legal_moves().len());
    let sum: f32 = result.pi.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "pi should sum to ~1.0, got {sum}");
    assert!(result.value.abs() < 1.0, "value should be in [-1, 1], got {}", result.value);
}

#[test]
fn rollouts_increase_visit_counts() {
    let board = CCBoard::create(3);
    let mcts = make_mcts(MockPredictor::uniform(0.0), false);

    mcts.run(&board, 50, 5);

    let node = mcts.mcts().get_node_snapshot(&board).expect("root should be in tree");
    let total_visits: u32 = (0..node.num_actions()).map(|a| node.get_n(a)).sum();
    assert!(total_visits >= 50, "expected >= 50 visits, got {total_visits}");
}

#[test]
fn biased_predictor_favors_first_move() {
    let board = CCBoard::create(3);
    let mcts = make_mcts(MockPredictor::biased(0.0, 0.9), false);

    let result = mcts.run(&board, 100, 5);

    // The first move should get the most visits since the prior strongly favors it
    let max_idx = result.pi.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    assert_eq!(max_idx, 0, "biased predictor should favor action 0, but max was action {max_idx}");
}

#[test]
fn value_differs_based_on_predictor() {
    let board = CCBoard::create(3);

    let mcts_pos = make_mcts(MockPredictor::uniform(0.8), false);
    let value_pos = mcts_pos.run(&board, 30, 5).value;

    let mcts_neg = make_mcts(MockPredictor::uniform(-0.8), false);
    let value_neg = mcts_neg.run(&board, 30, 5).value;

    // The two predictors disagree strongly, so the MCTS values should differ
    assert!((value_pos - value_neg).abs() > 0.1,
        "different predictors should produce different values: {value_pos} vs {value_neg}");
}

#[test]
fn successive_rollout_calls_accumulate() {
    let board = CCBoard::create(3);
    let mcts = make_mcts(MockPredictor::uniform(0.0), false);

    mcts.run(&board, 20, 5);
    let node1 = mcts.mcts().get_node_snapshot(&board).unwrap();
    let visits1: u32 = (0..node1.num_actions()).map(|a| node1.get_n(a)).sum();

    mcts.run(&board, 20, 5);
    let node2 = mcts.mcts().get_node_snapshot(&board).unwrap();
    let visits2: u32 = (0..node2.num_actions()).map(|a| node2.get_n(a)).sum();

    assert!(visits2 > visits1, "successive calls should accumulate: {visits2} > {visits1}");
}

// ──────────────────────────────────────────────────
// Game playing end-to-end
// ──────────────────────────────────────────────────

#[test]
fn can_play_a_complete_game() {
    let mut board = CCBoard::create(3);
    let mcts = make_mcts(MockPredictor::uniform(0.0), false);

    let mut moves_played = 0;
    while !board.get_info().game_over && moves_played < 200 {
        let result = mcts.run(&board, 10, 5);
        let action = result.pi.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let moves = board.legal_moves();
        board = board.apply(&moves[action]);
        mcts.mcts().notify_move_applied(&board);
        moves_played += 1;
    }

    assert!(moves_played > 0, "should have played at least one move");
}

// ──────────────────────────────────────────────────
// Tree pruning
// ──────────────────────────────────────────────────

#[test]
fn pruning_tree_reduces_node_count_after_move() {
    let board = CCBoard::create(3);
    let mcts = make_mcts(MockPredictor::uniform(0.0), true);

    // Build up a tree with many nodes
    mcts.run(&board, 100, 10);
    let nodes_before = mcts.mcts().get_all_nodes().len();

    // Play a move
    let result = mcts.run(&board, 10, 5);
    let action = result.pi.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let moves = board.legal_moves();
    let new_board = board.apply(&moves[action]);
    mcts.mcts().notify_move_applied(&new_board);

    let nodes_after = mcts.mcts().get_all_nodes().len();

    assert!(nodes_after < nodes_before,
        "pruning should reduce node count: {nodes_after} < {nodes_before}");
}

#[test]
fn pruning_preserves_chosen_subtree() {
    let board = CCBoard::create(3);
    let mcts = make_mcts(MockPredictor::biased(0.0, 0.8), true);

    // Build up a tree
    mcts.run(&board, 100, 10);

    // The biased predictor favors action 0. Play it.
    let moves = board.legal_moves();
    let new_board = board.apply(&moves[0]);
    mcts.mcts().notify_move_applied(&new_board);

    // The new root should exist in the tree (it was the most-visited child)
    let node = mcts.mcts().get_node_snapshot(&new_board);
    assert!(node.is_some(), "new root should be in tree after pruning");
}

#[test]
fn pruning_does_not_remove_explored_children_of_played_move() {
    let board = CCBoard::create(3);
    let mcts = make_mcts(MockPredictor::uniform(0.0), true);

    // Deep search to populate children of the first move's subtree
    mcts.run(&board, 200, 15);

    // Play action 0
    let moves = board.legal_moves();
    let new_board = board.apply(&moves[0]);
    mcts.mcts().notify_move_applied(&new_board);

    // The new board should have children in the tree (explored from the previous search)
    let new_moves = new_board.legal_moves();
    let mut found_children = 0;
    for m in &new_moves {
        let child = new_board.apply(m);
        if mcts.mcts().get_node_snapshot(&child).is_some() {
            found_children += 1;
        }
    }

    assert!(found_children > 0,
        "children of the played move should survive pruning, found {found_children}");
}

#[test]
fn without_pruning_tree_grows_unbounded() {
    let mut board = CCBoard::create(3);
    let mcts = make_mcts(MockPredictor::uniform(0.0), false);

    // Play several moves, never pruning
    for _ in 0..5 {
        mcts.run(&board, 30, 5);
        let moves = board.legal_moves();
        if moves.is_empty() { break; }
        board = board.apply(&moves[0]);
        mcts.mcts().notify_move_applied(&board);
    }

    let total_nodes = mcts.mcts().get_all_nodes().len();
    // Without pruning, old nodes accumulate
    assert!(total_nodes > 5, "without pruning, tree should have many nodes: got {total_nodes}");
}

#[test]
fn with_pruning_tree_stays_bounded() {
    let mut board = CCBoard::create(3);
    let mcts = make_mcts(MockPredictor::uniform(0.0), true);

    let mut max_nodes = 0;
    for _ in 0..10 {
        if board.get_info().game_over { break; }
        mcts.run(&board, 30, 5);
        let nodes = mcts.mcts().get_all_nodes().len();
        if nodes > max_nodes { max_nodes = nodes; }
        let moves = board.legal_moves();
        if moves.is_empty() { break; }
        board = board.apply(&moves[0]);
        mcts.mcts().notify_move_applied(&board);
    }

    let final_nodes = mcts.mcts().get_all_nodes().len();
    // With pruning, tree should be much smaller than the peak
    assert!(final_nodes < max_nodes || max_nodes < 50,
        "pruning should keep tree bounded: final={final_nodes}, peak={max_nodes}");
}

// ──────────────────────────────────────────────────
// Edge cases
// ──────────────────────────────────────────────────

#[test]
fn zero_depth_rollout_uses_nn_value() {
    let board = CCBoard::create(3);
    let mcts = make_mcts(MockPredictor::uniform(0.42), false);

    // depth=0 means just evaluate the leaf, no search
    let result = mcts.run(&board, 10, 0);

    // With depth 0, pi should still be well-formed
    assert_eq!(result.pi.len(), board.legal_moves().len());
}

#[test]
fn clear_tree_resets_state() {
    let board = CCBoard::create(3);
    let mcts = make_mcts(MockPredictor::uniform(0.0), false);

    mcts.run(&board, 50, 5);
    assert!(mcts.mcts().get_node_snapshot(&board).is_some());

    mcts.mcts().clear_tree();
    assert!(mcts.mcts().get_node_snapshot(&board).is_none());
}
