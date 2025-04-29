use arboriter_mcts::{
    config::BestChildCriteria,
    policy::{selection::UCB1Policy, simulation::RandomPolicy},
    Action, GameState, MCTSConfig, Player, MCTS,
};

// Simple tic-tac-toe implementation for testing
#[derive(Clone, Debug)]
struct TicTacToe {
    board: [Option<TicTacPlayer>; 9],
    current_player: TicTacPlayer,
    moves_played: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Copy)]
enum TicTacPlayer {
    X,
    O,
}

impl Player for TicTacPlayer {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TicTacMove {
    position: usize,
}

impl Action for TicTacMove {
    fn id(&self) -> usize {
        self.position
    }
}

impl TicTacToe {
    fn new() -> Self {
        TicTacToe {
            board: [None; 9],
            current_player: TicTacPlayer::X,
            moves_played: 0,
        }
    }

    fn get_winner(&self) -> Option<TicTacPlayer> {
        // Check rows
        for row in 0..3 {
            let i = row * 3;
            if self.board[i].is_some()
                && self.board[i] == self.board[i + 1]
                && self.board[i] == self.board[i + 2]
            {
                return self.board[i];
            }
        }

        // Check columns
        for col in 0..3 {
            if self.board[col].is_some()
                && self.board[col] == self.board[col + 3]
                && self.board[col] == self.board[col + 6]
            {
                return self.board[col];
            }
        }

        // Check diagonals
        if self.board[0].is_some()
            && self.board[0] == self.board[4]
            && self.board[0] == self.board[8]
        {
            return self.board[0];
        }
        if self.board[2].is_some()
            && self.board[2] == self.board[4]
            && self.board[2] == self.board[6]
        {
            return self.board[2];
        }

        None
    }
}

impl GameState for TicTacToe {
    type Action = TicTacMove;
    type Player = TicTacPlayer;

    fn get_legal_actions(&self) -> Vec<Self::Action> {
        let mut actions = Vec::new();
        for i in 0..9 {
            if self.board[i].is_none() {
                actions.push(TicTacMove { position: i });
            }
        }
        actions
    }

    fn apply_action(&self, action: &Self::Action) -> Self {
        let mut new_state = self.clone();

        // Make the move
        new_state.board[action.position] = Some(self.current_player);
        new_state.moves_played = self.moves_played + 1;

        // Switch player
        new_state.current_player = match self.current_player {
            TicTacPlayer::X => TicTacPlayer::O,
            TicTacPlayer::O => TicTacPlayer::X,
        };

        new_state
    }

    fn is_terminal(&self) -> bool {
        self.get_winner().is_some() || self.moves_played == 9
    }

    fn get_result(&self, for_player: &Self::Player) -> f64 {
        if let Some(winner) = self.get_winner() {
            if &winner == for_player {
                return 1.0; // Win
            } else {
                return 0.0; // Loss
            }
        }

        // Draw
        0.5
    }

    fn get_current_player(&self) -> Self::Player {
        self.current_player
    }
}

// Test helper
fn create_specific_board() -> TicTacToe {
    // Create a board where X has a winning move
    // X O X
    // - X -
    // O - O
    let mut game = TicTacToe::new();
    game.board[0] = Some(TicTacPlayer::X);
    game.board[1] = Some(TicTacPlayer::O);
    game.board[2] = Some(TicTacPlayer::X);
    game.board[4] = Some(TicTacPlayer::X);
    game.board[6] = Some(TicTacPlayer::O);
    game.board[8] = Some(TicTacPlayer::O);
    game.moves_played = 6;
    game.current_player = TicTacPlayer::X;
    game
}

#[test]
fn test_mcts_basic_functionality() {
    let initial_state = TicTacToe::new();

    let config = MCTSConfig::default()
        .with_exploration_constant(1.414)
        .with_max_iterations(100);

    let mut mcts = MCTS::new(initial_state, config);

    // Make sure we can run a search without errors
    let result = mcts.search();
    assert!(result.is_ok(), "MCTS search should succeed");

    let action = result.unwrap();
    assert!(action.position < 9, "Action should be valid");

    // Verify statistics
    let stats = mcts.get_statistics();
    assert_eq!(
        stats.iterations, 100,
        "Should have performed expected iterations"
    );
    assert!(stats.tree_size > 1, "Tree should have grown");
}

#[test]
fn test_mcts_finds_winning_move() {
    let game = create_specific_board();

    // In this position, X can win by playing at position 7
    let winning_position = 7;

    let config = MCTSConfig::default()
        .with_exploration_constant(0.5) // Favor exploitation
        .with_max_iterations(1000)
        .with_best_child_criteria(BestChildCriteria::MostVisits);

    let mut mcts = MCTS::new(game, config);

    // MCTS should find the winning move
    let result = mcts.search().unwrap();
    assert_eq!(
        result.position, winning_position,
        "MCTS should find the winning move"
    );
}

#[test]
fn test_mcts_selection_policy_customization() {
    let game = TicTacToe::new();

    let config = MCTSConfig::default().with_max_iterations(100);

    let mut mcts = MCTS::new(game, config)
        .with_selection_policy(UCB1Policy::new(0.1)) // Very exploitative
        .with_simulation_policy(RandomPolicy::new());

    // Just ensure the search runs without errors
    let result = mcts.search();
    assert!(result.is_ok());
}

#[test]
fn test_empty_game_state() {
    // Create a game state with no legal moves
    let mut game = TicTacToe::new();
    for i in 0..9 {
        game.board[i] = Some(if i % 2 == 0 {
            TicTacPlayer::X
        } else {
            TicTacPlayer::O
        });
    }
    game.moves_played = 9;

    let config = MCTSConfig::default();
    let mut mcts = MCTS::new(game, config);

    // Search should error with NoLegalActions
    let result = mcts.search();
    assert!(
        result.is_err(),
        "Search on a full board should return an error"
    );

    match result {
        Err(e) => assert!(
            format!("{}", e).contains("No legal actions"),
            "Error should indicate no legal actions"
        ),
        _ => panic!("Unexpected result"),
    }
}
