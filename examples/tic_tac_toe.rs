//! Tic-Tac-Toe example for the MCTS algorithm
//!
//! This example demonstrates how to use the MCTS algorithm
//! to play Tic-Tac-Toe.

use std::fmt;
use std::io::{self, Write};

use arboriter_mcts::{
    policy::{selection::UCB1Policy, simulation::RandomPolicy},
    Action, GameState, MCTSConfig, MCTS,
};

fn main() {
    // Initialize logging
    env_logger::init();

    println!("MCTS Tic-Tac-Toe Example");
    println!("========================");
    println!();

    // Set up a new game
    let mut game = TicTacToe::new();

    // Create MCTS configuration
    let config = MCTSConfig::default()
        .with_exploration_constant(1.414)
        .with_max_iterations(10_000);

    // Main game loop
    while !game.is_terminal() {
        // Display the board
        println!("{}", game);

        if game.current_player == Player::X {
            // Human player (X)
            println!("Your move (enter row column, e.g. '1 2'): ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();

            let coords: Vec<usize> = input
                .trim()
                .split_whitespace()
                .filter_map(|s| s.parse::<usize>().ok())
                .collect();

            if coords.len() != 2 || coords[0] > 2 || coords[1] > 2 {
                println!("Invalid move! Enter row and column (0-2).");
                continue;
            }

            let row = coords[0];
            let col = coords[1];

            let move_index = row * 3 + col;
            let action = Move { index: move_index };

            if !game.is_legal_move(&action) {
                println!("Illegal move! Try again.");
                continue;
            }

            // Apply the human's move
            game = game.apply_action(&action);
        } else {
            // AI player (O)
            println!("AI is thinking...");

            // Create a new MCTS search
            let mut mcts = MCTS::new(game.clone(), config.clone())
                .with_selection_policy(UCB1Policy::new(config.exploration_constant))
                .with_simulation_policy(RandomPolicy::new());

            // Find the best move
            match mcts.search() {
                Ok(action) => {
                    println!(
                        "AI chooses: {} (row {}, col {})",
                        action.index,
                        action.index / 3,
                        action.index % 3
                    );

                    // Apply the AI's move
                    game = game.apply_action(&action);

                    // Show stats
                    println!("{}", mcts.get_statistics().summary());
                }
                Err(e) => {
                    println!("Error: {:?}", e);
                    break;
                }
            }
        }
    }

    // Display final state
    println!("{}", game);

    // Report the result
    if let Some(winner) = game.get_winner() {
        println!("Player {:?} wins!", winner);
    } else {
        println!("The game is a draw!");
    }
}

/// Players in Tic-Tac-Toe
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Player {
    X,
    O,
}

impl arboriter_mcts::Player for Player {}

/// Tic-Tac-Toe move
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Move {
    /// Board position index (0-8)
    index: usize,
}

impl Action for Move {
    fn id(&self) -> usize {
        self.index
    }
}

/// Tic-Tac-Toe game state
#[derive(Clone)]
struct TicTacToe {
    /// Board representation (None = empty, Some(Player) = occupied)
    board: [Option<Player>; 9],

    /// Current player's turn
    current_player: Player,

    /// Number of moves played so far
    moves_played: usize,
}

impl TicTacToe {
    /// Creates a new empty Tic-Tac-Toe board
    fn new() -> Self {
        TicTacToe {
            board: [None; 9],
            current_player: Player::X,
            moves_played: 0,
        }
    }

    /// Checks if a move is legal
    fn is_legal_move(&self, action: &Move) -> bool {
        if action.index >= 9 {
            return false;
        }
        self.board[action.index].is_none()
    }

    /// Returns the winner of the game, if any
    fn get_winner(&self) -> Option<Player> {
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
    type Action = Move;
    type Player = Player;

    fn get_legal_actions(&self) -> Vec<Self::Action> {
        let mut actions = Vec::new();
        for i in 0..9 {
            if self.board[i].is_none() {
                actions.push(Move { index: i });
            }
        }
        actions
    }

    fn apply_action(&self, action: &Self::Action) -> Self {
        let mut new_state = self.clone();

        // Make the move
        new_state.board[action.index] = Some(self.current_player);
        new_state.moves_played = self.moves_played + 1;

        // Switch player
        new_state.current_player = match self.current_player {
            Player::X => Player::O,
            Player::O => Player::X,
        };

        new_state
    }

    fn is_terminal(&self) -> bool {
        self.get_winner().is_some() || self.moves_played == 9
    }

    fn get_result(&self, for_player: &Self::Player) -> f64 {
        if let Some(winner) = self.get_winner() {
            if winner == *for_player {
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

impl fmt::Display for TicTacToe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  0 1 2")?;
        for row in 0..3 {
            write!(f, "{} ", row)?;
            for col in 0..3 {
                let index = row * 3 + col;
                let symbol = match self.board[index] {
                    Some(Player::X) => "X",
                    Some(Player::O) => "O",
                    None => ".",
                };
                write!(f, "{} ", symbol)?;
            }
            writeln!(f)?;
        }

        writeln!(f, "\nPlayer {:?}'s turn", self.current_player)?;
        Ok(())
    }
}
