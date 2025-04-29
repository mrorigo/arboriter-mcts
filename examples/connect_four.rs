//! Connect Four example for the MCTS algorithm
//!
//! This example demonstrates how to use the MCTS algorithm
//! to play Connect Four.

use std::fmt;
use std::io::{self, Write};

use arboriter_mcts::{
    policy::{backpropagation::StandardPolicy, selection::UCB1Policy, simulation::RandomPolicy},
    Action, GameState, MCTSConfig, MCTS,
};

const ROWS: usize = 6;
const COLS: usize = 7;

fn main() {
    // Initialize logging
    env_logger::init();

    println!("MCTS Connect Four Example");
    println!("=========================");
    println!();

    // Set up a new game
    let mut game = ConnectFour::new();

    // Create MCTS configuration
    let config = MCTSConfig::default()
        .with_exploration_constant(1.414)
        .with_max_iterations(20_000);

    // Main game loop
    while !game.is_terminal() {
        // Display the board
        println!("{}", game);

        if game.current_player == Player::Human {
            // Human player
            println!("Your move (enter column 0-6): ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();

            let column = match input.trim().parse::<usize>() {
                Ok(col) if col < COLS => col,
                _ => {
                    println!("Invalid column! Please enter a number between 0 and 6.");
                    continue;
                }
            };

            let action = Move { column };

            if !game.is_legal_move(&action) {
                println!("Column {} is full! Choose another column.", column);
                continue;
            }

            // Apply the human's move
            game = game.apply_action(&action);
        } else {
            // AI player
            println!("AI is thinking...");

            // Create a new MCTS search
            let mut mcts = MCTS::new(game.clone(), config.clone())
                .with_selection_policy(UCB1Policy::new(config.exploration_constant))
                .with_simulation_policy(RandomPolicy::new())
                .with_backpropagation_policy(StandardPolicy::new());

            // Find the best move
            match mcts.search() {
                Ok(action) => {
                    println!("AI chooses column: {}", action.column);

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
    match game.get_winner() {
        Some(Player::Human) => println!("You win!"),
        Some(Player::AI) => println!("AI wins!"),
        None => println!("The game is a draw!"),
    }
}

/// Players in Connect Four
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Player {
    Human,
    AI,
}

impl arboriter_mcts::Player for Player {}

/// Connect Four move
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Move {
    /// Column to drop the piece into (0-6)
    column: usize,
}

impl Action for Move {
    fn id(&self) -> usize {
        self.column
    }
}

/// Connect Four game state
#[derive(Clone)]
struct ConnectFour {
    /// Board representation (None = empty, Some(Player) = occupied)
    board: [[Option<Player>; COLS]; ROWS],

    /// Current player's turn
    current_player: Player,

    /// Last move played
    last_move: Option<Move>,
}

impl ConnectFour {
    /// Creates a new empty Connect Four board
    fn new() -> Self {
        ConnectFour {
            board: [[None; COLS]; ROWS],
            current_player: Player::Human,
            last_move: None,
        }
    }

    /// Checks if a move is legal
    fn is_legal_move(&self, action: &Move) -> bool {
        if action.column >= COLS {
            return false;
        }

        // Column is legal if the top cell is empty
        self.board[0][action.column].is_none()
    }

    /// Finds the row where a piece would land in a given column
    fn find_row(&self, column: usize) -> Option<usize> {
        for row in (0..ROWS).rev() {
            if self.board[row][column].is_none() {
                return Some(row);
            }
        }
        None
    }

    /// Returns the winner of the game, if any
    fn get_winner(&self) -> Option<Player> {
        // Helper to check for 4 in a row
        let check_line = |line: Vec<Option<Player>>| -> Option<Player> {
            for i in 0..line.len() - 3 {
                if let Some(player) = line[i] {
                    if line[i + 1] == Some(player)
                        && line[i + 2] == Some(player)
                        && line[i + 3] == Some(player)
                    {
                        return Some(player);
                    }
                }
            }
            None
        };

        // Check horizontal lines
        for row in 0..ROWS {
            let line: Vec<Option<Player>> = (0..COLS).map(|col| self.board[row][col]).collect();
            if let Some(winner) = check_line(line) {
                return Some(winner);
            }
        }

        // Check vertical lines
        for col in 0..COLS {
            let line: Vec<Option<Player>> = (0..ROWS).map(|row| self.board[row][col]).collect();
            if let Some(winner) = check_line(line) {
                return Some(winner);
            }
        }

        // Check diagonals (bottom-left to top-right)
        for row in 0..ROWS - 3 {
            for col in 0..COLS - 3 {
                let line = vec![
                    self.board[row + 3][col],
                    self.board[row + 2][col + 1],
                    self.board[row + 1][col + 2],
                    self.board[row][col + 3],
                ];
                if let Some(winner) = check_line(line) {
                    return Some(winner);
                }
            }
        }

        // Check diagonals (top-left to bottom-right)
        for row in 0..ROWS - 3 {
            for col in 0..COLS - 3 {
                let line = vec![
                    self.board[row][col],
                    self.board[row + 1][col + 1],
                    self.board[row + 2][col + 2],
                    self.board[row + 3][col + 3],
                ];
                if let Some(winner) = check_line(line) {
                    return Some(winner);
                }
            }
        }

        None
    }

    /// Check if the board is full
    fn is_board_full(&self) -> bool {
        for col in 0..COLS {
            if self.board[0][col].is_none() {
                return false;
            }
        }
        true
    }
}

impl GameState for ConnectFour {
    type Action = Move;
    type Player = Player;

    fn get_legal_actions(&self) -> Vec<Self::Action> {
        let mut actions = Vec::new();
        for col in 0..COLS {
            if self.is_legal_move(&Move { column: col }) {
                actions.push(Move { column: col });
            }
        }
        actions
    }

    fn apply_action(&self, action: &Self::Action) -> Self {
        let mut new_state = self.clone();

        // Find the row where the piece will land
        if let Some(row) = self.find_row(action.column) {
            // Make the move
            new_state.board[row][action.column] = Some(self.current_player);
            new_state.last_move = Some(*action);

            // Switch player
            new_state.current_player = match self.current_player {
                Player::Human => Player::AI,
                Player::AI => Player::Human,
            };
        }

        new_state
    }

    fn is_terminal(&self) -> bool {
        self.get_winner().is_some() || self.is_board_full()
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

impl fmt::Display for ConnectFour {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Column numbers
        write!(f, " ")?;
        for col in 0..COLS {
            write!(f, " {}", col)?;
        }
        writeln!(f)?;

        // Board
        for row in 0..ROWS {
            write!(f, "|")?;
            for col in 0..COLS {
                let symbol = match self.board[row][col] {
                    Some(Player::Human) => "X",
                    Some(Player::AI) => "O",
                    None => " ",
                };
                write!(f, "{}|", symbol)?;
            }
            writeln!(f)?;
        }

        // Bottom border
        write!(f, "+")?;
        for _ in 0..COLS {
            write!(f, "-+")?;
        }
        writeln!(f)?;

        writeln!(f, "\nPlayer {:?}'s turn", self.current_player)?;
        Ok(())
    }
}
