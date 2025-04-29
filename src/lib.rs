//! # arboriter-mcts
//!
//! A Monte Carlo Tree Search (MCTS) implementation built on top of the arboriter tree traversal primitive.
//!
//! This crate provides a flexible and efficient implementation of the MCTS algorithm, which can be
//! used for decision making in games, planning problems, and other domains where a tree of possible
//! futures can be explored.
//!
//! ## Features
//!
//! - Generic implementation that works with any game or decision process
//! - Multiple selection policies (UCB1, UCB1-Tuned, PUCT)
//! - Customizable simulation strategies
//! - Detailed search statistics and visualization
//! - Built on arboriter's tree traversal primitives for elegant implementation
//!
//! ## Basic Usage
//!
//! ```
//! // This example demonstrates how to use arboriter-mcts
//! // with a simple game implementation
//! use arboriter_mcts::{MCTS, MCTSConfig, GameState};
//!
//! // First, define our action and player types
//! #[derive(Clone, Debug, PartialEq, Eq)]
//! struct MyAction(u8);
//!
//! impl arboriter_mcts::Action for MyAction {
//!     fn id(&self) -> usize { self.0 as usize }
//! }
//!
//! #[derive(Clone, Debug, PartialEq, Eq)]
//! struct MyPlayer(u8);
//!
//! impl arboriter_mcts::Player for MyPlayer {}
//!
//! // A very simple game where the state reaches terminal after one move
//! #[derive(Clone)]
//! struct MyGame {
//!     player_turn: MyPlayer,
//!     moves_made: u8,
//!     max_moves: u8, // Game ends after this many moves
//! }
//!
//! impl GameState for MyGame {
//!     type Action = MyAction;
//!     type Player = MyPlayer;
//!     
//!     fn get_legal_actions(&self) -> Vec<Self::Action> {
//!         if self.is_terminal() {
//!             return vec![];
//!         }
//!         // Just return a single possible action for simplicity
//!         vec![MyAction(0)]
//!     }
//!     
//!     fn apply_action(&self, _action: &Self::Action) -> Self {
//!         let mut new_state = self.clone();
//!         new_state.moves_made += 1;
//!         new_state.player_turn = MyPlayer(1 - self.player_turn.0); // Switch players
//!         new_state
//!     }
//!     
//!     fn is_terminal(&self) -> bool {
//!         self.moves_made >= self.max_moves
//!     }
//!     
//!     fn get_result(&self, for_player: &Self::Player) -> f64 {
//!         // Simple draw result for testing
//!         0.5
//!     }
//!     
//!     fn get_current_player(&self) -> Self::Player {
//!         self.player_turn.clone()
//!     }
//! }
//!
//! fn main() -> Result<(), arboriter_mcts::MCTSError> {
//!     // Create initial game state
//!     let initial_state = MyGame { 
//!         player_turn: MyPlayer(0),
//!         moves_made: 0,
//!         max_moves: 1, // Game ends after just one move
//!     };
//!
//!     // Create a configuration with very few iterations for quick testing
//!     let config = MCTSConfig::default()
//!         .with_exploration_constant(1.414)
//!         .with_max_iterations(10);
//!
//!     // Create the MCTS searcher with initial state
//!     let mut mcts = MCTS::new(initial_state, config);
//!
//!     // Find the best action
//!     let best_action = mcts.search()?;
//!
//!     // Get search statistics
//!     println!("Search statistics: {}", mcts.get_statistics().summary());
//!
//!     // Use the action in your game
//!     println!("Best action: {:?}", best_action);
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## How It Works
//!
//! MCTS consists of four main phases:
//!
//! 1. **Selection**: Starting from the root, select successive child nodes down to a leaf node.
//!    This phase uses a selection policy (like UCB1) to balance exploration and exploitation.
//!
//! 2. **Expansion**: If the leaf node is not terminal and has untried actions, create one or more
//!    child nodes by applying those actions.
//!
//! 3. **Simulation**: From the new node, simulate a game to completion using a default policy.
//!    This often involves random play, but can use domain-specific heuristics.
//!
//! 4. **Backpropagation**: Update the statistics (visit counts and rewards) for all nodes
//!    in the path from the selected node to the root.
//!
//! This process is repeated many times, gradually building a tree of game states and improving
//! the value estimates for each action.
//!
//! ## Customizing Policies
//!
//! You can customize each aspect of the MCTS algorithm by providing different policies:
//!
//! ```rust
//! use arboriter_mcts::{MCTS, MCTSConfig, GameState};
//! 
//! // Example with minimal configuration for customizing policies
//! 
//! // Define the types and structures needed
//! #[derive(Clone, Debug, PartialEq, Eq)]
//! struct MyPlayer(u8);
//! 
//! impl arboriter_mcts::Player for MyPlayer {}
//! 
//! #[derive(Clone, Debug, PartialEq, Eq)]
//! struct MyAction(u8);
//! 
//! impl arboriter_mcts::Action for MyAction {
//!     fn id(&self) -> usize { self.0 as usize }
//! }
//! 
//! // Create a simple game with a terminal state
//! #[derive(Clone)]
//! struct MyGame {
//!     player_turn: MyPlayer,
//!     is_over: bool,
//! }
//! 
//! impl GameState for MyGame {
//!     type Action = MyAction;
//!     type Player = MyPlayer;
//!     
//!     fn get_legal_actions(&self) -> Vec<Self::Action> {
//!         if self.is_over { return vec![]; }
//!         vec![MyAction(0)]
//!     }
//!     
//!     fn apply_action(&self, _: &Self::Action) -> Self {
//!         // Create a terminal state to ensure search completes quickly
//!         Self {
//!             player_turn: self.player_turn.clone(),
//!             is_over: true, // Make sure we reach a terminal state
//!         }
//!     }
//!     
//!     fn is_terminal(&self) -> bool { self.is_over }
//!     
//!     fn get_result(&self, _: &Self::Player) -> f64 { 0.5 }
//!     
//!     fn get_current_player(&self) -> Self::Player { self.player_turn.clone() }
//! }
//! 
//! // Import the policy types
//! use arboriter_mcts::policy::{
//!     selection::UCB1Policy,
//!     simulation::RandomPolicy,
//!     backpropagation::StandardPolicy,
//! };
//! 
//! fn main() -> Result<(), arboriter_mcts::MCTSError> {
//!     // Create a terminal state game to ensure fast completion
//!     let initial_state = MyGame {
//!         player_turn: MyPlayer(0),
//!         is_over: false,
//!     };
//!     
//!     // Configure with very few iterations
//!     let config = MCTSConfig::default()
//!         .with_max_iterations(10); // Only do a few iterations for the doctest
//!     
//!     // Customize policies
//!     let mut mcts = MCTS::new(initial_state, config)
//!         .with_selection_policy(UCB1Policy::new(1.5))
//!         .with_simulation_policy(RandomPolicy::new())
//!         .with_backpropagation_policy(StandardPolicy::new());
//!     
//!     // Run a quick search
//!     let result = mcts.search();
//!     println!("Search completed with result: {:?}", result);
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Examples
//!
//! The crate includes complete examples for:
//!
//! - Tic-Tac-Toe: A simple 3x3 game
//! - Connect Four: A more complex 7x6 game
//!
//! To run the examples:
//!
//! ```bash
//! cargo run --example tic_tac_toe
//! cargo run --example connect_four
//! ```

pub mod config;
pub mod game_state;
pub mod mcts;
pub mod policy;
pub mod stats;
pub mod tree;
pub mod utils;

pub use config::MCTSConfig;
pub use game_state::{Action, GameState, Player};
pub use mcts::MCTS;
pub use policy::{BackpropagationPolicy, SelectionPolicy, SimulationPolicy};
pub use stats::SearchStatistics;
pub use tree::{MCTSNode, NodePath};

/// Error types for the MCTS algorithm
#[derive(thiserror::Error, Debug)]
pub enum MCTSError {
    /// No legal actions are available from the current state
    #[error("No legal actions available from current state")]
    NoLegalActions,

    /// Search was stopped before completion
    #[error("Search stopped: {0}")]
    SearchStopped(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
}

/// Result type for MCTS operations
pub type Result<T> = std::result::Result<T, MCTSError>;
