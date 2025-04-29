//! Policies for different phases of the MCTS algorithm
//!
//! This module contains implementations of various policies used in MCTS:
//! - Selection policies: How to choose which nodes to explore
//! - Simulation policies: How to play out games from a node
//! - Backpropagation policies: How to update node statistics
//! - Expansion policies: How to create new nodes

pub mod backpropagation;
pub mod selection;
pub mod simulation;

pub use backpropagation::{BackpropagationPolicy, StandardPolicy};
pub use selection::{SelectionPolicy, UCB1Policy};
pub use simulation::{RandomPolicy, SimulationPolicy};
