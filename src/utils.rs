//! Utility functions for the MCTS algorithm
//!
//! This module contains various helper functions and utilities used
//! throughout the MCTS implementation.

/// Calculates the exploitation term for UCB1
///
/// This is simply the average reward for a node.
pub fn exploitation_term(total_reward: f64, visits: u64) -> f64 {
    if visits == 0 {
        return 0.0;
    }
    total_reward / visits as f64
}

/// Calculates the exploration term for UCB1
///
/// This is the term that encourages exploration of less-visited nodes.
pub fn exploration_term(parent_visits: u64, child_visits: u64, exploration_constant: f64) -> f64 {
    if child_visits == 0 {
        return f64::INFINITY;
    }

    exploration_constant * ((parent_visits as f64).ln() / child_visits as f64).sqrt()
}

/// Calculates the UCB1 value for a node
///
/// UCB1 balances exploitation (using known good nodes) with exploration
/// (trying less-visited nodes that might be better).
pub fn ucb1_value(
    total_reward: f64,
    visits: u64,
    parent_visits: u64,
    exploration_constant: f64,
) -> f64 {
    if visits == 0 {
        return f64::INFINITY;
    }

    let exploitation = exploitation_term(total_reward, visits);
    let exploration = exploration_term(parent_visits, visits, exploration_constant);

    exploitation + exploration
}

/// Safely calculates the win rate from wins and visits
///
/// Returns 0.0 if no visits have occurred.
pub fn win_rate(wins: u64, visits: u64) -> f64 {
    if visits == 0 {
        return 0.0;
    }
    wins as f64 / visits as f64
}
