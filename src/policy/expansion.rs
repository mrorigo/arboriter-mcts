//! Expansion policies determine which unexpanded action to choose
//! when expanding a leaf node.

use crate::{game_state::GameState, tree::MCTSNode};
use rand::prelude::IteratorRandom;

/// Trait for policies that select which action to expand
pub trait ExpansionPolicy<S: GameState>: Send + Sync {
    /// Selects an action to expand from the node's unexpanded actions.
    /// Returns the index of the action in `node.unexpanded_actions` and the prior probability to assign.
    fn select_action_to_expand(&self, node: &MCTSNode<S>) -> Option<(usize, f64)>;

    /// Create a boxed clone of this policy
    fn clone_box(&self) -> Box<dyn ExpansionPolicy<S>>;
}

/// Random expansion policy
///
/// Selects an unexpanded action uniformly at random.
/// Assigns a uniform prior (1/N).
#[derive(Debug, Clone)]
pub struct RandomExpansionPolicy;

impl RandomExpansionPolicy {
    /// Creates a new random expansion policy
    pub fn new() -> Self {
        RandomExpansionPolicy
    }
}

impl Default for RandomExpansionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: GameState> ExpansionPolicy<S> for RandomExpansionPolicy {
    fn select_action_to_expand(&self, node: &MCTSNode<S>) -> Option<(usize, f64)> {
        if node.unexpanded_actions.is_empty() {
            return None;
        }

        let mut rng = rand::thread_rng();
        let index = (0..node.unexpanded_actions.len()).choose(&mut rng)?;

        // Calculate uniform prior: 1.0 / number of legal actions (at creation time)
        // Note: unexpanded_actions shrinks, so we need total actions.
        // But MCTSNode doesn't store total action count, only children + unexpanded.
        let total_actions = node.children.len() + node.unexpanded_actions.len();
        let prior = if total_actions > 0 {
            1.0 / total_actions as f64
        } else {
            1.0 // Should not happen if unexpanded is not empty
        };

        Some((index, prior))
    }

    fn clone_box(&self) -> Box<dyn ExpansionPolicy<S>> {
        Box::new(self.clone())
    }
}

// Implement ExpansionPolicy for Box<dyn ExpansionPolicy>
impl<S: GameState> ExpansionPolicy<S> for Box<dyn ExpansionPolicy<S>> {
    fn select_action_to_expand(&self, node: &MCTSNode<S>) -> Option<(usize, f64)> {
        (**self).select_action_to_expand(node)
    }

    fn clone_box(&self) -> Box<dyn ExpansionPolicy<S>> {
        (**self).clone_box()
    }
}
