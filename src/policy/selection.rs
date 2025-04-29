//! Selection policies for the MCTS algorithm
//!
//! Selection policies determine which nodes to visit during the selection
//! phase of MCTS, balancing exploration and exploitation.

use std::f64;

use crate::{game_state::GameState, tree::MCTSNode};

/// Trait for policies that select nodes to explore
pub trait SelectionPolicy<S: GameState>: Send + Sync {
    /// Selects a child index based on the policy
    fn select_child(&self, node: &MCTSNode<S>) -> usize;

    /// Create a boxed clone of this policy
    fn clone_box(&self) -> Box<dyn SelectionPolicy<S>>;
}

/// Upper Confidence Bound 1 (UCB1) selection policy
///
/// This is the classic selection policy for MCTS, which balances
/// exploration and exploitation using the UCB1 formula:
///
/// ```text
/// UCB1 = average_reward + exploration_constant * sqrt(ln(parent_visits) / child_visits)
/// ```
///
/// Where:
/// - `average_reward` is the average reward from simulations through this node
/// - `exploration_constant` controls the balance between exploration and exploitation
/// - `parent_visits` is the number of visits to the parent node
/// - `child_visits` is the number of visits to the child node
///
/// Higher exploration constants favor exploration (trying less-visited nodes),
/// while lower values favor exploitation (choosing nodes with higher values).
///
/// The commonly used value for the exploration constant is sqrt(2) ≈ 1.414,
/// which is the default in this implementation.
#[derive(Debug, Clone)]
pub struct UCB1Policy {
    /// Exploration constant that controls the balance between exploration and exploitation.
    /// Higher values favor exploration of less-visited nodes.
    pub exploration_constant: f64,
}

impl UCB1Policy {
    /// Creates a new UCB1 policy with the given exploration constant
    pub fn new(exploration_constant: f64) -> Self {
        UCB1Policy {
            exploration_constant,
        }
    }

    /// Calculates the UCB1 value for a node
    pub fn ucb1_value(&self, child_value: f64, child_visits: u64, parent_visits: u64) -> f64 {
        if child_visits == 0 {
            return f64::INFINITY; // Always explore nodes that have never been visited
        }

        // UCB1 formula: value + C * sqrt(ln(parent_visits) / child_visits)
        let exploitation = child_value;
        let exploration =
            self.exploration_constant * ((parent_visits as f64).ln() / child_visits as f64).sqrt();

        exploitation + exploration
    }
}

impl<S: GameState> SelectionPolicy<S> for UCB1Policy {
    fn select_child(&self, node: &MCTSNode<S>) -> usize {
        if node.children.is_empty() {
            return 0;
        }

        let parent_visits = node.visits();
        let mut best_value = f64::NEG_INFINITY;
        let mut best_index = 0;

        for (i, child) in node.children.iter().enumerate() {
            let child_value = child.value();
            let child_visits = child.visits();

            let ucb_value = self.ucb1_value(child_value, child_visits, parent_visits);

            if ucb_value > best_value {
                best_value = ucb_value;
                best_index = i;
            }
        }

        best_index
    }

    fn clone_box(&self) -> Box<dyn SelectionPolicy<S>> {
        Box::new(self.clone())
    }
}

/// Upper Confidence Bound 1 Tuned (UCB1-Tuned) selection policy
///
/// An improved version of UCB1 that takes into account the variance
/// of the rewards.
#[derive(Debug, Clone)]
pub struct UCB1TunedPolicy {
    /// Exploration constant
    pub exploration_constant: f64,
}

impl UCB1TunedPolicy {
    /// Creates a new UCB1-Tuned policy
    pub fn new(exploration_constant: f64) -> Self {
        UCB1TunedPolicy {
            exploration_constant,
        }
    }
}

impl<S: GameState> SelectionPolicy<S> for UCB1TunedPolicy {
    fn select_child(&self, node: &MCTSNode<S>) -> usize {
        if node.children.is_empty() {
            return 0;
        }

        let parent_visits = node.visits();
        let mut best_value = f64::NEG_INFINITY;
        let mut best_index = 0;

        for (i, child) in node.children.iter().enumerate() {
            let child_value = child.value();
            let child_visits = child.visits();

            if child_visits == 0 {
                return i; // Always explore nodes that have never been visited
            }

            // UCB1-Tuned uses an upper bound on the variance
            // This is a simplified implementation
            let variance_bound = 0.25; // Maximum variance for rewards in [0,1]

            let exploration = self.exploration_constant
                * ((parent_visits as f64).ln() / child_visits as f64).sqrt()
                * f64::min(0.25, variance_bound);

            let ucb_value = child_value + exploration;

            if ucb_value > best_value {
                best_value = ucb_value;
                best_index = i;
            }
        }

        best_index
    }

    fn clone_box(&self) -> Box<dyn SelectionPolicy<S>> {
        Box::new(self.clone())
    }
}

/// Polynomial Upper Confidence Trees (PUCT) selection policy
///
/// This policy is used in AlphaZero and similar algorithms. It uses
/// a prior probability distribution over actions.
#[derive(Debug, Clone)]
pub struct PUCTPolicy {
    /// Exploration constant
    pub exploration_constant: f64,

    /// Prior probabilities for each action
    /// If none are provided, all actions are equally likely
    pub priors: Option<Vec<f64>>,
}

impl PUCTPolicy {
    /// Creates a new PUCT policy
    pub fn new(exploration_constant: f64) -> Self {
        PUCTPolicy {
            exploration_constant,
            priors: None,
        }
    }

    /// Creates a new PUCT policy with prior probabilities
    pub fn with_priors(exploration_constant: f64, priors: Vec<f64>) -> Self {
        PUCTPolicy {
            exploration_constant,
            priors: Some(priors),
        }
    }
}

impl<S: GameState> SelectionPolicy<S> for PUCTPolicy {
    fn select_child(&self, node: &MCTSNode<S>) -> usize {
        if node.children.is_empty() {
            return 0;
        }

        let parent_visits = node.visits();
        let mut best_value = f64::NEG_INFINITY;
        let mut best_index = 0;

        // Get priors or use uniform distribution
        let uniform_prior = vec![1.0];
        let priors = match &self.priors {
            Some(p) => p,
            None => {
                // Uniform distribution
                &uniform_prior
            }
        };

        for (i, child) in node.children.iter().enumerate() {
            let child_value = child.value();
            let child_visits = child.visits();

            if child_visits == 0 {
                return i; // Always explore nodes that have never been visited
            }

            // Get prior for this action (or 1.0 if using uniform)
            let prior = if i < priors.len() { priors[i] } else { 1.0 };

            // PUCT formula from AlphaZero: Q(s,a) + U(s,a)
            // where U(s,a) = c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
            let exploitation = child_value;
            let exploration = self.exploration_constant * prior * (parent_visits as f64).sqrt()
                / (1.0 + child_visits as f64);

            let puct_value = exploitation + exploration;

            if puct_value > best_value {
                best_value = puct_value;
                best_index = i;
            }
        }

        best_index
    }

    fn clone_box(&self) -> Box<dyn SelectionPolicy<S>> {
        Box::new(self.clone())
    }
}

impl Default for PUCTPolicy {
    fn default() -> Self {
        Self::new(1.414)
    }
}

// Implement SelectionPolicy for Box<dyn SelectionPolicy>
impl<S: GameState> SelectionPolicy<S> for Box<dyn SelectionPolicy<S>> {
    fn select_child(&self, node: &MCTSNode<S>) -> usize {
        (**self).select_child(node)
    }

    fn clone_box(&self) -> Box<dyn SelectionPolicy<S>> {
        (**self).clone_box()
    }
}
