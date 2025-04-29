//! Tree data structures for Monte Carlo Tree Search
//!
//! This module defines the tree representation used in MCTS, including
//! nodes, edges, and paths through the tree.

use rand::prelude::IteratorRandom;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::game_state::GameState;

/// Represents a node in the MCTS tree
///
/// Each node contains the game state, the action that led to it,
/// statistics about visits and rewards, and references to child nodes.
/// The tree is built incrementally during the search process.
pub struct MCTSNode<S: GameState> {
    /// The game state at this node
    pub state: S,

    /// The action that led to this state (None for root)
    pub action: Option<S::Action>,

    /// Number of times this node has been visited
    /// Uses atomic operations to support potential future parallelization
    pub visits: AtomicU64,

    /// Total reward accumulated from simulations through this node
    /// Uses atomic operations and fixed-point representation internally
    pub total_reward: AtomicU64,

    /// Children nodes representing states reachable from this one
    pub children: Vec<MCTSNode<S>>,

    /// Actions that have not yet been expanded into child nodes
    /// As the search progresses, actions are moved from this list to children
    pub unexpanded_actions: Vec<S::Action>,

    /// Depth of this node in the tree (root = 0)
    pub depth: usize,

    /// Player who made the move to reach this state
    /// For the root node, this is the starting player
    pub player: S::Player,
}

/// Internal representation of a fixed-point value for rewards
/// This allows atomic operations on floating point rewards
const REWARD_SCALE: f64 = 1_000_000.0;

/// Safely convert a floating point reward to a scaled integer
fn float_to_scaled_u64(value: f64) -> u64 {
    ((value * REWARD_SCALE).max(0.0) as u64).min(u64::MAX / 2)
}

/// Safely convert a scaled integer back to a floating point reward
fn scaled_u64_to_float(value: u64) -> f64 {
    value as f64 / REWARD_SCALE
}

impl<S: GameState> MCTSNode<S> {
    /// Creates a new node with the given state and action
    pub fn new(
        state: S,
        action: Option<S::Action>,
        parent_player: Option<S::Player>,
        depth: usize,
    ) -> Self {
        let player = parent_player.unwrap_or_else(|| state.get_current_player());
        let unexpanded_actions = state.get_legal_actions();

        MCTSNode {
            state,
            action,
            visits: AtomicU64::new(0),
            total_reward: AtomicU64::new(0),
            children: Vec::new(),
            unexpanded_actions,
            depth,
            player,
        }
    }

    /// Returns the number of visits to this node
    pub fn visits(&self) -> u64 {
        self.visits.load(Ordering::Relaxed)
    }

    /// Returns the total reward accumulated at this node
    pub fn total_reward(&self) -> f64 {
        scaled_u64_to_float(self.total_reward.load(Ordering::Relaxed))
    }

    /// Returns the average reward (value) of this node
    pub fn value(&self) -> f64 {
        let visits = self.visits();
        if visits == 0 {
            return 0.0;
        }
        self.total_reward() / visits as f64
    }

    /// Increments the visit count
    pub fn increment_visits(&self) {
        self.visits.fetch_add(1, Ordering::Relaxed);
    }

    /// Adds reward to the total
    pub fn add_reward(&self, reward: f64) {
        self.total_reward
            .fetch_add(float_to_scaled_u64(reward), Ordering::Relaxed);
    }

    /// Returns true if this node is fully expanded
    pub fn is_fully_expanded(&self) -> bool {
        self.unexpanded_actions.is_empty()
    }

    /// Returns true if this node is a leaf (has no children)
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Expands the node by creating a child for an unexpanded action
    ///
    /// This method takes an action from the unexpanded actions list,
    /// applies it to create a new game state, and creates a child node
    /// for this new state.
    ///
    /// # Arguments
    ///
    /// * `action_index` - Index into the `unexpanded_actions` list
    ///
    /// # Returns
    ///
    /// * `Some(&mut MCTSNode<S>)` - Reference to the newly created child node
    /// * `None` - If the action index is out of bounds
    ///
    /// # Note
    ///
    /// This method uses `swap_remove` on the unexpanded actions list, which
    /// changes the order of the remaining unexpanded actions. If order
    /// matters to your application, be aware of this side effect.
    pub fn expand(&mut self, action_index: usize) -> Option<&mut MCTSNode<S>> {
        if action_index >= self.unexpanded_actions.len() {
            return None;
        }

        let action = self.unexpanded_actions.swap_remove(action_index);
        let next_state = self.state.apply_action(&action);
        let current_player = self.state.get_current_player();

        let child = MCTSNode::new(
            next_state,
            Some(action),
            Some(current_player),
            self.depth + 1,
        );

        self.children.push(child);
        self.children.last_mut()
    }

    /// Expands a random unexpanded action
    pub fn expand_random(&mut self) -> Option<&mut MCTSNode<S>> {
        if self.unexpanded_actions.is_empty() {
            return None;
        }

        // Use IteratorRandom trait for choose method on range
        let mut rng = rand::thread_rng();
        let index = (0..self.unexpanded_actions.len()).choose(&mut rng).unwrap();

        self.expand(index)
    }
}

/// Represents a path through the MCTS tree
///
/// A path is a sequence of indices that can be used to navigate from
/// the root node to a specific node in the tree.
#[derive(Debug, Clone)]
pub struct NodePath {
    /// Indices of children to follow from the root
    pub indices: Vec<usize>,
}

impl NodePath {
    /// Creates a new empty path (pointing to the root)
    pub fn new() -> Self {
        NodePath {
            indices: Vec::new(),
        }
    }

    /// Creates a path with the given indices
    pub fn from_indices(indices: Vec<usize>) -> Self {
        NodePath { indices }
    }

    /// Extends the path with a new index
    pub fn push(&mut self, index: usize) {
        self.indices.push(index);
    }

    /// Returns the length of the path
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns true if the path is empty
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

impl Default for NodePath {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for NodePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Path[")?;
        for (i, idx) in self.indices.iter().enumerate() {
            if i > 0 {
                write!(f, " -> ")?;
            }
            write!(f, "{}", idx)?;
        }
        write!(f, "]")
    }
}
