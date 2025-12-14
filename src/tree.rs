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

    /// Sum of squared rewards (for variance calculation in UCB1-Tuned)
    pub sum_squared_reward: AtomicU64,

    /// Number of RAVE visits (AMAF)
    pub rave_visits: AtomicU64,

    /// Total RAVE reward
    pub rave_reward: AtomicU64,

    /// Prior probability for this node (P(s,a))
    /// Used by PUCT policy. Defaults to 1.0 if not set.
    pub prior: AtomicU64,

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
            sum_squared_reward: AtomicU64::new(0),
            rave_visits: AtomicU64::new(0),
            rave_reward: AtomicU64::new(0),
            prior: AtomicU64::new(float_to_scaled_u64(1.0)), // Default prior is 1.0
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

    /// Returns the prior probability of this node
    pub fn prior(&self) -> f64 {
        scaled_u64_to_float(self.prior.load(Ordering::Relaxed))
    }

    /// Sets the prior probability of this node
    pub fn set_prior(&self, prior: f64) {
        self.prior
            .store(float_to_scaled_u64(prior), Ordering::Relaxed);
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

    /// Adds squared reward (for UCB1-Tuned)
    pub fn add_squared_reward(&self, reward: f64) {
        self.sum_squared_reward
            .fetch_add(float_to_scaled_u64(reward * reward), Ordering::Relaxed);
    }

    /// Returns the sum of squared rewards
    pub fn sum_squared_reward(&self) -> f64 {
        scaled_u64_to_float(self.sum_squared_reward.load(Ordering::Relaxed))
    }

    /// Increments the RAVE visit count
    pub fn increment_rave_visits(&self) {
        self.rave_visits.fetch_add(1, Ordering::Relaxed);
    }

    /// Adds RAVE reward
    pub fn add_rave_reward(&self, reward: f64) {
        self.rave_reward
            .fetch_add(float_to_scaled_u64(reward), Ordering::Relaxed);
    }

    /// Returns the number of RAVE visits
    pub fn rave_visits(&self) -> u64 {
        self.rave_visits.load(Ordering::Relaxed)
    }

    /// Returns the RAVE value (average RAVE reward)
    pub fn rave_value(&self) -> f64 {
        let visits = self.rave_visits();
        if visits == 0 {
            return 0.0;
        }
        scaled_u64_to_float(self.rave_reward.load(Ordering::Relaxed)) / visits as f64
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

    /// Expands the node using a node pool for better performance
    ///
    /// This version of expand uses a node pool to reduce allocation overhead.
    /// It's recommended for performance-critical applications.
    pub fn expand_with_pool(
        &mut self,
        action_index: usize,
        pool: &mut NodePool<S>,
    ) -> Option<&mut MCTSNode<S>> {
        if action_index >= self.unexpanded_actions.len() {
            return None;
        }

        let action = self.unexpanded_actions.swap_remove(action_index);
        let next_state = self.state.apply_action(&action);
        let current_player = self.state.get_current_player();

        // Create a new node using the pool
        let node = pool.create_node(
            next_state,
            Some(action),
            Some(current_player),
            self.depth + 1,
        );

        self.children.push(node);
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

    /// Expands a random unexpanded action using a node pool
    pub fn expand_random_with_pool(&mut self, pool: &mut NodePool<S>) -> Option<&mut MCTSNode<S>> {
        if self.unexpanded_actions.is_empty() {
            return None;
        }

        // Use IteratorRandom trait for choose method on range
        let mut rng = rand::thread_rng();
        let index = (0..self.unexpanded_actions.len()).choose(&mut rng).unwrap();

        self.expand_with_pool(index, pool)
    }
}

/// Pool for efficient node allocation in MCTS
///
/// This implementation provides memory reuse by creating and recycling nodes
/// instead of frequently allocating and deallocating them. This can significantly
/// improve performance in large MCTS searches.
pub struct NodePool<S: GameState> {
    /// Template state used for creating new nodes
    template_state: S,

    /// Preallocated, reusable nodes for efficient reuse
    free_nodes: Vec<MCTSNode<S>>,

    /// Statistics about allocations
    stats: NodePoolStats,
}

/// Statistics for node pool performance tracking
#[derive(Debug, Default, Clone)]
pub struct NodePoolStats {
    /// Total nodes created by the pool
    pub total_created: usize,

    /// Total nodes allocated (both new and reused)
    pub total_allocations: usize,

    /// Total nodes recycled back to the pool
    pub total_recycled: usize,
}

impl<S: GameState> NodePool<S> {
    /// Creates a new node pool with the given template state
    ///
    /// # Arguments
    ///
    /// * `template_state` - A template state that can be cloned when creating new nodes
    /// * `initial_size` - Number of nodes to preallocate
    pub fn new(template_state: S, initial_size: usize) -> Self {
        let mut pool = NodePool {
            template_state,
            free_nodes: Vec::with_capacity(initial_size),
            stats: NodePoolStats::default(),
        };

        // Preallocate nodes if requested
        if initial_size > 0 {
            pool.preallocate(initial_size);
        }

        pool
    }

    /// Preallocate nodes to reduce allocation pressure during search
    fn preallocate(&mut self, count: usize) {
        for _ in 0..count {
            let node = MCTSNode {
                state: self.template_state.clone(),
                action: None,
                visits: AtomicU64::new(0),
                total_reward: AtomicU64::new(0),
                sum_squared_reward: AtomicU64::new(0),
                rave_visits: AtomicU64::new(0),
                rave_reward: AtomicU64::new(0),
                prior: AtomicU64::new(float_to_scaled_u64(1.0)),
                children: Vec::new(),
                unexpanded_actions: Vec::new(),
                depth: 0,
                player: self.template_state.get_current_player(),
            };

            self.free_nodes.push(node);
            self.stats.total_created += 1;
        }
    }

    /// Creates a new node, either from the pool or by allocating a new one
    pub fn create_node(
        &mut self,
        state: S,
        action: Option<S::Action>,
        parent_player: Option<S::Player>,
        depth: usize,
    ) -> MCTSNode<S> {
        self.stats.total_allocations += 1;

        if let Some(mut node) = self.free_nodes.pop() {
            // Get player before moving state
            let player = match &parent_player {
                Some(p) => p.clone(),
                None => state.get_current_player(),
            };

            // Get legal actions before moving state
            let legal_actions = state.get_legal_actions();

            // Reuse an existing node
            node.state = state;
            node.action = action;
            node.visits = AtomicU64::new(0);
            node.total_reward = AtomicU64::new(0);
            node.sum_squared_reward = AtomicU64::new(0);
            node.rave_visits = AtomicU64::new(0);
            node.rave_reward = AtomicU64::new(0);
            node.prior = AtomicU64::new(float_to_scaled_u64(1.0));
            node.children.clear();
            node.depth = depth;
            node.player = player;
            node.unexpanded_actions = legal_actions;

            node
        } else {
            // Create a new node if the pool is empty
            self.stats.total_created += 1;
            MCTSNode::new(state, action, parent_player, depth)
        }
    }

    /// Recycles a node back to the pool for future reuse
    pub fn recycle_node(&mut self, mut node: MCTSNode<S>) {
        self.stats.total_recycled += 1;

        // Clear any large data structures to prevent memory bloat
        node.children.clear();
        node.unexpanded_actions.clear();

        // Add the node back to the free list
        self.free_nodes.push(node);
    }

    /// Recycles all nodes in a tree by recursively adding them to the pool
    pub fn recycle_tree(&mut self, mut root: MCTSNode<S>) {
        // First, recursively recycle all children
        let mut children = std::mem::take(&mut root.children);
        for child in children.drain(..) {
            self.recycle_tree(child);
        }

        // Then recycle the root node itself
        self.recycle_node(root);
    }

    /// Get statistics about pool utilization
    pub fn get_stats(&self) -> &NodePoolStats {
        &self.stats
    }

    /// Get current pool size (available nodes)
    pub fn available_nodes(&self) -> usize {
        self.free_nodes.len()
    }
}

// Manual Clone implementation for NodePool
impl<S: GameState> Clone for NodePool<S> {
    fn clone(&self) -> Self {
        // Create a new pool with the same template state and stats
        // We don't clone the free_nodes as they cannot be shared between instances
        // Instead, we'll create new nodes when needed
        NodePool {
            template_state: self.template_state.clone(),
            free_nodes: Vec::new(), // Start with empty free_nodes
            stats: self.stats.clone(),
        }
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

/// Standalone helper function for tree recycling
///
/// This needs to be outside the MCTS impl to avoid borrow checker issues
pub fn recycle_subtree_recursive<S: GameState>(mut node: MCTSNode<S>, pool: &mut NodePool<S>) {
    // First take all children
    let mut children = std::mem::take(&mut node.children);

    // Recursively recycle each child
    for child in children.drain(..) {
        recycle_subtree_recursive(child, pool);
    }

    // Now recycle the node itself
    pool.recycle_node(node);
}
