//! Configuration options for the MCTS algorithm
//!
//! This module defines the configuration parameters that control the
//! behavior of the MCTS algorithm.

use std::time::Duration;

/// Criteria for selecting the best child after search is complete
///
/// This determines how the final action is selected after the search budget is exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BestChildCriteria {
    /// Select the child with the most visits
    ///
    /// This is generally more robust and conservative, as it relies on
    /// statistical confidence rather than potentially noisy value estimates.
    /// This is the standard approach in most MCTS implementations.
    ///
    /// Use this approach when reliability is more important than maximizing potential gain.
    MostVisits,
    
    /// Select the child with the highest average value
    ///
    /// This can be more aggressive by favoring high-value moves even if they
    /// haven't been visited as often. May perform better in some domains but
    /// can be less robust overall.
    ///
    /// Use this approach when you want to maximize expected value and are willing
    /// to accept more risk.
    HighestValue,
}

/// Configuration for the MCTS algorithm
///
/// This struct contains all parameters that control the behavior of the MCTS search.
/// Use the builder methods to create a customized configuration.
///
/// # Example
///
/// ```
/// use arboriter_mcts::{MCTSConfig, config::BestChildCriteria};
/// use std::time::Duration;
///
/// let config = MCTSConfig::default()
///     .with_exploration_constant(1.5)
///     .with_max_iterations(10_000)
///     .with_max_time(Duration::from_secs(5))
///     .with_best_child_criteria(BestChildCriteria::MostVisits)
///     .with_node_pool_config(1000, 500);  // Enable node pooling
/// ```
#[derive(Debug, Clone)]
pub struct MCTSConfig {
    /// Exploration constant for UCB1
    ///
    /// Controls the balance between exploration and exploitation.
    /// Higher values favor exploration of less-visited nodes.
    /// The standard value is sqrt(2) ≈ 1.414.
    pub exploration_constant: f64,
    
    /// Maximum number of iterations to run
    ///
    /// The search will stop after this many iterations, even if there's
    /// still time available.
    pub max_iterations: usize,
    
    /// Maximum time to run the search
    ///
    /// If set, the search will stop after this duration, even if the
    /// maximum iterations haven't been reached.
    pub max_time: Option<Duration>,
    
    /// Maximum depth to search
    ///
    /// If set, the tree will not be expanded beyond this depth.
    pub max_depth: Option<usize>,
    
    /// Whether to use transposition tables
    ///
    /// Transposition tables allow reusing evaluations for states that
    /// can be reached through different sequences of moves.
    pub use_transpositions: bool,
    
    /// Criteria for selecting the best child after search
    ///
    /// Determines how the final action is selected once the search is complete.
    pub best_child_criteria: BestChildCriteria,
    
    /// Node pool initial size
    ///
    /// If set (non-zero), enables the node pool allocator with this initial capacity.
    /// Node pooling can significantly improve performance by reducing allocation overhead.
    pub node_pool_size: usize,
    
    /// Node pool chunk size for growth
    ///
    /// When the node pool needs to grow, it will allocate this many new nodes at once.
    pub node_pool_chunk_size: usize,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        MCTSConfig {
            exploration_constant: 1.414, // sqrt(2)
            max_iterations: 10_000,
            max_time: None,
            max_depth: None,
            use_transpositions: false,
            best_child_criteria: BestChildCriteria::MostVisits,
            node_pool_size: 0, // Disabled by default
            node_pool_chunk_size: 500,
        }
    }
}

impl MCTSConfig {
    /// Sets the exploration constant
    pub fn with_exploration_constant(mut self, constant: f64) -> Self {
        self.exploration_constant = constant;
        self
    }

    /// Sets the maximum number of iterations
    pub fn with_max_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations;
        self
    }

    /// Sets the maximum time to run the search
    pub fn with_max_time(mut self, duration: Duration) -> Self {
        self.max_time = Some(duration);
        self
    }

    /// Sets the maximum depth to search
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Sets whether to use transposition tables
    pub fn with_transpositions(mut self, use_transpositions: bool) -> Self {
        self.use_transpositions = use_transpositions;
        self
    }

    /// Sets the criteria for selecting the best child
    pub fn with_best_child_criteria(mut self, criteria: BestChildCriteria) -> Self {
        self.best_child_criteria = criteria;
        self
    }
    /// Sets the node pool configuration
    ///
    /// Using a node pool can significantly improve performance by reducing
    /// allocation overhead during tree expansion. This is especially
    /// important for games with large branching factors or deep searches.
    ///
    /// # Arguments
    ///
    /// * `initial_size` - Initial number of nodes to pre-allocate
    /// * `chunk_size` - Number of nodes to allocate when the pool needs to grow
    ///
    /// # Returns
    ///
    /// The updated configuration
    pub fn with_node_pool_config(mut self, initial_size: usize, chunk_size: usize) -> Self {
        self.node_pool_size = initial_size;
        self.node_pool_chunk_size = chunk_size;
        self
    }
    
    /// Disables the node pool
    ///
    /// This method disables the node pool, causing MCTS to allocate nodes
    /// directly on the heap. This is the default behavior.
    ///
    /// # Returns
    ///
    /// The updated configuration
    pub fn without_node_pool(mut self) -> Self {
        self.node_pool_size = 0;
        self
    }
    // Thread-local pool support removed for now
}
