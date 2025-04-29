//! Statistics collection for MCTS searches
//!
//! This module provides structures for collecting and reporting statistics
//! about MCTS search processes.

use std::time::Duration;

/// Statistics collected during an MCTS search
#[derive(Debug, Clone)]
pub struct SearchStatistics {
    /// Number of iterations performed
    pub iterations: usize,

    /// Total time spent searching
    pub total_time: Duration,

    /// Total number of nodes in the tree
    pub tree_size: usize,

    /// Maximum depth reached in the tree
    pub max_depth: usize,

    /// Whether the search was stopped early due to time constraints
    pub stopped_early: bool,
    
    /// Node pool metrics (if node pool is used)
    pub node_pool_stats: Option<NodePoolStats>,
}

/// Statistics about the node pool
#[derive(Debug, Clone)]
pub struct NodePoolStats {
    /// Total capacity of the node pool
    pub capacity: usize,
    
    /// Number of available nodes in the pool
    pub available: usize,
    
    /// Total nodes allocated from the pool
    pub total_allocated: usize,
    
    /// Total nodes returned to the pool
    pub total_returned: usize,
}

impl SearchStatistics {
    /// Creates a new, empty statistics object
    pub fn new() -> Self {
        SearchStatistics {
            iterations: 0,
            total_time: Duration::from_secs(0),
            tree_size: 1, // Start with root node
            max_depth: 0,
            stopped_early: false,
            node_pool_stats: None,
        }
    }
    
    /// Update node pool statistics
    pub fn update_node_pool_stats(&mut self, capacity: usize, available: usize, allocated: usize, returned: usize) {
        self.node_pool_stats = Some(NodePoolStats {
            capacity,
            available,
            total_allocated: allocated,
            total_returned: returned,
        });
    }

    /// Returns the average time per iteration in microseconds
    pub fn avg_time_per_iteration_us(&self) -> f64 {
        if self.iterations == 0 {
            return 0.0;
        }
        self.total_time.as_micros() as f64 / self.iterations as f64
    }

    /// Returns the number of iterations per second
    pub fn iterations_per_second(&self) -> f64 {
        if self.total_time.as_secs_f64() <= 0.0 {
            return 0.0;
        }
        self.iterations as f64 / self.total_time.as_secs_f64()
    }

    /// Returns a summary of the statistics as a string
    pub fn summary(&self) -> String {
        let mut summary = format!(
            "MCTS Search Statistics:\n\
             - Iterations: {}\n\
             - Total time: {:.3} seconds\n\
             - Tree size: {} nodes\n\
             - Max depth: {}\n\
             - Avg time per iteration: {:.3} µs\n\
             - Iterations per second: {:.1}\n\
             - Stopped early: {}",
            self.iterations,
            self.total_time.as_secs_f64(),
            self.tree_size,
            self.max_depth,
            self.avg_time_per_iteration_us(),
            self.iterations_per_second(),
            self.stopped_early
        );
        
        // Add node pool stats if available
        if let Some(pool_stats) = &self.node_pool_stats {
            summary.push_str(&format!(
                "\n\nNode Pool Statistics:\n\
                 - Capacity: {}\n\
                 - Available nodes: {}\n\
                 - Total allocated: {}\n\
                 - Total returned: {}\n\
                 - Reuse ratio: {:.2}%",
                pool_stats.capacity,
                pool_stats.available,
                pool_stats.total_allocated,
                pool_stats.total_returned,
                if pool_stats.total_allocated > 0 {
                    (pool_stats.total_returned as f64 / pool_stats.total_allocated as f64) * 100.0
                } else {
                    0.0
                }
            ));
        }
        
        summary
    }
}

impl Default for SearchStatistics {
    fn default() -> Self {
        Self::new()
    }
}
