//! Main implementation of the Monte Carlo Tree Search algorithm
//!
//! This module contains the core MCTS implementation, orchestrating the
//! four phases of selection, expansion, simulation, and backpropagation.

use std::time::{Duration, Instant};

use rand::prelude::IteratorRandom;

use crate::{
    config::MCTSConfig,
    game_state::GameState,
    policy::{
        backpropagation::{BackpropagationPolicy, StandardPolicy},
        selection::{SelectionPolicy, UCB1Policy},
        simulation::{RandomPolicy, SimulationPolicy},
    },
    stats::SearchStatistics,
    tree::{MCTSNode, NodePath},
    MCTSError, Result,
};
/// Standalone helper function to recursively recycle a subtree
/// 
/// This needs to be outside the MCTS impl to avoid borrow checker issues
fn recycle_subtree_recursive<S: GameState>(
    mut node: MCTSNode<S>,
    pool: &mut crate::tree::NodePool<S>
) {
    // First take all children
    let mut children = std::mem::take(&mut node.children);
    
    // Recursively recycle each child
    for child in children.drain(..) {
        recycle_subtree_recursive(child, pool);
    }
    
    // Now recycle the node itself
    pool.recycle_node(node);
}

/// The main Monte Carlo Tree Search implementation
///
/// This struct manages the MCTS algorithm, including tree building and traversal,
/// and provides methods to run the search and retrieve results.
pub struct MCTS<S: GameState> {
    /// Root node of the search tree
    pub root: MCTSNode<S>,

    /// Configuration for the search
    config: MCTSConfig,

    /// Statistics gathered during search
    statistics: SearchStatistics,

    /// Policy for selecting nodes during the selection phase
    selection_policy: Box<dyn SelectionPolicy<S>>,

    /// Policy for simulating games during the simulation phase
    simulation_policy: Box<dyn SimulationPolicy<S>>,

    /// Policy for backpropagating results
    backpropagation_policy: Box<dyn BackpropagationPolicy<S>>,
    
    /// Node pool for efficient node allocation
    node_pool: Option<crate::tree::NodePool<S>>,
}

impl<S: GameState + 'static> MCTS<S> {
    /// Creates a new MCTS instance with the given initial state and configuration
    pub fn new(initial_state: S, config: MCTSConfig) -> Self {
        // Create the root node
        let root = MCTSNode::new(initial_state, None, None, 0);

        // Create default policies
        let selection_policy: Box<dyn SelectionPolicy<S>> =
            Box::new(UCB1Policy::new(config.exploration_constant));

        let simulation_policy: Box<dyn SimulationPolicy<S>> = Box::new(RandomPolicy::new());

        let backpropagation_policy: Box<dyn BackpropagationPolicy<S>> =
            Box::new(StandardPolicy::new());

        // Create an initial node pool - disabled by default
        let node_pool = None;

        MCTS {
            root,
            config,
            statistics: SearchStatistics::new(),
            selection_policy,
            simulation_policy,
            backpropagation_policy,
            node_pool,
        }
    }
    
    /// Creates a new MCTS instance with a node pool for improved performance
    ///
    /// This constructor initializes MCTS with a node pool, which can significantly
    /// reduce allocation overhead during search.
    ///
    /// # Arguments
    ///
    /// * `initial_state` - The initial game state
    /// * `config` - Configuration for the search
    /// * `initial_pool_size` - Initial number of nodes to pre-allocate
    pub fn with_node_pool(
        initial_state: S, 
        config: MCTSConfig,
        initial_pool_size: usize,
        _pool_chunk_size: usize  // Kept for API compatibility
    ) -> Self {
        let mut mcts = Self::new(initial_state.clone(), config);
        
        // Initialize the node pool with the template state
        let pool = crate::tree::NodePool::new(initial_state, initial_pool_size);
        mcts.node_pool = Some(pool);
        
        mcts
    }

    /// Sets the selection policy to use
    pub fn with_selection_policy<P: SelectionPolicy<S> + 'static>(mut self, policy: P) -> Self {
        self.selection_policy = Box::new(policy);
        self
    }

    /// Sets the simulation policy to use
    pub fn with_simulation_policy<P: SimulationPolicy<S> + 'static>(mut self, policy: P) -> Self {
        self.simulation_policy = Box::new(policy);
        self
    }

    /// Sets the backpropagation policy to use
    pub fn with_backpropagation_policy<P: BackpropagationPolicy<S> + 'static>(
        mut self,
        policy: P,
    ) -> Self {
        self.backpropagation_policy = Box::new(policy);
        self
    }

    /// Runs the search algorithm and returns the best action
    pub fn search(&mut self) -> Result<S::Action> {
        // Initialize node pool if it's enabled in the config but not created yet
        if self.node_pool.is_none() && self.config.node_pool_size > 0 {
            self.node_pool = Some(crate::tree::NodePool::new(
                self.root.state.clone(),
                self.config.node_pool_size
            ));
        }
        
        // First recycle the previous search tree if we have one
        self.recycle_tree();
        
        // Perform the search
        let result = self.search_for_iterations(self.config.max_iterations);
        
        // If using node pooling, we need to select the best action before recycling
        let best_action = if result.is_ok() {
            Some(result.as_ref().unwrap().clone())
        } else {
            None
        };
        
        // Return the result
        match best_action {
            Some(action) => Ok(action),
            None => result,
        }
    }

    // Removed duplicated function - using the top-level function instead
    /// Runs the search for the specified number of iterations
    pub fn search_for_iterations(&mut self, iterations: usize) -> Result<S::Action> {
        // Reset statistics
        self.statistics = SearchStatistics::new();

        // Check if we have any legal actions
        if self.root.unexpanded_actions.is_empty() && self.root.children.is_empty() {
            return Err(MCTSError::NoLegalActions);
        }

        let start_time = Instant::now();
        let max_time = self.config.max_time;

        // Main search loop
        for i in 0..iterations {
            // Check time constraints if set
            if let Some(max_duration) = max_time {
                if start_time.elapsed() >= max_duration {
                    self.statistics.stopped_early = true;
                    println!("Search stopped early due to time limit");
                    break;
                }
            }

            // Execute one iteration of MCTS
            self.execute_iteration()?;

            // Update stats
            self.statistics.iterations = i + 1;
        }

        self.statistics.total_time = start_time.elapsed();
        
        // Collect node pool statistics if available
        if let Some(pool) = &self.node_pool {
            let stats = pool.get_stats();
            self.statistics.update_node_pool_stats(
                stats.total_created, 
                pool.available_nodes(), 
                stats.total_allocations, 
                stats.total_recycled
            );
        }

        // Select the best action based on configured criteria
        self.select_best_action()
    }

    /// Runs the search for the specified duration
    ///
    /// Creates a new MCTS search with the given time limit and runs it.
    /// This is useful when you want to run a time-limited search without
    /// changing the original configuration.
    ///
    /// # Arguments
    ///
    /// * `duration` - The maximum time to run the search
    ///
    /// # Returns
    ///
    /// * `Ok(action)` - The best action found during the search
    /// * `Err(MCTSError)` - If the search couldn't complete successfully
    pub fn search_for_time(&mut self, duration: Duration) -> Result<S::Action> {
        // First recycle the previous search tree
        self.recycle_tree();
        
        let mut config = self.config.clone();
        config.max_time = Some(duration);
        
        // Keep a reasonable max iterations to prevent runaway search
        // if time checking fails for some reason
        if config.max_iterations == usize::MAX {
            config.max_iterations = 1_000_000;
        }

        // Create a new MCTS instance with or without node pool based on configuration
        let mut mcts = if config.node_pool_size > 0 {
            // We need to extract the values before moving config
            let node_pool_size = config.node_pool_size;
            let node_pool_chunk_size = config.node_pool_chunk_size;
            
            // If we already have a node pool, we can share it
            if let Some(_) = &self.node_pool {
                // Create a new instance using the clone of the state
                let mut new_mcts = MCTS::new(self.root.state.clone(), config);
                
                // Create a new node pool with the same size
                new_mcts.node_pool = Some(crate::tree::NodePool::new(
                    self.root.state.clone(),
                    node_pool_size
                ));
                
                new_mcts
            } else {
                // Create a fresh instance with a new pool
                MCTS::with_node_pool(
                    self.root.state.clone(), 
                    config,
                    node_pool_size,
                    node_pool_chunk_size
                )
            }
        } else {
            MCTS::new(self.root.state.clone(), config)
        };
        
        // Set policies
        mcts = mcts
            .with_selection_policy(self.selection_policy.clone_box())
            .with_simulation_policy(self.simulation_policy.clone_box())
            .with_backpropagation_policy(self.backpropagation_policy.clone_box());

        let result = mcts.search();
        
        // If the search was successful, update our statistics
        if result.is_ok() {
            self.statistics = mcts.statistics.clone();
        }
        
        result
    }

    /// Execute a single iteration of the MCTS algorithm
    fn execute_iteration(&mut self) -> Result<()> {
        // 1. Selection phase
        let selected_path = self.selection();

        // 2. Expansion phase
        let (_expanded_node, expanded_state) = self.expansion(&selected_path)?;

        // 3. Simulation phase
        let result = self.simulation(&expanded_state);

        // 4. Backpropagation phase
        self.backpropagation(&selected_path, result);

        Ok(())
    }

    /// Selection phase: Find a promising node to expand
    fn selection(&mut self) -> NodePath {
        let mut path = NodePath::new();

        // Restructured to avoid closure borrowing issues
        // This manually implements what for_tree would do
        let mut current = &self.root;
        let mut depth = 0;

        // Continue while the node meets the traversal conditions
        while !current.state.is_terminal()
            && current.is_fully_expanded()
            && !current.children.is_empty()
        {
            // Select the best child according to the selection policy
            let best_child_idx = self.selection_policy.select_child(current);

            // Update the path
            path.push(best_child_idx);

            // Move to the selected child
            current = &current.children[best_child_idx];
            depth += 1;

            // Update statistics
            self.statistics.max_depth = self.statistics.max_depth.max(depth);

            // Check exit conditions
            if current.state.is_terminal() || !current.is_fully_expanded() {
                break;
            }
        }

        path
    }

    /// Expansion phase: Create a new child node for the selected node
    fn expansion(&mut self, path: &NodePath) -> Result<(NodePath, S)> {
        // Navigate to the selected node
        let mut node = &mut self.root;
        let mut expanded_path = path.clone();

        // Follow the path to get to the selected node
        for &index in &path.indices {
            node = &mut node.children[index];
        }

        // If the node is terminal, we can't expand it
        if node.state.is_terminal() {
            return Ok((expanded_path, node.state.clone()));
        }

        // If there are unexpanded actions, choose one randomly
        if !node.unexpanded_actions.is_empty() {
            let mut rng = rand::thread_rng();
            let action_index = (0..node.unexpanded_actions.len()).choose(&mut rng).unwrap();

            // Decide whether to use the node pool
            let expansion_result = if let Some(pool) = &mut self.node_pool {
                // Use the node pool for better performance
                node.expand_with_pool(action_index, pool)
            } else {
                // Use standard expansion without a pool
                node.expand(action_index)
            };

            // If expansion was successful
            if expansion_result.is_some() {
                // The index of the new child is the last one
                let new_child_index = node.children.len() - 1;

                // Add the expanded node to the path
                expanded_path.push(new_child_index);

                // Update statistics
                self.statistics.tree_size += 1;

                // Update node pool statistics if available
                if let Some(pool) = &self.node_pool {
                    let pool_stats = pool.get_stats();
                    self.statistics.node_pool_stats = Some(crate::stats::NodePoolStats {
                        capacity: pool_stats.total_created,
                        available: pool.available_nodes(),
                        total_allocated: pool_stats.total_allocations,
                        total_returned: pool_stats.total_recycled,
                    });
                }

                // Access the state after expansion is complete
                let expanded_state = node.children[new_child_index].state.clone();

                return Ok((expanded_path, expanded_state));
            }
        }

        // If we couldn't expand, just return the original node
        Ok((expanded_path, node.state.clone()))
    }

    /// Simulation phase: Play out the game from the expanded node
    fn simulation(&self, state: &S) -> f64 {
        self.simulation_policy.simulate(state)
    }

    /// Backpropagation phase: Update statistics in all nodes along the path
    fn backpropagation(&mut self, path: &NodePath, result: f64) {
        // First, update the root node
        self.backpropagation_policy
            .update_stats(&mut self.root, result);

        // Then update all nodes along the path
        let mut node = &mut self.root;

        for &index in &path.indices {
            node = &mut node.children[index];
            self.backpropagation_policy.update_stats(node, result);
        }
    }

    /// Selects the best action based on configured criteria
    fn select_best_action(&self) -> Result<S::Action> {
        // If there are no children, we need to make a first-play move
        if self.root.children.is_empty() {
            if self.root.unexpanded_actions.is_empty() {
                return Err(MCTSError::NoLegalActions);
            }

            // Choose the first unexpanded action
            return Ok(self.root.unexpanded_actions[0].clone());
        }

        // Depending on the best child criteria in config
        match self.config.best_child_criteria {
            // Most visits (robust choice)
            crate::config::BestChildCriteria::MostVisits => {
                let mut best_visits = 0;
                let mut best_index = 0;

                for (i, child) in self.root.children.iter().enumerate() {
                    let visits = child.visits();
                    if visits > best_visits {
                        best_visits = visits;
                        best_index = i;
                    }
                }

                // Get the action that led to this child
                let action = self.root.children[best_index]
                    .action
                    .clone()
                    .ok_or(MCTSError::NoLegalActions)?;

                Ok(action)
            }

            // Highest value (can be more exploitative)
            crate::config::BestChildCriteria::HighestValue => {
                let mut best_value = f64::NEG_INFINITY;
                let mut best_index = 0;

                for (i, child) in self.root.children.iter().enumerate() {
                    let value = child.value();
                    if value > best_value {
                        best_value = value;
                        best_index = i;
                    }
                }

                // Get the action that led to this child
                let action = self.root.children[best_index]
                    .action
                    .clone()
                    .ok_or(MCTSError::NoLegalActions)?;

                Ok(action)
            }
        }
    }

    /// Returns the search statistics
    pub fn get_statistics(&self) -> &SearchStatistics {
        &self.statistics
    }
    /// Recycles the entire search tree back to the node pool
    ///
    /// This releases all nodes (except the root) back to the pool for reuse in
    /// future searches. This can significantly improve performance when
    /// running multiple consecutive searches.
    pub fn recycle_tree(&mut self) {
        if let Some(pool) = &mut self.node_pool {
            // Take all children from the root
            let mut children = std::mem::take(&mut self.root.children);
            
            // Recycle each child tree (using a standalone function to avoid borrow issues)
            for child in children.drain(..) {
                recycle_subtree_recursive(child, pool);
            }
            
            // Make sure the root node still has valid unexpanded_actions
            // This is critical for subsequent searches
            if self.root.unexpanded_actions.is_empty() {
                // Regenerate the unexpanded actions if they're missing
                self.root.unexpanded_actions = self.root.state.get_legal_actions();
            }
            
            // Update statistics
            let stats = pool.get_stats();
            self.statistics.update_node_pool_stats(
                stats.total_created,
                pool.available_nodes(),
                stats.total_allocations,
                stats.total_recycled
            );
        }
    }

    /// Returns a visualization of the search tree
    pub fn visualize_tree(&self) -> String {
        let mut result = String::new();
        Self::visualize_node(&self.root, 0, &mut result);
        result
    }

    /// Helper method to visualize a node and its children
    fn visualize_node(node: &MCTSNode<S>, depth: usize, output: &mut String) {
        let indent = "  ".repeat(depth);
        let action_str = match &node.action {
            Some(action) => format!("{:?}", action),
            None => "Root".to_string(),
        };

        output.push_str(&format!(
            "{}{} (visits: {}, value: {:.3})\n",
            indent,
            action_str,
            node.visits(),
            node.value()
        ));

        for child in &node.children {
            Self::visualize_node(child, depth + 1, output);
        }
    }
}
