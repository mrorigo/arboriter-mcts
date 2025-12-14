//! Backpropagation policies for the MCTS algorithm
//!
//! Backpropagation policies determine how to update node statistics
//! after a simulation.

use crate::{
    game_state::{Action, GameState},
    tree::MCTSNode,
};

/// Trait for policies that backpropagate simulation results
pub trait BackpropagationPolicy<S: GameState>: Send + Sync {
    /// Updates statistics for a node based on a simulation result
    fn update_stats(&self, node: &mut MCTSNode<S>, result: f64, trace: Option<&[S::Action]>);

    /// Create a boxed clone of this policy
    fn clone_box(&self) -> Box<dyn BackpropagationPolicy<S>>;
}

/// Standard backpropagation policy
///
/// This policy simply increments the visit count and adds the result
/// to the total reward.
#[derive(Debug, Clone)]
pub struct StandardPolicy;

impl StandardPolicy {
    /// Creates a new standard policy
    pub fn new() -> Self {
        StandardPolicy
    }
}

impl Default for StandardPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: GameState> BackpropagationPolicy<S> for StandardPolicy {
    fn update_stats(&self, node: &mut MCTSNode<S>, result: f64, _trace: Option<&[S::Action]>) {
        node.increment_visits();
        node.add_reward(result);
        node.add_squared_reward(result);
    }

    fn clone_box(&self) -> Box<dyn BackpropagationPolicy<S>> {
        Box::new(self.clone())
    }
}

// Implement BackpropagationPolicy for Box<dyn BackpropagationPolicy>
impl<S: GameState> BackpropagationPolicy<S> for Box<dyn BackpropagationPolicy<S>> {
    fn update_stats(&self, node: &mut MCTSNode<S>, result: f64, trace: Option<&[S::Action]>) {
        (**self).update_stats(node, result, trace)
    }

    fn clone_box(&self) -> Box<dyn BackpropagationPolicy<S>> {
        (**self).clone_box()
    }
}

/// Weighted backpropagation policy
///
/// This policy weights results based on how deep they are in the tree.
/// Early results might be more reliable than deep ones, or vice versa.
#[derive(Debug, Clone)]
pub struct WeightedPolicy {
    /// Depth weighting factor (how quickly weight changes with depth)
    /// - Positive values make deeper nodes less influential
    /// - Negative values make deeper nodes more influential
    pub depth_factor: f64,
}

impl WeightedPolicy {
    /// Creates a new weighted policy with the given depth factor
    pub fn new(depth_factor: f64) -> Self {
        WeightedPolicy { depth_factor }
    }
}

impl<S: GameState> BackpropagationPolicy<S> for WeightedPolicy {
    fn update_stats(&self, node: &mut MCTSNode<S>, result: f64, _trace: Option<&[S::Action]>) {
        // Calculate weight based on depth
        // Higher depth means lower weight if depth_factor is positive
        let weight = 1.0 / (1.0 + self.depth_factor * node.depth as f64);

        node.increment_visits();
        node.add_reward(result * weight);
        node.add_squared_reward(result * weight);
    }

    fn clone_box(&self) -> Box<dyn BackpropagationPolicy<S>> {
        Box::new(self.clone())
    }
}

/// Rave (Rapid Action Value Estimation) backpropagation policy
///
/// This policy updates statistics for all nodes in the tree that
/// correspond to the same action, not just those in the current path.
/// This can accelerate learning in games where the same action
/// can occur in different states with similar values.
#[derive(Debug, Clone)]
pub struct RavePolicy {
    /// Weight given to RAVE updates (between 0 and 1)
    pub rave_weight: f64,
}

impl RavePolicy {
    /// Creates a new RAVE policy with the given weight
    pub fn new(rave_weight: f64) -> Self {
        RavePolicy {
            rave_weight: rave_weight.clamp(0.0, 1.0),
        }
    }
}

impl<S: GameState> BackpropagationPolicy<S> for RavePolicy {
    fn update_stats(&self, node: &mut MCTSNode<S>, result: f64, trace: Option<&[S::Action]>) {
        // Standard update
        node.increment_visits();
        node.add_reward(result);
        node.add_squared_reward(result);

        // RAVE (AMAF) update
        if let (Some(trace), Some(node_action)) = (trace, &node.action) {
            // Check if the action leading to this node appears in the action trace
            // (i.e., if this action was played later in the simulation)
            let action_in_trace = trace.iter().any(|a| a.id() == node_action.id());

            if action_in_trace {
                node.increment_rave_visits();
                node.add_rave_reward(result);
            }
        }
    }

    fn clone_box(&self) -> Box<dyn BackpropagationPolicy<S>> {
        Box::new(self.clone())
    }
}
