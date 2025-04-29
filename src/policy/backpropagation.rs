//! Backpropagation policies for the MCTS algorithm
//!
//! Backpropagation policies determine how to update node statistics
//! after a simulation.

use crate::{game_state::GameState, tree::MCTSNode};

/// Trait for policies that backpropagate simulation results
pub trait BackpropagationPolicy<S: GameState>: Send + Sync {
    /// Updates statistics for a node based on a simulation result
    fn update_stats(&self, node: &mut MCTSNode<S>, result: f64);

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
    fn update_stats(&self, node: &mut MCTSNode<S>, result: f64) {
        node.increment_visits();
        node.add_reward(result);
    }

    fn clone_box(&self) -> Box<dyn BackpropagationPolicy<S>> {
        Box::new(self.clone())
    }
}

// Implement BackpropagationPolicy for Box<dyn BackpropagationPolicy>
impl<S: GameState> BackpropagationPolicy<S> for Box<dyn BackpropagationPolicy<S>> {
    fn update_stats(&self, node: &mut MCTSNode<S>, result: f64) {
        (**self).update_stats(node, result)
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
    fn update_stats(&self, node: &mut MCTSNode<S>, result: f64) {
        // Calculate weight based on depth
        // Higher depth means lower weight if depth_factor is positive
        let weight = 1.0 / (1.0 + self.depth_factor * node.depth as f64);

        node.increment_visits();
        node.add_reward(result * weight);
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
    fn update_stats(&self, node: &mut MCTSNode<S>, result: f64) {
        // Standard update
        node.increment_visits();

        // RAVE is an advanced policy that requires tracking actions played during
        // the simulation. This is a simplified implementation that just applies
        // a weighted update.
        let standard_weight = 1.0 - self.rave_weight;
        node.add_reward(result * standard_weight);

        // In a real RAVE implementation, we would also update statistics for all
        // nodes corresponding to actions played during the simulation.
        // This would require tracking the actions played, which is beyond the
        // scope of this example.
    }

    fn clone_box(&self) -> Box<dyn BackpropagationPolicy<S>> {
        Box::new(self.clone())
    }
}
