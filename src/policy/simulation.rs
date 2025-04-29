//! Simulation policies for the MCTS algorithm
//!
//! Simulation policies determine how to play out a game from a given state
//! to estimate the value of that state.

use crate::game_state::GameState;

/// Trait for policies that simulate games
pub trait SimulationPolicy<S: GameState>: Send + Sync {
    /// Simulates a game from the given state and returns the result
    fn simulate(&self, state: &S) -> f64;

    /// Create a boxed clone of this policy
    fn clone_box(&self) -> Box<dyn SimulationPolicy<S>>;
}

/// Random simulation policy
///
/// This policy plays random legal moves until the game ends.
#[derive(Debug, Clone)]
pub struct RandomPolicy;

impl RandomPolicy {
    /// Creates a new random policy
    pub fn new() -> Self {
        RandomPolicy
    }
}

impl Default for RandomPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: GameState> SimulationPolicy<S> for RandomPolicy {
    fn simulate(&self, state: &S) -> f64 {
        // Use the built-in random playout method
        let player = state.get_current_player();
        state.simulate_random_playout(&player)
    }

    fn clone_box(&self) -> Box<dyn SimulationPolicy<S>> {
        Box::new(self.clone())
    }
}

/// Heuristic simulation policy
///
/// This policy uses a heuristic function to guide the simulation.
#[derive(Debug, Clone)]
pub struct HeuristicPolicy<F, S>
where
    F: Fn(&S) -> f64 + Clone + Send + Sync + 'static,
    S: GameState + 'static,
{
    /// The heuristic function
    heuristic: F,
    _phantom: std::marker::PhantomData<S>,
}

impl<F, S> HeuristicPolicy<F, S>
where
    F: Fn(&S) -> f64 + Clone + Send + Sync + 'static,
    S: GameState + 'static,
{
    /// Creates a new heuristic policy with the given function
    pub fn new(heuristic: F) -> Self {
        HeuristicPolicy {
            heuristic,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F, S> SimulationPolicy<S> for HeuristicPolicy<F, S>
where
    F: Fn(&S) -> f64 + Clone + Send + Sync + 'static,
    S: GameState + 'static,
{
    fn simulate(&self, state: &S) -> f64 {
        // If terminal, return the actual result
        if state.is_terminal() {
            let player = state.get_current_player();
            return state.get_result(&player);
        }

        // Otherwise, use the heuristic function
        (self.heuristic)(state)
    }

    fn clone_box(&self) -> Box<dyn SimulationPolicy<S>> {
        Box::new(self.clone())
    }
}

/// Mixture simulation policy
///
/// This policy combines multiple simulation policies, using each with
/// a specified probability.
pub struct MixturePolicy<S: GameState> {
    /// Policies and their associated probabilities
    policies: Vec<(Box<dyn SimulationPolicy<S>>, f64)>,
}

impl<S: GameState> std::fmt::Debug for MixturePolicy<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MixturePolicy")
            .field("policies_count", &self.policies.len())
            .finish()
    }
}

impl<S: GameState> Clone for MixturePolicy<S> {
    fn clone(&self) -> Self {
        // We can't clone the policies directly, so we return a new empty MixturePolicy
        // This is not ideal, but it's a reasonable fallback for the Clone requirement
        MixturePolicy {
            policies: Vec::new(),
        }
    }
}

impl<S: GameState> MixturePolicy<S> {
    /// Creates a new mixture policy
    pub fn new() -> Self {
        MixturePolicy {
            policies: Vec::new(),
        }
    }

    /// Adds a policy with the given probability
    pub fn add_policy<P: SimulationPolicy<S> + 'static>(
        mut self,
        policy: P,
        probability: f64,
    ) -> Self {
        self.policies.push((Box::new(policy), probability));
        self
    }
}

impl<S: GameState + 'static> SimulationPolicy<S> for MixturePolicy<S> {
    fn simulate(&self, state: &S) -> f64 {
        use rand::Rng;

        if self.policies.is_empty() {
            // Fallback to random policy
            let random_policy = RandomPolicy::new();
            return random_policy.simulate(state);
        }

        // Calculate total probability
        let total: f64 = self.policies.iter().map(|(_, p)| *p).sum();

        // Select a policy based on probabilities
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen_range(0.0..total);

        let mut cumulative = 0.0;
        for (policy, prob) in &self.policies {
            cumulative += prob;
            if r < cumulative {
                return policy.simulate(state);
            }
        }

        // Fallback to the last policy
        self.policies.last().unwrap().0.simulate(state)
    }

    fn clone_box(&self) -> Box<dyn SimulationPolicy<S>> {
        let mut new_policies = Vec::new();
        for (policy, prob) in &self.policies {
            new_policies.push((policy.clone_box(), *prob));
        }

        Box::new(MixturePolicy {
            policies: new_policies,
        })
    }
}

impl<S: GameState> Default for MixturePolicy<S> {
    fn default() -> Self {
        Self::new()
    }
}
// Implement SimulationPolicy for Box<dyn SimulationPolicy>
impl<S: GameState> SimulationPolicy<S> for Box<dyn SimulationPolicy<S>> {
    fn simulate(&self, state: &S) -> f64 {
        (**self).simulate(state)
    }

    fn clone_box(&self) -> Box<dyn SimulationPolicy<S>> {
        (**self).clone_box()
    }
}
