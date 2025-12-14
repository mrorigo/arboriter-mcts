//! Traits defining game state representation for MCTS.
//!
//! The GameState trait is the primary interface that must be implemented for any
//! game or decision process that will be used with the MCTS algorithm.

use std::fmt::Debug;

/// Trait for actions that can be taken in a game
///
/// Actions represent the moves or decisions that can be made in a game.
pub trait Action: Clone + Debug + Send + Sync {
    /// Returns a unique identifier for this action
    fn id(&self) -> usize;
}

/// Trait for players in a game
///
/// Players represent the entities making decisions in a game.
pub trait Player: Clone + Debug + PartialEq + Send + Sync {}

/// Trait defining the game state interface required for MCTS
///
/// This trait must be implemented for any game or decision process that will be
/// used with the MCTS algorithm. It defines how to get legal actions, apply actions,
/// determine if the game is over, and calculate results.
pub trait GameState: Clone + Send + Sync {
    /// The type of actions that can be taken in this game
    type Action: Action;

    /// The type of players in this game
    type Player: Player;

    /// Returns the list of legal actions from this state
    ///
    /// This method should return all possible moves that can be made from the current state.
    /// For efficiency, avoid recomputing this list repeatedly if the state doesn't change.
    ///
    /// # Example
    ///
    /// ```
    /// # use arboriter_mcts::{GameState, Action};
    /// # #[derive(Debug, Clone, PartialEq)]
    /// # struct MyAction(usize);
    /// # impl Action for MyAction { fn id(&self) -> usize { self.0 } }
    /// # #[derive(Debug, Clone, PartialEq)]
    /// # struct MyPlayer;
    /// # impl arboriter_mcts::Player for MyPlayer {}
    /// # #[derive(Clone)]
    /// # struct MyGame { /* ... */ }
    /// # impl GameState for MyGame {
    /// # type Action = MyAction;
    /// # type Player = MyPlayer;
    /// fn get_legal_actions(&self) -> Vec<MyAction> {
    ///     // Return all valid moves in the current position
    ///     vec![MyAction(0), MyAction(1), MyAction(2)]
    /// }
    /// # fn apply_action(&self, action: &MyAction) -> Self { self.clone() }
    /// # fn is_terminal(&self) -> bool { false }
    /// # fn get_result(&self, _: &MyPlayer) -> f64 { 0.5 }
    /// # fn get_current_player(&self) -> MyPlayer { MyPlayer }
    /// # }
    /// ```
    fn get_legal_actions(&self) -> Vec<Self::Action>;

    /// Applies an action to the current state, returning the new state
    ///
    /// This method should:
    /// 1. Create a copy of the current state
    /// 2. Apply the given action to modify the state
    /// 3. Return the new state without modifying the original
    ///
    /// It's important that this method is pure (doesn't modify the original state)
    /// since MCTS needs to explore multiple paths from the same state.
    ///
    /// # Parameters
    ///
    /// * `action`: The action to apply, which should be one of the legal actions
    ///
    /// # Returns
    ///
    /// A new state that results from applying the action
    fn apply_action(&self, action: &Self::Action) -> Self;

    /// Returns true if this state is terminal (game over)
    ///
    /// A terminal state is one where no further actions can be taken,
    /// either because the game has been won/lost/drawn or because
    /// some other termination condition has been reached.
    ///
    /// Terminal states should return an empty list from `get_legal_actions()`.
    fn is_terminal(&self) -> bool;

    /// Returns the result of the game from the perspective of the given player
    ///
    /// This method evaluates the current state and returns a score indicating how
    /// good the state is for the specified player. This is used during the backpropagation
    /// phase to update node statistics.
    ///
    /// # Parameters
    ///
    /// * `for_player`: The player from whose perspective to evaluate the result
    ///
    /// # Returns
    ///
    /// A value between 0.0 and 1.0, where:
    /// - 1.0 represents a win for the player
    /// - 0.5 represents a draw or neutral position
    /// - 0.0 represents a loss for the player
    ///
    /// You can also use intermediate values to represent partial wins/losses.
    fn get_result(&self, for_player: &Self::Player) -> f64;

    /// Returns the player whose turn it is in this state
    ///
    /// This is used by MCTS to determine which player will make the next move.
    /// For games with alternating turns, this should return the player who will
    /// act next. For simultaneous-move games, this could return a special marker
    /// or the player whose perspective is being considered.
    fn get_current_player(&self) -> Self::Player;

    /// Performs a random simulation from this state to a terminal state
    ///
    /// This method has a default implementation that uses random actions,
    /// but it can be overridden to use domain-specific knowledge.
    ///
    /// Returns the result from the perspective of the given player and the list of actions taken.
    fn simulate_random_playout(&self, for_player: &Self::Player) -> (f64, Vec<Self::Action>) {
        use rand::seq::SliceRandom;

        let mut rng = rand::thread_rng();
        let mut current_state = self.clone();
        let mut trace = Vec::new();

        // Play random moves until the game is over
        while !current_state.is_terminal() {
            let legal_actions = current_state.get_legal_actions();
            if legal_actions.is_empty() {
                break;
            }

            // Choose a random action
            let action = legal_actions.choose(&mut rng).unwrap();
            trace.push(action.clone());
            current_state = current_state.apply_action(action);
        }

        // Return the result
        (current_state.get_result(for_player), trace)
    }

    /// Returns a hash representing this state, used for transposition tables
    ///
    /// Default implementation returns a constant, effectively disabling
    /// transposition tables. Override this for better performance.
    fn hash(&self) -> u64 {
        0
    }
}

/// Simplified imlementation of Player trait for common types
impl Player for usize {}
impl Player for i32 {}
impl Player for char {}
impl Player for String {}

/// Unit marker for games with no explicit player representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoPlayer;

impl Player for NoPlayer {}
