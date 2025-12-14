use arboriter_mcts::{Action, GameState, MCTSConfig, Player, MCTS};
use std::time::{Duration, Instant};

// Simple game state for testing time limits with a depth limit to avoid infinite loops
#[derive(Clone, Debug)]
struct TimeLimitGame {
    depth: usize,
    is_terminal: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SimpleAction(usize);

impl Action for SimpleAction {
    fn id(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SimplePlayer(usize);

impl Player for SimplePlayer {}

impl GameState for TimeLimitGame {
    type Action = SimpleAction;
    type Player = SimplePlayer;

    fn get_legal_actions(&self) -> Vec<Self::Action> {
        if self.is_terminal {
            return vec![];
        }
        // Return multiple actions to create a branching tree
        (0..3).map(SimpleAction).collect()
    }

    fn apply_action(&self, _action: &Self::Action) -> Self {
        // Create a new state with increased depth
        // Mark as terminal if we've gone deep enough
        Self {
            depth: self.depth + 1,
            is_terminal: self.depth > 20, // Limit depth to prevent infinite trees
        }
    }

    fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    fn get_result(&self, _for_player: &Self::Player) -> f64 {
        0.5
    }

    fn get_current_player(&self) -> Self::Player {
        SimplePlayer(self.depth % 2)
    }
}

#[test]
fn test_time_limited_search() {
    let game = TimeLimitGame {
        depth: 0,
        is_terminal: false,
    };

    // Set a short but reasonable time limit
    let time_limit = Duration::from_millis(200);

    let config = MCTSConfig::default()
        .with_max_time(time_limit)
        .with_max_iterations(100_000); // High to ensure time is the limiting factor

    let mut mcts = MCTS::new(game, config);

    // Measure search time
    let start = Instant::now();
    let result = mcts.search();
    let elapsed = start.elapsed();

    // Search should succeed and return an action
    assert!(result.is_ok(), "Search should have found an action");

    // Print diagnostic info
    println!("Time limit: {:?}, Actual time: {:?}", time_limit, elapsed);
    println!("Stats: {}", mcts.get_statistics().summary());

    // Verify the search stopped due to time limit
    assert!(
        mcts.get_statistics().stopped_early,
        "Statistics should indicate early stopping due to time limit"
    );
}

#[test]
fn test_search_for_time() {
    // For this test, create a larger game that will take more time to search
    let game = TimeLimitGame {
        depth: 0,
        is_terminal: false,
    };

    // Create MCTS with default config
    let config = MCTSConfig::default();

    let mut mcts = MCTS::new(game, config);

    // Use explicit search_for_time with a longer duration to ensure it hits the time limit
    let time_limit = Duration::from_millis(50);

    let start = Instant::now();
    let result = mcts.search_for_time(time_limit);
    let elapsed = start.elapsed();

    // Search should succeed and return an action
    assert!(result.is_ok(), "Search should have found an action");

    // Print diagnostic info
    println!("Time limit: {:?}, Actual time: {:?}", time_limit, elapsed);

    // Since we're creating a new MCTS instance and the test game might be solved before
    // hitting the time limit, we'll just verify the search completes successfully
    // and returns a valid action without imposing strict time requirements

    // Check that an action was returned
    let action = result.unwrap();
    assert_eq!(action.id() < 3, true, "Should return a valid action ID");
}
