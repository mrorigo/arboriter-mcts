use arboriter_mcts::{
    game_state::{Action, Player},
    policy::simulation::{HeuristicPolicy, MixturePolicy, RandomPolicy, SimulationPolicy},
    GameState,
};

/// Simple game state for testing
#[derive(Clone, Debug)]
struct TestGameState {
    terminal: bool,
    actions: Vec<TestAction>,
    player: TestPlayer,
    result: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TestPlayer(u8);

impl Player for TestPlayer {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TestAction(u8);

impl Action for TestAction {
    fn id(&self) -> usize {
        self.0 as usize
    }
}

impl GameState for TestGameState {
    type Action = TestAction;
    type Player = TestPlayer;

    fn get_legal_actions(&self) -> Vec<Self::Action> {
        self.actions.clone()
    }

    fn apply_action(&self, _action: &Self::Action) -> Self {
        // Generate a terminal state for simulation
        let mut new_state = self.clone();
        new_state.terminal = true;
        new_state
    }

    fn is_terminal(&self) -> bool {
        self.terminal
    }

    fn get_result(&self, _for_player: &Self::Player) -> f64 {
        self.result
    }

    fn get_current_player(&self) -> Self::Player {
        self.player.clone()
    }

    // Override the random playout to return a predictable result
    fn simulate_random_playout(&self, _for_player: &Self::Player) -> f64 {
        self.result
    }
}

#[test]
fn test_random_policy() {
    let state = TestGameState {
        terminal: false,
        actions: vec![TestAction(0), TestAction(1)],
        player: TestPlayer(1),
        result: 0.75,
    };

    let policy = RandomPolicy::new();
    let result = policy.simulate(&state);

    // Should use the built-in random playout which returns our fixed result
    assert_eq!(result, 0.75);

    // Test with terminal state
    let terminal_state = TestGameState {
        terminal: true,
        actions: vec![],
        player: TestPlayer(1),
        result: 0.25,
    };

    let result_terminal = policy.simulate(&terminal_state);
    assert_eq!(result_terminal, 0.25);
}

#[test]
fn test_heuristic_policy() {
    let state = TestGameState {
        terminal: false,
        actions: vec![TestAction(0), TestAction(1)],
        player: TestPlayer(1),
        result: 0.75,
    };

    // Create a heuristic that always returns 0.42
    let heuristic = |_state: &TestGameState| 0.42;
    let policy = HeuristicPolicy::new(heuristic);

    let result = policy.simulate(&state);
    assert_eq!(
        result, 0.42,
        "Heuristic function should be used for non-terminal states"
    );

    // Terminal state should use actual result
    let terminal_state = TestGameState {
        terminal: true,
        actions: vec![],
        player: TestPlayer(1),
        result: 0.25,
    };

    let result_terminal = policy.simulate(&terminal_state);
    assert_eq!(
        result_terminal, 0.25,
        "Terminal state should use actual result"
    );
}

#[test]
fn test_mixture_policy() {
    let state = TestGameState {
        terminal: false,
        actions: vec![TestAction(0), TestAction(1)],
        player: TestPlayer(1),
        result: 0.5,
    };

    // Always returns 0.3
    let heuristic1 = |_state: &TestGameState| 0.3;
    let policy1 = HeuristicPolicy::new(heuristic1);

    // MixturePolicy with only one policy should return that policy's result
    let mixture_policy = MixturePolicy::new()
        .add_policy(policy1, 1.0);
    
    // Should be using our heuristic consistently
    let mut sum = 0.0;
    for _ in 0..10 {
        sum += mixture_policy.simulate(&state);
    }
    
    // Use approximate comparison for floating point
    let expected = 3.0;
    let epsilon = 1e-10;
    assert!((sum - expected).abs() < epsilon, 
            "With only one policy, should consistently return that policy's result. Expected {}, got {}", 
            expected, sum);

    // Test empty policy (should fall back to random)
    let empty_policy = MixturePolicy::new();
    assert_eq!(
        empty_policy.simulate(&state),
        0.5,
        "Empty policy should fall back to random"
    );
}

#[test]
fn test_simulation_policy_cloning() {
    let random_policy = RandomPolicy::new();
    let boxed: Box<dyn SimulationPolicy<TestGameState>> = Box::new(random_policy);
    let _cloned = boxed.clone_box();

    // Mainly testing that we don't panic
}
