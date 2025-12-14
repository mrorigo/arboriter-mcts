use arboriter_mcts::{
    game_state::{Action, Player},
    policy::selection::{PUCTPolicy, SelectionPolicy, UCB1Policy, UCB1TunedPolicy},
    tree::MCTSNode,
    GameState,
};

/// Simple game state for testing
#[derive(Clone, Debug)]
struct TestGameState {
    terminal: bool,
    actions: Vec<TestAction>,
    player: TestPlayer,
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
        // For testing, just return a clone
        self.clone()
    }

    fn is_terminal(&self) -> bool {
        self.terminal
    }

    fn get_result(&self, _for_player: &Self::Player) -> f64 {
        0.5 // Draw for testing
    }

    fn get_current_player(&self) -> Self::Player {
        self.player.clone()
    }
}

// Create a test node with specific characteristics for policy testing
fn create_test_node_for_policy() -> MCTSNode<TestGameState> {
    let state = TestGameState {
        terminal: false,
        actions: vec![TestAction(0), TestAction(1), TestAction(2)],
        player: TestPlayer(1),
    };

    let mut node = MCTSNode::new(state, None, None, 0);

    // Manually expand to create children
    node.expand(0); // Child 0
    node.expand(0); // Child 1

    // Set parent visits
    for _ in 0..100 {
        node.increment_visits();
    }

    // Child 0: High value (0.9), high visits
    for _ in 0..50 {
        node.children[0].increment_visits();
        node.children[0].add_reward(0.9);
    }

    // Child 1: Low value (0.4), low visits
    for _ in 0..10 {
        node.children[1].increment_visits();
        node.children[1].add_reward(0.4);
    }

    node
}

#[test]
fn test_ucb1_exploration_exploitation_balance() {
    let node = create_test_node_for_policy();

    // Debug output to see the actual values
    println!(
        "Child 0: value = {}, visits = {}",
        node.children[0].value(),
        node.children[0].visits()
    );
    println!(
        "Child 1: value = {}, visits = {}",
        node.children[1].value(),
        node.children[1].visits()
    );

    // With low exploration constant, exploitation dominates
    // Child 0 has higher value (0.9) than Child 1 (0.4), so should be chosen
    let policy_exploitative = UCB1Policy::new(0.1);
    let choice_exploitative = policy_exploitative.select_child(&node);
    assert_eq!(
        choice_exploitative, 0,
        "With low exploration constant, should prefer child with higher value"
    );

    // With extremely high exploration constant, exploration dominates
    // Child 1 has fewer visits (10) than Child 0 (50), so should be chosen
    let policy_explorative = UCB1Policy::new(100.0);
    let choice_explorative = policy_explorative.select_child(&node);
    assert_eq!(
        choice_explorative, 1,
        "With very high exploration constant, should prefer less-visited child"
    );
}

#[test]
fn test_ucb1_tuned_policy() {
    let node = create_test_node_for_policy();

    let policy = UCB1TunedPolicy::new(1.414);
    let choice = policy.select_child(&node);

    // We're just ensuring it runs without crashing here
    // Deeper verification would require more specific test cases
    assert!(
        choice == 0 || choice == 1,
        "UCB1Tuned should select a valid child"
    );
}

#[test]
fn test_puct_policy() {
    let node = create_test_node_for_policy();

    let policy_default = PUCTPolicy::new(1.0);
    let choice_default = policy_default.select_child(&node);
    assert!(
        choice_default == 0 || choice_default == 1,
        "PUCT with default priors should select a valid child"
    );

    // Test with explicit priors set on the nodes
    let mut node_with_priors = create_test_node_for_policy();
    node_with_priors.children[0].set_prior(0.1);
    node_with_priors.children[1].set_prior(0.9);

    let policy_with_priors = PUCTPolicy::new(1.0); // Policy itself doesn't need priors now
    let choice_with_priors = policy_with_priors.select_child(&node_with_priors);

    // With strong prior for child 1, it should be selected
    assert_eq!(
        choice_with_priors, 1,
        "PUCT with priors should favor child with higher prior"
    );
}

#[test]
fn test_clone_box() {
    let policy1 = UCB1Policy::new(1.414);
    let boxed: Box<dyn SelectionPolicy<TestGameState>> = Box::new(policy1);
    let _cloned = boxed.clone_box();

    // If we got here without panicking, it works
    // We can't really compare the cloned box easily
}
