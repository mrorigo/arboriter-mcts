use arboriter_mcts::{
    game_state::{Action, Player},
    policy::backpropagation::{BackpropagationPolicy, RavePolicy, StandardPolicy, WeightedPolicy},
    tree::MCTSNode,
    GameState,
};

/// Simple game state for testing
#[derive(Clone, Debug)]
struct TestGameState {
    terminal: bool,
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
        vec![TestAction(0), TestAction(1)]
    }

    fn apply_action(&self, _action: &Self::Action) -> Self {
        self.clone()
    }

    fn is_terminal(&self) -> bool {
        self.terminal
    }

    fn get_result(&self, _for_player: &Self::Player) -> f64 {
        0.5
    }

    fn get_current_player(&self) -> Self::Player {
        self.player.clone()
    }
}

#[test]
fn test_standard_policy() {
    let state = TestGameState {
        terminal: false,
        player: TestPlayer(1),
    };

    let mut node = MCTSNode::new(state, None, None, 0);

    let policy = StandardPolicy::new();

    // Initial state
    assert_eq!(node.visits(), 0);
    assert_eq!(node.total_reward(), 0.0);

    // Update once
    policy.update_stats(&mut node, 0.5);
    assert_eq!(node.visits(), 1);
    assert_eq!(node.total_reward(), 0.5);

    // Update again
    policy.update_stats(&mut node, 0.75);
    assert_eq!(node.visits(), 2);
    assert_eq!(node.total_reward(), 1.25);
}

#[test]
fn test_weighted_policy() {
    let state = TestGameState {
        terminal: false,
        player: TestPlayer(1),
    };

    // Create two nodes at different depths
    let mut shallow_node = MCTSNode::new(state.clone(), None, None, 1);
    let mut deep_node = MCTSNode::new(state.clone(), None, None, 5);

    // Policy that gives less weight to deeper nodes
    let positive_weight_policy = WeightedPolicy::new(0.5);

    positive_weight_policy.update_stats(&mut shallow_node, 1.0);
    positive_weight_policy.update_stats(&mut deep_node, 1.0);

    // Shallow node should get more reward
    assert!(
        shallow_node.total_reward() > deep_node.total_reward(),
        "Shallow node should get more reward with positive depth factor"
    );

    // Reset nodes
    let mut shallow_node = MCTSNode::new(state.clone(), None, None, 1);
    let mut deep_node = MCTSNode::new(state.clone(), None, None, 5);

    // Policy that gives more weight to deeper nodes
    let negative_weight_policy = WeightedPolicy::new(-0.1);

    negative_weight_policy.update_stats(&mut shallow_node, 1.0);
    negative_weight_policy.update_stats(&mut deep_node, 1.0);

    // Deep node should get more reward
    assert!(
        deep_node.total_reward() > shallow_node.total_reward(),
        "Deep node should get more reward with negative depth factor"
    );
}

#[test]
fn test_rave_policy() {
    let state = TestGameState {
        terminal: false,
        player: TestPlayer(1),
    };

    let mut node = MCTSNode::new(state, None, None, 0);

    let policy = RavePolicy::new(0.5);

    // Initial state
    assert_eq!(node.visits(), 0);
    assert_eq!(node.total_reward(), 0.0);

    // Update
    policy.update_stats(&mut node, 1.0);
    assert_eq!(node.visits(), 1);

    // With 0.5 RAVE weight, the standard update should only get 0.5 weight
    // But since there's no real RAVE implementation yet, we expect a value lower than 1.0
    assert!(node.total_reward() > 0.0);
    assert!(node.total_reward() <= 0.5);
}

#[test]
fn test_backpropagation_policy_cloning() {
    let policy = StandardPolicy::new();
    let boxed: Box<dyn BackpropagationPolicy<TestGameState>> = Box::new(policy);
    let _cloned = boxed.clone_box();

    // Just testing that we don't panic
}
