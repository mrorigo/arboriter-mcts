use arboriter_mcts::{
    game_state::{Action, Player},
    policy::backpropagation::{BackpropagationPolicy, RavePolicy},
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

    fn simulate_random_playout(&self, _for_player: &Self::Player) -> (f64, Vec<Self::Action>) {
        (0.5, Vec::new())
    }
}

#[test]
fn test_rave_update_logic() {
    let state = TestGameState {
        terminal: false,
        actions: vec![TestAction(0), TestAction(1)],
        player: TestPlayer(1),
    };

    // Node representing result of taking Action(0)
    let mut node = MCTSNode::new(state, Some(TestAction(0)), Some(TestPlayer(0)), 1);

    let policy = RavePolicy::new(0.5);

    // 1. Trace contains match
    let trace_match = vec![TestAction(2), TestAction(0), TestAction(3)];
    policy.update_stats(&mut node, 1.0, Some(&trace_match));

    // Should update RAVE stats
    assert_eq!(
        node.rave_visits(),
        1,
        "RAVE visits should increment on match"
    );
    assert_eq!(node.rave_value(), 1.0, "RAVE value should update on match");
    // Standard stats also update
    assert_eq!(node.visits(), 1);
    assert_eq!(node.total_reward(), 1.0);

    // 2. Trace contains NO match
    let trace_no_match = vec![TestAction(2), TestAction(3)];
    policy.update_stats(&mut node, 0.0, Some(&trace_no_match));

    // RAVE stats should NOT update
    assert_eq!(
        node.rave_visits(),
        1,
        "RAVE visits should NOT increment on mismatch"
    );
    assert_eq!(node.rave_value(), 1.0, "RAVE value should stay same");
    // Standard stats update
    assert_eq!(node.visits(), 2);
    assert_eq!(node.total_reward(), 1.0); // 1.0 + 0.0

    // 3. Trace is None
    policy.update_stats(&mut node, 1.0, None);
    // RAVE stats should NOT update
    assert_eq!(node.rave_visits(), 1);
    // Standard stats update
    assert_eq!(node.visits(), 3);
    assert_eq!(node.total_reward(), 2.0);
}

#[test]
fn test_rave_weight_clamping() {
    let p1 = RavePolicy::new(1.5);
    assert_eq!(p1.rave_weight, 1.0, "Should clamp to 1.0");

    let p2 = RavePolicy::new(-0.5);
    assert_eq!(p2.rave_weight, 0.0, "Should clamp to 0.0");
}
