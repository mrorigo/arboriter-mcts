use arboriter_mcts::{
    game_state::{Action, Player},
    policy::selection::{SelectionPolicy, UCB1TunedPolicy},
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
        0.0
    }

    fn get_current_player(&self) -> Self::Player {
        self.player.clone()
    }

    fn simulate_random_playout(&self, _for_player: &Self::Player) -> (f64, Vec<Self::Action>) {
        (0.0, Vec::new())
    }
}

#[test]
fn test_ucb1_tuned_selection_bias() {
    let state = TestGameState {
        terminal: false,
        actions: vec![TestAction(0), TestAction(1)],
        player: TestPlayer(1),
    };

    // Root node
    let mut root = MCTSNode::new(state.clone(), None, None, 0);
    // Two children
    // Child 0: High variance (alternating wins/losses)
    // Child 1: Low variance (constant draws)
    let child0 = MCTSNode::new(state.clone(), Some(TestAction(0)), Some(TestPlayer(1)), 1);
    let child1 = MCTSNode::new(state.clone(), Some(TestAction(1)), Some(TestPlayer(1)), 1);

    root.children.push(child0);
    root.children.push(child1);

    // Setup stats manually
    let visits = 100;
    root.visits
        .store(visits * 2, std::sync::atomic::Ordering::Relaxed);

    // Child 0: 50 wins (1.0), 50 losses (0.0). Avg = 0.5.
    // Sum = 50. SumSq = 50 (1^2 * 50 + 0^2 * 50).
    // Variance = (50/100) - 0.5^2 = 0.5 - 0.25 = 0.25.
    root.children[0]
        .visits
        .store(visits, std::sync::atomic::Ordering::Relaxed);
    // Use proper atomic addition or just plain store since we are in test setup
    // But store takes u64 representation of f64? No, MCTSNode stores total_reward as AtomicU64 (bits of f64)
    // Wait, MCTSNode uses helper methods.

    // We can use the helper methods directly
    for _ in 0..50 {
        root.children[0].add_reward(1.0);
        root.children[0].add_squared_reward(1.0);
    }
    for _ in 0..50 {
        root.children[0].add_reward(0.0);
        root.children[0].add_squared_reward(0.0);
    }

    // Child 1: 100 draws (0.5). Avg = 0.5.
    // Sum = 50. SumSq = 25 (0.5^2 * 100 = 0.25 * 100 = 25).
    // Variance = (25/100) - 0.5^2 = 0.25 - 0.25 = 0.0.
    // Low variance!
    for _ in 0..100 {
        root.children[1].add_reward(0.5);
        root.children[1].add_squared_reward(0.5 * 0.5);
    }
    // Need to increment visits manually or via update?
    // add_reward doesn't increment visits.
    // MCTSNode doesn't expose raw visit setting easily?
    // It has `visits: AtomicU64`. Public? Yes.
    root.children[0]
        .visits
        .store(100, std::sync::atomic::Ordering::Relaxed);
    root.children[1]
        .visits
        .store(100, std::sync::atomic::Ordering::Relaxed);

    // Verify stats
    assert_eq!(root.children[0].value(), 0.5);
    assert_eq!(root.children[1].value(), 0.5);

    // Child 0 variance term: ~0.25 + exploration
    // Child 1 variance term: ~0.0 + exploration
    // UCB1-Tuned adds min(0.25, variance + ...) to exploration.
    // The variance term is higher for Child 0.
    // So Child 0 should have higher UCB value?
    // Formula: value + C * sqrt(ln(N)/n * min(1/4, V + ...))
    // V_0 = 0.25. V_1 = 0.0.
    // So Child 0 gets bigger boost.

    // Wait, UCB1-Tuned favors HIGHER variance?
    // "An improved version of UCB1 that takes into account the variance"
    // Usually it uses variance to BOUND the exploration.
    // If variance is LOW, we are more confident, so we explore LESS?
    // UCB1-Tuned upper bound: mean + sqrt(ln N / n * min(1/4, V + ...))
    // If V is small, the term is small -> LESS optimistic.
    // If V is large, the term is larger -> MORE optimistic (explore uncertain arms).
    // So Child 0 (high variance) should be selected over Child 1 (low variance).

    let policy = UCB1TunedPolicy::new(1.0);
    let best_idx = policy.select_child(&root);

    assert_eq!(
        best_idx, 0,
        "Should select high variance child when means are equal"
    );
}
