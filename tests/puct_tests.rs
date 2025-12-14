use arboriter_mcts::{
    game_state::{Action, Player},
    policy::{
        expansion::ExpansionPolicy,
        selection::PUCTPolicy,
    },
    tree::MCTSNode,
    GameState, MCTSConfig, MCTS,
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

/// A deterministic expansion policy for testing
struct DeterministicExpansionPolicy {
    /// Index to select
    index: usize,
    /// Prior to assign
    prior: f64,
}

impl DeterministicExpansionPolicy {
    fn new(index: usize, prior: f64) -> Self {
        Self { index, prior }
    }
}

impl<S: GameState> ExpansionPolicy<S> for DeterministicExpansionPolicy {
    fn select_action_to_expand(&self, node: &MCTSNode<S>) -> Option<(usize, f64)> {
        if node.unexpanded_actions.is_empty() {
             return None;
        }
        // Always select the configured index (clamped to bounds)
        let idx = std::cmp::min(self.index, node.unexpanded_actions.len() - 1);
        Some((idx, self.prior))
    }

    fn clone_box(&self) -> Box<dyn ExpansionPolicy<S>> {
        Box::new(Self { index: self.index, prior: self.prior })
    }
}

#[test]
fn test_expansion_policy_sets_prior() {
    let state = TestGameState {
        terminal: false,
        actions: vec![TestAction(0), TestAction(1)],
        player: TestPlayer(1),
    };

    let config = MCTSConfig::default()
        .with_max_iterations(1);
    
    // We want to force expansion of action index 0 with prior 0.8
    let expansion_policy = DeterministicExpansionPolicy::new(0, 0.8);
    
    let mut mcts = MCTS::new(state, config)
        .with_expansion_policy(expansion_policy);

    // Expand root once
    // Currently MCTS::search runs full simulations. 
    // We can just call execute_iteration directly if it was public, 
    // or just run search for 1 iteration.
    
    // Let's run 1 iteration
    let _ = mcts.search();
    
    // Check root's child
    let root = mcts.root();
    assert_eq!(root.children.len(), 1, "Should have expanded 1 child");
    
    // Verify prior was set
    assert_eq!(root.children[0].prior(), 0.8, "Prior should be set by policy");
    
    // Now verify PUCT uses this prior
    let _puct = PUCTPolicy::new(1.0);
    // Root child 0 has 1 visit (from expansion/simulation)
    // Parent should have 1 visit.
    
    // To properly test prior influence, we need 2 children
    // Let's manually add another child with different prior and same stats
    // But MCTS struct is owning the root.
    
    // We can rely on MCTS API to expand another node?
    // We need to change the expansion policy first.
    // But we can't change policy on existing MCTS easily without rebuilding it?
    // MCTS::with_expansion_policy consumes self.
    
    // Actually, we can just test MCTSNode and PUCTPolicy in isolation for the selection logic
    // which is covered in selection_policy_tests.rs.
    // This test confirms that ExpansionPolicy -> MCTSNode propagation works.
}

#[test]
fn test_mcts_respects_expansion_policy() {
     let state = TestGameState {
        terminal: false,
        actions: vec![TestAction(0), TestAction(1), TestAction(2)],
        player: TestPlayer(1),
    };
    
    // Policy that always picks the LAST action (index 100 clamped) with prior 0.99
    let expansion_policy = DeterministicExpansionPolicy::new(100, 0.99);
    
    let config = MCTSConfig::default().with_max_iterations(1);
    let mut mcts = MCTS::new(state, config)
        .with_expansion_policy(expansion_policy);
        
    let _ = mcts.search();
    
    let root = mcts.root();
    assert_eq!(root.children.len(), 1);
    // Should have expanded the last action in the list (TestAction(2))
    // unexpanded_actions starts with [0, 1, 2]. 
    // MCTS expansion uses swap_remove.
    // If we pick index 2 (last), it removes index 2.
    // Child action should be TestAction(2).
    
    assert_eq!(root.children[0].action.as_ref().unwrap().0, 2);
    assert_eq!(root.children[0].prior(), 0.99);
}
