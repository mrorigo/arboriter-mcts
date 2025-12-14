use arboriter_mcts::{Action, GameState, MCTSConfig, Player, MCTS};

// Simple game state for testing visualization
#[derive(Clone, Debug)]
struct TestGame {
    depth: usize,
    max_depth: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TestAction(usize);

impl Action for TestAction {
    fn id(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TestPlayer(usize);

impl Player for TestPlayer {}

impl GameState for TestGame {
    type Action = TestAction;
    type Player = TestPlayer;

    fn get_legal_actions(&self) -> Vec<Self::Action> {
        if self.depth >= self.max_depth {
            vec![]
        } else {
            vec![TestAction(0), TestAction(1)]
        }
    }

    fn apply_action(&self, action: &Self::Action) -> Self {
        Self {
            depth: self.depth + 1,
            max_depth: self.max_depth,
        }
    }

    fn is_terminal(&self) -> bool {
        self.depth >= self.max_depth
    }

    fn get_result(&self, _for_player: &Self::Player) -> f64 {
        0.5
    }

    fn get_current_player(&self) -> Self::Player {
        TestPlayer(self.depth % 2)
    }
}

#[test]
fn test_tree_visualization() {
    // Create a small, fixed-depth game
    let game = TestGame {
        depth: 0,
        max_depth: 2,
    };

    let config = MCTSConfig::default().with_max_iterations(20); // Small number of iterations for testing

    let mut mcts = MCTS::new(game, config);

    // Run search to build a tree
    let _ = mcts.search();

    // Get tree visualization
    let tree_vis = mcts.visualize_tree();

    // Basic validation of the visualization
    assert!(tree_vis.contains("Root"));
    assert!(tree_vis.starts_with("Root"));
    assert!(tree_vis.contains("visits:"));
    assert!(tree_vis.contains("value:"));

    // Should contain indented levels
    assert!(tree_vis.contains("  TestAction"));

    // Print for inspection during test development
    // println!("{}", tree_vis);
}
