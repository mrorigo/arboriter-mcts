#[macro_use]
extern crate criterion;

use criterion::{black_box, Criterion, BenchmarkId};
use arboriter_mcts::{MCTS, MCTSConfig, GameState, Action, Player};
use std::time::Duration;

// Simple game state for benchmarking
#[derive(Clone, Debug)]
struct BenchGameState {
    depth: usize,
    branching_factor: usize,
    max_depth: usize,
    player: BenchPlayer,
}

impl BenchGameState {
    fn new(branching_factor: usize, max_depth: usize) -> Self {
        BenchGameState {
            depth: 0,
            branching_factor,
            max_depth,
            player: BenchPlayer(0),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BenchAction(usize);

impl Action for BenchAction {
    fn id(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BenchPlayer(usize);

impl Player for BenchPlayer {}

impl GameState for BenchGameState {
    type Action = BenchAction;
    type Player = BenchPlayer;
    
    fn get_legal_actions(&self) -> Vec<Self::Action> {
        // Terminal nodes have no actions
        if self.depth >= self.max_depth {
            return vec![];
        }
        
        // Otherwise generate actions based on branching factor
        (0..self.branching_factor).map(BenchAction).collect()
    }
    
    fn apply_action(&self, action: &Self::Action) -> Self {
        let mut new_state = self.clone();
        new_state.depth += 1;
        // Use action to affect state to prevent optimizer from removing it
        new_state.player = BenchPlayer((self.player.0 + action.0) % 2);
        new_state
    }
    
    fn is_terminal(&self) -> bool {
        self.depth >= self.max_depth
    }
    
    fn get_result(&self, for_player: &Self::Player) -> f64 {
        // Simple result function based on depth and player
        if self.depth == self.max_depth {
            if self.player.0 == for_player.0 {
                0.75 // Win-ish
            } else {
                0.25 // Loss-ish
            }
        } else {
            0.5 // Draw for non-terminal
        }
    }
    
    fn get_current_player(&self) -> Self::Player {
        self.player.clone()
    }
}

fn bench_mcts_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_search");
    group.measurement_time(Duration::from_secs(10));
    
    // Test different branching factors
    for bf in [2, 3, 5].iter() {
        // Constant depth = 4 for reasonable benchmarks
        let max_depth = 4;
        let initial_state = BenchGameState::new(*bf, max_depth);
        
        let config = MCTSConfig::default()
            .with_exploration_constant(1.414)
            .with_max_iterations(1000);
        
        group.bench_with_input(
            BenchmarkId::new("branching_factor", bf), 
            bf, 
            |b, &_| {
                b.iter(|| {
                    let mut mcts = MCTS::new(initial_state.clone(), config.clone());
                    black_box(mcts.search())
                })
            }
        );
    }
    
    // Test different iteration counts
    let iterations = [100, 1000, 5000];
    for &iter_count in &iterations {
        let initial_state = BenchGameState::new(2, 4); // Fixed branching and depth
        
        let config = MCTSConfig::default()
            .with_exploration_constant(1.414)
            .with_max_iterations(iter_count);
        
        group.bench_with_input(
            BenchmarkId::new("iterations", iter_count), 
            &iter_count, 
            |b, &_| {
                b.iter(|| {
                    let mut mcts = MCTS::new(initial_state.clone(), config.clone());
                    black_box(mcts.search())
                })
            }
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_mcts_search);
criterion_main!(benches);