extern crate criterion;

use arboriter_mcts::tree::MCTSNode;
use arboriter_mcts::{Action, GameState, MCTSConfig, Player, MCTS};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
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

        group.bench_with_input(BenchmarkId::new("branching_factor", bf), bf, |b, &_| {
            b.iter(|| {
                let mut mcts = MCTS::new(initial_state.clone(), config.clone());
                black_box(mcts.search())
            })
        });
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
            },
        );
    }

    // Test different branching factors with and without node pool
    for bf in [2, 3, 5].iter() {
        // Constant depth = 4 for reasonable benchmarks
        let max_depth = 4;
        let initial_state = BenchGameState::new(*bf, max_depth);

        // Configuration without node pool
        let config_no_pool = MCTSConfig::default()
            .with_exploration_constant(1.414)
            .with_max_iterations(1000);

        // Configuration with node pool
        let config_with_pool = MCTSConfig::default()
            .with_exploration_constant(1.414)
            .with_max_iterations(1000)
            .with_node_pool_config(1000, 500);

        // Benchmark without node pool
        group.bench_with_input(BenchmarkId::new("no_pool/branching", bf), bf, |b, &_| {
            b.iter(|| {
                let mut mcts = MCTS::new(initial_state.clone(), config_no_pool.clone());
                black_box(mcts.search())
            })
        });

        // Benchmark with node pool
        group.bench_with_input(BenchmarkId::new("with_pool/branching", bf), bf, |b, &_| {
            b.iter(|| {
                let mut mcts = MCTS::with_node_pool(
                    initial_state.clone(),
                    config_with_pool.clone(),
                    1000, // Initial pool size
                    500,  // Chunk size (not used in new implementation)
                );
                black_box(mcts.search())
            })
        });
    }

    // Test different iteration counts with and without node pool
    let iterations = [500, 2000, 5000];
    for &iter_count in &iterations {
        let initial_state = BenchGameState::new(3, 4); // Fixed branching and depth

        // Configuration without node pool
        let config_no_pool = MCTSConfig::default()
            .with_exploration_constant(1.414)
            .with_max_iterations(iter_count);

        // Configuration with node pool
        let config_with_pool = MCTSConfig::default()
            .with_exploration_constant(1.414)
            .with_max_iterations(iter_count)
            .with_node_pool_config(2000, 500);

        // Benchmark without node pool
        group.bench_with_input(
            BenchmarkId::new("no_pool/iterations", iter_count),
            &iter_count,
            |b, &_| {
                b.iter(|| {
                    let mut mcts = MCTS::new(initial_state.clone(), config_no_pool.clone());
                    black_box(mcts.search())
                })
            },
        );

        // Benchmark with node pool
        group.bench_with_input(
            BenchmarkId::new("with_pool/iterations", iter_count),
            &iter_count,
            |b, &_| {
                b.iter(|| {
                    let mut mcts = MCTS::with_node_pool(
                        initial_state.clone(),
                        config_with_pool.clone(),
                        2000, // Initial pool size
                        500,  // Chunk size (not used in new implementation)
                    );
                    black_box(mcts.search())
                })
            },
        );
    }

    // Test sequential searches to demonstrate node recycling benefits
    // This benchmark performs multiple searches with the same MCTS instance
    {
        // Use a smaller depth to ensure we have actions
        let initial_state = BenchGameState::new(3, 3); // Smaller depth, same branching factor
        let search_iterations = 300; // Each search does this many iterations
        let search_count = 5; // Number of sequential searches to perform

        // Configuration without node pool
        let config_no_pool = MCTSConfig::default()
            .with_exploration_constant(1.414)
            .with_max_iterations(search_iterations);

        // Configuration with node pool
        let config_with_pool = MCTSConfig::default()
            .with_exploration_constant(1.414)
            .with_max_iterations(search_iterations)
            .with_node_pool_config(2000, 500);

        // Benchmark sequential searches without node pool
        group.bench_function("sequential_searches_no_pool", |b| {
            b.iter(|| {
                // Create a new MCTS instance
                let mut mcts = MCTS::new(initial_state.clone(), config_no_pool.clone());

                // Perform multiple searches
                for _ in 0..search_count {
                    // Do the search, ignoring errors (for benchmark purposes)
                    let _ = mcts.search();

                    // For benchmarking purposes, make sure we have legal actions for next search
                    if mcts.root.unexpanded_actions.is_empty() && mcts.root.children.is_empty() {
                        // Reset to fresh state for next search
                        mcts.root = MCTSNode::new(initial_state.clone(), None, None, 0);
                    }
                }
            })
        });

        // Benchmark sequential searches with node pool
        group.bench_function("sequential_searches_with_pool", |b| {
            b.iter(|| {
                // Create a new MCTS instance with node pool
                let mut mcts = MCTS::with_node_pool(
                    initial_state.clone(),
                    config_with_pool.clone(),
                    2000, // Initial pool size
                    500,  // Chunk size (not used in new implementation)
                );

                // Perform multiple searches
                for i in 0..search_count {
                    // Do the search, ignoring errors (for benchmark purposes)
                    let _ = mcts.search();

                    // For benchmarking purposes, make sure we have legal actions for next search
                    if mcts.root.unexpanded_actions.is_empty() && mcts.root.children.is_empty() {
                        // Reset to fresh state for next search
                        mcts.root = MCTSNode::new(initial_state.clone(), None, None, 0);
                    }
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_mcts_search);
criterion_main!(benches);
