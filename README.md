# arboriter-mcts

[![Crates.io](https://img.shields.io/crates/v/arboriter-mcts.svg)](https://crates.io/crates/arboriter-mcts)
[![Documentation](https://docs.rs/arboriter-mcts/badge.svg)](https://docs.rs/arboriter-mcts)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://github.com/mrorigo/arboriter-mcts/workflows/CI/badge.svg)](https://github.com/mrorigo/arboriter-mcts/actions)

A flexible, well-documented Monte Carlo Tree Search (MCTS) implementation for Rust, built for game AI and decision-making processes.

## Features

- 🧰 **Generic implementation** that works with any game or decision process
- 🔄 **Multiple selection policies** (UCB1, UCB1-Tuned, PUCT) for optimal exploration/exploitation
- 🎲 **Customizable simulation strategies** to match your domain knowledge
- 📊 **Detailed search statistics and visualization** for debugging and analysis
- 🧪 **Comprehensive test suite** ensuring correctness and reliability
- 📝 **Thorough documentation** with examples for easy integration

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
arboriter-mcts = "0.1.0"
```

## Basic Usage

Here's a simple example of how to use the library:

```rust
use arboriter_mcts::{MCTS, MCTSConfig, GameState};

// Implement GameState for your game
impl GameState for MyGame {
    type Action = MyAction;
    type Player = MyPlayer;
    
    // Return legal actions from the current state
    fn get_legal_actions(&self) -> Vec<Self::Action> { /* ... */ }
    
    // Apply an action and return the new state
    fn apply_action(&self, action: &Self::Action) -> Self { /* ... */ }
    
    // Check if the game is over
    fn is_terminal(&self) -> bool { /* ... */ }
    
    // Get the result (0.0 = loss, 0.5 = draw, 1.0 = win)
    fn get_result(&self, for_player: &Self::Player) -> f64 { /* ... */ }
    
    // Get the current player
    fn get_current_player(&self) -> Self::Player { /* ... */ }
}

// Create a configuration for the search
let config = MCTSConfig::default()
    .with_exploration_constant(1.414)
    .with_max_iterations(10_000);

// Create the MCTS searcher with initial state
let mut mcts = MCTS::new(initial_state, config);

// Find the best action
let best_action = mcts.search()?;

// Get search statistics
println!("{}", mcts.get_statistics().summary());
```

## Running the Examples

The repository includes complete examples for common games that demonstrate the MCTS algorithm in action:

- **Tic-Tac-Toe**: A simple 3x3 game where you can play against the AI
- **Connect Four**: A more complex 7x6 game with stronger tactical elements

To run the examples:

```bash
# Play Tic-Tac-Toe against the AI
cargo run --example tic_tac_toe

# Play Connect Four against the AI
cargo run --example connect_four
```

## How MCTS Works

Monte Carlo Tree Search combines tree search with random sampling to find optimal decisions:

1. **Selection**: Starting from the root, select successive child nodes down to a leaf node using a selection policy that balances exploration and exploitation.

2. **Expansion**: If the leaf node is not terminal and has untried actions, create one or more child nodes by applying those actions.

3. **Simulation**: From the new node, simulate a game to completion using a default policy (often random play).

4. **Backpropagation**: Update the statistics (visit counts and rewards) for all nodes in the path from the selected node to the root.

This process is repeated many times, gradually building a tree of game states and improving the value estimates for each action.

## Advanced Customization

You can customize all aspects of the MCTS algorithm to match your specific needs:

```rust
let mut mcts = MCTS::new(initial_state, config)
    // Use UCB1 for selection with custom exploration constant
    .with_selection_policy(UCB1Policy::new(1.414))
    
    // Use domain-specific heuristics for simulation
    .with_simulation_policy(HeuristicPolicy::new(my_heuristic_fn))
    
    // Use a weighted backpropagation policy
    .with_backpropagation_policy(WeightedPolicy::new(0.5));

// Configure time-based search limits
let action = mcts.search_for_time(Duration::from_secs(5))?;
```

## Documentation

For detailed documentation and API reference, visit [docs.rs/arboriter-mcts](https://docs.rs/arboriter-mcts).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This crate is licensed under the MIT license. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on the [arboriter](https://github.com/mrorigo/arboriter) tree traversal primitive
- Inspired by Tyler Glaiel's blog post on tree traversal primitives
- Influenced by successful MCTS implementations in AlphaGo and other game AI systems