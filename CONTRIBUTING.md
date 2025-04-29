# Contributing to arboriter-mcts

Thank you for considering contributing to arboriter-mcts! This document provides guidelines and instructions to help you contribute effectively.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct: be respectful, considerate, and constructive in all interactions.

## How to Contribute

### Reporting Bugs

Bug reports are valuable contributions. To report a bug:

1. Check if the bug has already been reported in the Issues section
2. Use the bug report template if available
3. Include:
   - A clear title and description
   - Steps to reproduce the issue
   - Expected vs. actual behavior
   - Version information (Rust version, crate version)
   - Any relevant code samples

### Suggesting Enhancements

Feature suggestions are welcome! When suggesting enhancements:

1. Check if the enhancement has already been suggested
2. Provide a clear description of the proposed feature
3. Explain why this feature would be useful to most users
4. Consider how it integrates with existing functionality

### Pull Requests

When submitting a pull request:

1. Fork the repository and create your branch from `main`
2. If you've added code that should be tested, add tests
3. Ensure the test suite passes (`cargo test`)
4. Make sure your code follows the project's style guidelines (`cargo clippy`)
5. Update documentation for any new features or changes
6. Include a good description of what the PR does

## Development Process

### Setting up the Development Environment

1. Fork and clone the repository
2. Run `cargo build` to build the project
3. Run `cargo test` to ensure everything works

### Coding Standards

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Run `cargo fmt` before committing to ensure consistent formatting
- Run `cargo clippy` to catch common mistakes and non-idiomatic code
- Write documentation for all public items

### Testing

- Write unit tests for all new functionality
- Include integration tests where appropriate
- Documentation tests are encouraged for examples

## Architecture Overview

arboriter-mcts is structured around several key components:

- `mcts.rs`: The core MCTS algorithm implementation
- `policy/`: Different policies for selection, simulation, and backpropagation
- `tree.rs`: Tree data structures used by the algorithm
- `game_state.rs`: Traits defining the interface for games or decision processes

When contributing, consider where your changes fit into this architecture.

## License

By contributing to arboriter-mcts, you agree that your contributions will be licensed under the project's MIT license.