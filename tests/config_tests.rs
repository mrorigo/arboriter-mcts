use arboriter_mcts::{config::BestChildCriteria, MCTSConfig};
use std::time::Duration;

#[test]
fn test_config_builder_methods() {
    // Test that all builder methods correctly set their respective values
    let config = MCTSConfig::default()
        .with_exploration_constant(2.0)
        .with_max_iterations(5000)
        .with_max_time(Duration::from_secs(30))
        .with_max_depth(20)
        .with_transpositions(true)
        .with_best_child_criteria(BestChildCriteria::HighestValue);
    
    // Verify each setting was applied correctly
    assert_eq!(config.exploration_constant, 2.0);
    assert_eq!(config.max_iterations, 5000);
    assert_eq!(config.max_time, Some(Duration::from_secs(30)));
    assert_eq!(config.max_depth, Some(20));
    assert_eq!(config.use_transpositions, true);
    assert_eq!(config.best_child_criteria, BestChildCriteria::HighestValue);
}

#[test]
fn test_config_default_values() {
    // Test that default values are set correctly
    let config = MCTSConfig::default();
    
    // Default exploration constant should be sqrt(2)
    assert!((config.exploration_constant - 1.414).abs() < 0.001);
    assert_eq!(config.max_iterations, 10_000);
    assert_eq!(config.max_time, None);
    assert_eq!(config.max_depth, None);
    assert_eq!(config.use_transpositions, false);
    assert_eq!(config.best_child_criteria, BestChildCriteria::MostVisits);
}