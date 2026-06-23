# Review Checklist

- Does the change preserve the kappa-first public API?
- Does `import cosmock` work without optional dependencies?
- Are optional dependency errors clear and tied to the right extra?
- Are input shapes validated near the API boundary?
- Are stochastic tests seeded?
- Are generated maps, private data paths, and large arrays excluded from git?
- Did any scientific formula change without a targeted test or notebook
  comparison?

