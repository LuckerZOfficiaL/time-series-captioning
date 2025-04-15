
  time_steps = np.array([
      285654436, 396618, 38416, 14322, 37354,
      720236, 33842, 233506, 6435, 7416, 48728
  ])

  # Reduce Air Quality weight by half
  adjusted_steps = time_steps.copy()
  adjusted_steps[0] = adjusted_steps[0] / 2

  total_samples = 20000
  min_samples = 500

  # Step 1: Assign minimum samples
  n_datasets = len(datasets)
  assigned = np.full(n_datasets, min_samples)
  remaining = total_samples - min_samples * n_datasets

  # Step 2: Proportional distribution
  weights = adjusted_steps / adjusted_steps.sum()
  proportional = np.floor(weights * remaining).astype(int)

  # Step 3: Final allocation
  final_samples = assigned + proportional

  # Step 4: Fix rounding errors
  while final_samples.sum() < total_samples:
      final_samples[np.argmax(weights)] += 1
  while final_samples.sum() > total_samples:
      final_samples[np.argmax(final_samples)] -= 1

  # Print result
  for name, samples in zip(datasets, final_samples):
      print(f"{name:<16} {samples}")