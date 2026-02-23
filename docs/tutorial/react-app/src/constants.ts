export const MM = {
  vocab_size: 6400,
  hidden_size: 512,
  num_heads: 8,
  num_kv_heads: 2,
  head_dim: 64,
  num_layers: 8,
  intermediate_size: 1408,
  rope_base: 1e6,
  max_pos: 32768,
  rms_norm_eps: 1e-5,
  n_routed_experts: 4,
  num_experts_per_tok: 2,
  n_shared_experts: 1,
  aux_loss_alpha: 0.01,
  yarn_factor: 16,
  yarn_orig_max: 2048,
  yarn_beta_fast: 32,
  yarn_beta_slow: 1,
} as const;

export const COLORS = [
  '#4f46e5', '#7c3aed', '#06b6d4', '#10b981', '#f59e0b',
  '#ef4444', '#ec4899', '#8b5cf6', '#14b8a6', '#f97316',
] as const;
