"""Quick check of pair weight distribution and gradient balance."""
import pyarrow.feather as feather

pairs = feather.read_feather(
    "src/clesso_v2/output/VAS_20260310_092634/nn_export/pairs.feather"
)

within = pairs[pairs["is_within"] == 1]
between = pairs[pairs["is_within"] == 0]

print("=== Effective weight contribution to loss ===")
print(f"Within:  {len(within):>8d} pairs, total_w = {within.w.sum():.2f}"
      f"  ({within.w.sum() / pairs.w.sum() * 100:.1f}%)")
print(f"Between: {len(between):>8d} pairs, total_w = {between.w.sum():.2f}"
      f"  ({between.w.sum() / pairs.w.sum() * 100:.1f}%)")
print(f"Total:   {len(pairs):>8d} pairs, total_w = {pairs.w.sum():.2f}")
print()
print("=== Effect on gradient balance ===")
tw = pairs.w.sum()
bw = between.w.sum()
print(f"Alpha gets gradient from: within ({within.w.sum():.1f}) + between ({bw:.1f}) = {tw:.1f} total")
print(f"Beta  gets gradient from: between ({bw:.1f}) only")
print(f"Ratio of alpha signal to beta signal: {tw:.1f} / {bw:.1f} = {tw / bw:.1f}x")
print()
print("=== But additionally, beta gradients are diluted by 1/alpha ===")
print("With alpha ~50, dp/d_eta ~ exp(-eta)/alpha ~ 1/50")
print("So effective beta gradient is ~1/50 of the alpha gradient")
print(f"Combined disadvantage: {tw/bw:.1f}x weight ratio * 50x dilution = {tw/bw*50:.0f}x")
