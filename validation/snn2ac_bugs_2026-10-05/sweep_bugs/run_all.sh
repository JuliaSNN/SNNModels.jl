#!/usr/bin/env bash
# Regenerate all data and figures of the sweep-bug validation.
# ENVS must point to a folder with one Julia environment per variant (Project.toml + Manifest.toml
# dev-linking the four packages):
#   unfixed   : SNNModels docs/sweep (779f5a8) = released 1.8.4 behaviour
#   unfixedIs : unfixed + BallAndStick Is/Id applied (scratch commit a9a9b61 on 779f5a8)
#   dtonly    : fix/sweep-bugs b042299 (Heun fix) + BallAndStick Is/Id (scratch 31a883d)
#   glonly    : 779f5a8 + gl in the exponential term (scratch e05a855) + Is/Id (scratch f2f8780)
#   fixed     : fix/sweep-bugs head
set -euo pipefail
ENVS=${ENVS:-/tmp/claude-1001/-home-aquaresi-Documents-Research/19c4b373-426f-4ac3-a89e-24e2b15bc645/scratchpad/envs}
cd "$(dirname "$0")"
for v in unfixed unfixedIs dtonly glonly fixed; do
    julia --project=$ENVS/$v measure_multicompartment.jl $v
done
for v in unfixed dtonly glonly fixed; do julia --project=$ENVS/$v measure_network.jl $v; done
for v in unfixed fixed; do julia --project=$ENVS/$v measure_network.jl $v 4321; done
for v in unfixed fixed; do julia --project=$ENVS/$v measure_other.jl $v; done
julia --project=$ENVS/fixed summarize_multicompartment.jl
julia --project=$ENVS/fixed summarize_network.jl
julia --project=$ENVS/fixed summarize_other.jl
julia --project=$ENVS/fixed plot_multicompartment.jl
julia --project=$ENVS/fixed plot_network.jl
julia --project=$ENVS/fixed plot_other.jl
