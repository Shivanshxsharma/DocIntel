#!/bin/sh
# Builds the static part of the Vercel deployment into public/:
# the web app (pointed at the /api function) plus the benchmark results, so the
# Benchmark and Metrics pages load even before the function has warmed up.
set -e
rm -rf public
mkdir -p public
sed 's#<!-- docintel:config -->#<script>window.DOCINTEL_API = "/api";</script>#' frontend/index.html > public/index.html
cp eval/results/summary.json public/summary.json
cp eval/results/docintel_card.png public/docintel_card.png
echo "static site built in public/"
