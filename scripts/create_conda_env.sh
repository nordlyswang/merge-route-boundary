#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/create_conda_env.sh [cuda|cpu] [--force]

Creates the project Conda environment with mamba when available, otherwise conda.
Existing environments are left untouched unless --force is supplied.
EOF
}

mode="${1:-}"
force="false"

if [[ "${mode}" == "-h" || "${mode}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${mode}" != "cuda" && "${mode}" != "cpu" ]]; then
  usage
  exit 2
fi

if [[ "${2:-}" == "--force" ]]; then
  force="true"
elif [[ -n "${2:-}" ]]; then
  usage
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available on PATH." >&2
  exit 1
fi

solver="conda"
if command -v mamba >/dev/null 2>&1; then
  solver="mamba"
fi

if [[ "${mode}" == "cuda" ]]; then
  env_file="environment.yml"
  env_name="moe"
else
  env_file="environment.cpu.yml"
  env_name="moe-cpu"
fi

if conda env list | awk '{print $1}' | grep -Fxq "${env_name}"; then
  if [[ "${force}" != "true" ]]; then
    echo "Environment '${env_name}' already exists. Re-run with --force to rebuild."
    exit 0
  fi
  echo "Removing existing environment '${env_name}'..."
  "${solver}" env remove -n "${env_name}" -y
fi

echo "Creating '${env_name}' from ${env_file} with ${solver}..."
"${solver}" env create -f "${env_file}"
echo "Done. Activate with: conda activate ${env_name}"
