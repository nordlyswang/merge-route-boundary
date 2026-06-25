#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${MRB_PROJECT_ROOT:-/root/rivermind-data/projects/merge-route-boundary}
DATA_ROOT=${MRB_DATA_ROOT:-/root/rivermind-data/datasets}
PROJECT_DATA=${MRB_PROJECT_DATA:-${PROJECT_ROOT}/data}

mkdir -p "${PROJECT_ROOT}"
mkdir -p "${DATA_ROOT}"

if [ -e "${PROJECT_DATA}" ] && [ -L "${PROJECT_DATA}" ]; then
  current_target="$(readlink -f "${PROJECT_DATA}")"
  expected_target="$(readlink -f "${DATA_ROOT}")"
  if [ "${current_target}" = "${expected_target}" ]; then
    echo "Project data symlink already exists: ${PROJECT_DATA} -> ${DATA_ROOT}"
    exit 0
  fi
  echo "ERROR: ${PROJECT_DATA} is a symlink to ${current_target}, expected ${expected_target}."
  exit 1
fi

if [ -e "${PROJECT_DATA}" ]; then
  if [ ! -d "${PROJECT_DATA}" ]; then
    echo "ERROR: ${PROJECT_DATA} exists and is not a directory. Refusing to overwrite."
    exit 1
  fi

  while IFS= read -r entry; do
    if [ ! -L "${entry}" ]; then
      echo "ERROR: ${PROJECT_DATA} contains non-symlink entry ${entry}. Refusing to overwrite."
      exit 1
    fi
  done < <(find "${PROJECT_DATA}" -mindepth 1 -maxdepth 1 -print)

  while IFS= read -r entry; do
    unlink "${entry}"
  done < <(find "${PROJECT_DATA}" -mindepth 1 -maxdepth 1 -type l -print)

  rmdir "${PROJECT_DATA}"
fi

ln -s "${DATA_ROOT}" "${PROJECT_DATA}"

echo "Project data symlink: ${PROJECT_DATA} -> ${DATA_ROOT}"
echo "Shared data root: ${DATA_ROOT}"
