#!/bin/bash

REPO_NAME="AlphaCC"

#######
### Main functions!
#####

function target_env_delete () {
  if [[ "$(command -v deactivate)" == "deactivate" ]]; then
    deactivate
    echo "Deactivating virtual environment"
  fi
  if venv_exists; then
    echo "Deleting virtual environment"
    rm -rf .venv
  fi
}

function target_env_create () {
  target_env_delete
  python3 -m venv .venv --prompt $REPO_NAME
  . .venv/bin/activate && pip install -r requirements.txt
}

function target_build () {
  (venv_is_active || . .venv/bin/activate) && bash -c "cd alpha-cc && maturin develop"
}

function target_install () {
  (venv_exists || target_env_create) && (. .venv/bin/activate && target_build && target_test)
}

function target_test () {
  (venv_is_active || . .venv/bin/activate) && pytest tests
}

#######
### Helpers!
#####

function venv_exists () {
  [ -d .venv ] && return 0 || return 1
}

function venv_is_active () {
  if [[ $(echo $VIRTUAL_ENVIRONMENT | grep $REPO_NAME | wc -l) -gt 0 ]] \
  || [[ $(echo $VIRTUAL_ENV         | grep $REPO_NAME | wc -l) -gt 0 ]]; then
    return 0
  else
    return 1
  fi
}

#######
### Sanity checks before attempting anything!
#####

function from_repo_root_check () {
  if [[ "$(basename $PWD)" == "$REPO_NAME" ]]; then
    return 0
  else
    return 1
  fi
}
function print_usage () {
  echo "usage: ./build.sh <env-delete|env-create|env-activate|build|install|test>";
  exit 1;
}

if [[ $# -ne 1 ]]; then
  print_usage
fi

if from_repo_root_check; then
  :
else
  echo "Must be executed from repo $REPO_NAME root!"
fi

case $1 in
  env-delete)
    target_env_delete
  ;;
  env-create)
    target_env_create
  ;;
  env-activate)
    target_env_activate
  ;;
  build)
    target_build
  ;;
  test)
    target_test
  ;;
  install)
    target_install
  ;;
  *)
    print_usage
  ;;
esac
