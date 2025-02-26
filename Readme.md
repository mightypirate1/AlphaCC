# AlphaCC

Ever got beaten in Chinese Checkers, and really feel you need to cheat to get back at your opponent?

Fear not - AlphaCC is here!

## What is it?

AlphaCC is a 2-player version of the game [Chinese Checkers](https://en.wikipedia.org/wiki/Chinese_checkers)*, together with:
- an [Alpha-Zero](https://arxiv.org/abs/1712.01815) style RL algorithm for training AI agents to play the game.
- a webapp to play against the bots! (big shout-out  to [Lucas Kylberg](https://github.com/lukac-k) for making the app happen!)

(*) For simplicity, laziness, and memory-optimizations sake, the starting areas for the non-existing players are removed from the board.

## How do I run it?

Hopefully all is smooth, but some tinkering might be needed to get the deps in order. Please let us know about any issues, or if something should be added to the docs or to the `Makefile`!

### Requirements:
To run this in your local environment, you need to make sure you have the following installed first:

Base reqirements:
- [git-lfs](https://git-lfs.com/) (or `sudo apt install git-lfs`)
- [docker](https://docs.docker.com/engine/install/)
- [docker compose v2](https://docs.docker.com/compose/install/linux/)
- make

Dev requirements:
- linux
- bash
- rustup (with build-essential or equivalent for your system)
- python3.12 (with venv and pip)
- npm ([nvm](https://www.linode.com/docs/guides/how-to-install-use-node-version-manager-nvm/#install-nvm) is nice; get node >=22.0.0)

## Now what can I do with it?

### Play against pre-trained bots:

To get the weights to be able to use a trained agent, you need git lfs!
```sh
git lfs install  # if you didn't already do this on your computer
git lfs pull     # fetches pre-trained models
```

Launch the webapp with
```sh
make build-and-run-webapp
```
> if your user is not a member of the `docker` group, you should consider [adding it](https://docs.docker.com/engine/install/linux-postinstall/), or alternatively running the previous command with `sudo`.

Go to `http://localhost:8080/` in your browser (tested on chrome and firefox) to play!

### Train your own bots:

You can train your own bots using docker compose.

The defaults provided are the ones used to train the default size-9 bots included. It takes a long time to train, so if you want something faster, you will have to change the parameters in `docker-compose.training.yaml` (see below).

To start training, you can run
```sh
docker compose -f docker-compose.training.yaml up --build
```

You can track the progress via the terminal, and the tensorboard at `http://localhost:6006/`.

#### Change training parameters
Edit the `docker-compose.training.yaml` to set settings as you like them `:-)`!

> If you are just trying this out and want to see some results in a reasonable amount of time, see `Reducing size` below.

To do more advanced settings, you will need to understand RL and to some extent MCTS. You will also have to read some code. For settings related to rollouts and training respectively, look in
- `alpha_cc/entrypoints/worker_thread.py`
- `alpha_cc/entrypoints/trainer_thread.py`

respectively.

##### Reducing size

Typically, you might want to change size of the board for faster training (supported sizes are `5`, `7`, `9`).

When reducing size, you might want to change (values in parentheses are the ones I've used for the different sizes):

`worker`:
- `size` `(5, 7, 9)`
- lower `--n-rollouts` `(200, 500, 1000)`
- lower `--max-game-length` `(200, 200, 300)`

`trainer`:
- `size` `(5, 7, 9)`
- lower `--n-train-samples` `(2000, 3000, 5000)`
- lower `--replay-buffer-size` `(5000, 15000, 25000)`

`nn-service`:
- `size` `(5, 7, 9)`

## Development
### Installation:
"In theory", all you need to do once requirements are in place is 
```sh
make install
```
which should:
- create a python venv in `.venv`
- install python requirements
- build the game engine
- build the webapp

For development, you probably want to run things in the terminal:
```sh
# in the first terminal
./run-app.sh redis

# in a second terminal
source .venv/bin/activate
./run-app.sh backend

# in a third terminal
./run-app.sh frontend
```
> the frontend is then at `http://localhost:4200`

---

## Contribute!
This project needs many things, don't hesitate to send a PR or get in touch if you wnat to contribute!

Also, see `Roadmap.md` for details and inspiration `:-)`

---
If you have some skill and too much time, hit me up and I'd gladly work with you!
- e-mail on [my github](https://www.github.com/mightypirate1/) to the left
- [linkedin](https://www.linkedin.com/in/martin-frisk-9674981ab/)
