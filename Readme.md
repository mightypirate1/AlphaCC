# AlphaCC

Ever got beaten in Chinese Checkers, and really feel you need to cheat to get back at your opponent?

Fear not - AlphaCC is here!

## How do I run it?

Hopefully all is smooth, but some tinkering might be needed to get the deps in order. Please let us know about any issues, or if something should be added to the docs or to the `Makefile`!

##### Requirements:
To run this in your local environment, you need to make sure you have the following installed first:

Base reqirements:
- linux
- bash
- rustup (with build-essential or equivalent for your system)
- python3.11 (with venv and pip)
- docker (with docker compose)

Application/evaluation requirements:
- npm ([nvm](https://www.linode.com/docs/guides/how-to-install-use-node-version-manager-nvm/#install-nvm) is nice; get node >=22.0.0)
- git lfs (used to track the trained models)

##### Installation:
"In theory", all you need to do once requirements are in place is 
```sh
make install
```
which should:
- create a python venv in `.venv`
- install python requirements
- build the game engine
- build the webapp

To get the weights to be able to use a trained agent, you need git lfs!
```sh
git lfs install
git lfs pull
```

## Now what can I do with it?

#### Play or watch the pre-trained bots:
For now, the webapp is run "manually":
```
# in the first terminal
./run-app.sh redis

# in a second terminal
source .venv/bin/activate
./run-app.sh backend

# in a third terminal
./run-app.sh frontend
```
Go to `http://localhost:4200/` in your browser (tested on chrome and firefox).

#### Train your own bots:
You can train your own bots using docker compose

To start training, you can run e.g. `docker compose up --build`. Just build and run:
```sh
docker compose up --build
```

Edit the `docker-compose.yaml` to set settings as you like them `:-)`!

> Note: the docker build is not exactly optimized for size. The image is currently `~5GB`.


## What is AlphaCC

For now, this implements a 2-player version of the game [Chinese Checkers](https://en.wikipedia.org/wiki/Chinese_checkers)*.

(*) For simplicity, laziness, and memory-optimizations sake, the starting areas for the non-existing players are removed from the board.

---

## Contribute!
This project needs many things, don't hesitate to send a PR or get in touch if you wnat to contribute!

Also, see `Roadmap.md` for details and inspiration `:-)`

---
If you have some skill and too much time, hit me up and I'd gladly work with you!
- e-mail on [my github](https://www.github.com/mightypirate1/) to the left
- [linkedin](https://www.linkedin.com/in/martin-frisk-9674981ab/)
