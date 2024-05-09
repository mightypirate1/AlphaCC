# AlphaCC

Ever got beaten in Chinese Checkers, and really feel you need to cheat to get back at your opponent?

Fear not - AlphaCC is here*!

(*) Rather, it's on it's way - see `Contribute!` below.

## How do I run it?

The intended way of running this project is Docker/docker-compose, but that has not yet been written (see `Contribute!` below). Currentyl, just run as is:

##### Requirements:
To run this in your local environment, you need to make sure you have the following installed first:

- linux
- bash
- rustup (with build-essential or equivalent for your system)
- python3.11 (with venv)
- pip
- docker (with docker compose)

The the game interface runs as a webapp and requires:
- npm
- angular/cli

##### Installation:
"In theory", all you need to do once requirements are in place is `make install`. You should now have a virtual environment with everyhing you need installed in it.

This is enough to start coding.

To start training, you can run e.g. `docker compose up --build`. Edit the `docker-compose.yaml` to your preferences

## Now what can I do with it?

For now, this implements a 2-player version of the game [Chinese Checkers](https://en.wikipedia.org/wiki/Chinese_checkers)*.

(*) For simplicity, laziness, and memory-optimizations sake, the starting areas for the non-existing players are removed from the board.

---

## Contribute!
This project needs many things:
- RL improvements (tournament to decide which nn to keep training, and much more)
- Game backend improvements
- Webinterface for monitoring training and playing the game
- Etc etc

See `Roadmap.md` for details and inspiration :-)

---
If you have some skill and too much time, hit me up and I'd gladly work with you!
- e-mail on [my github](https://www.github.com/mightypirate1/) to the left
- [linkedin](https://www.linkedin.com/in/martin-frisk-9674981ab/)
