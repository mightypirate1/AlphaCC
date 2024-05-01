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

The the game interface runs as a webapp and requires:
- npm
- angular/cli

##### Installation:
"In theory", all you need to do once requirements are in place is `make install`. Once that has run to completion and produced the happy news `Installation successful!`you are good to go!

## Now what can I do with it?

For now, this implements a 2-player version of the game [Chinese Checkers](https://en.wikipedia.org/wiki/Chinese_checkers)*.

(*) For simplicity, laziness, and memory-optimizations sake, the starting areas for the non-existing players are removed from the board.

Run / look at `demo.py` to see the main concepts in action - it's not much, but it's honest work! Hopefully it gives you an idea of how to get tinkering.

Again; feel free to contribute :-)

---

#### Low-level functionality:
If you want to interact with the game backend itself, here's showing the main functionality it provides:
```
 # In repo-root after successful installation:
 source .venv/bin/activate
 ipython

 > from alpha_cc.engine import Board
 > board = alpha_cc.Board(9)
 >
 > # These are all boards that are possible to reach through legal moves. You don't get to know what those moves are, but the action space is simply the index in the list `next_states`.
 > next_states = board.get_next_states()  # These are type Board too!
 >
 > # How the board looks under the hood, i.e. what your policy will get to see:
 > explicit_view_of_board_were_i_to_play_move_7 = next_states[7].get_matrix_from_perspective_of_current_player()
 >
 > # Some extra info is in:
 > game_info = board.get_game_info()
 > print(game_info.winner, game_info.current_player)
 0, 1
 > # No one won yet, and it's player 1's turn!
 >
 > # I don't know this game, so I think `7` looked just as good as any move I could think of!
 > board = board.perform_move(7)  # Plays move `7` for the current player, i.e. player 1.
 >
 > # Now I want to know what I did :-)
 > board.render()
  [1, 1, 1, 1, 0, 0, 0, 0, 0]
    [1, 1, 0, 0, 0, 0, 0, 0, 0]
      [1, 1, 1, 0, 0, 0, 0, 0, 0]
        [1, 0, 0, 0, 0, 0, 0, 0, 0]
          [0, 0, 0, 0, 0, 0, 0, 0, 0]
            [0, 0, 0, 0, 0, 0, 0, 0, 2]
              [0, 0, 0, 0, 0, 0, 0, 2, 2]
                [0, 0, 0, 0, 0, 0, 2, 2, 2]
                  [0, 0, 0, 0, 0, 2, 2, 2, 2]
Current player: 2
>
> # Put it back! :-)
> board = board.reset()  # Randomizes staring player
```
So... that's it. Not impressed? You could  `Contribute!` (see below).

## Contribute!
This project needs many things:
- RL
- Game backend improvements
- Webinterface for monitoring training and playing the game
- Etc etc

See `Roadmap.md` for details and inspiration :-)

---
If you have some skill and too much time, hit me up and I'd gladly work with you!
- e-mail on [my github](https://www.github.com/mightypirate1/) to the left
- [linkedin](https://www.linkedin.com/in/martin-frisk-9674981ab/)
