## Tentative road-map:
The items in each list is very roughly in order of importance.

### Agents
1. Implement MCTS model (refine this point as needed)
2. Reward-shaping: some slight bonus for leaving home and for entering the end-zone - ideally subtracted from eventual reward. Make sure it's easy to toggle / scale over time!
3. Find/implement hex-convolutions if that's a thing. I feel it might be.

### Game logic
1. Sanity check game by playing it somehow.
2. Time limit for games
3. Fix win condition: seems a stalling tactic is to just not move out of the home, thus forcing draw.
4. Make move finding symmetrical; i.e. positions that are identical up to isomorphism should get the moves in the same order. This might affect MCTS and improve sample efficiency.

### Orchestration
1. Figure out how to parallelize, ideally some  `x * workers + 1 * trainer` setup.
2. Create a `docker-compose` rig that runs training.
3. Create a `docker-compose` rig that runs an the interface.
4. Add `redis` if needed for passing around data.

### Interface
1. Set up some kind of back-end that handles the game and interacts with http.
2. Build some kind of graphical interface.
3. Use the interface to show games that have been played during training.
4. Make it so that a human can play against an agent on the backend.
5. Write pyo3-stuff for checking whether there exists a legal move between a pair of coordinates.
