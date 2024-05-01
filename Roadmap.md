## Tentative road-map:
The items in each list is very roughly in order of importance.

### Agents
1. Implement MCTS model (refine this point as needed)
2. Reward-shaping: some slight bonus for leaving home and for entering the end-zone - ideally subtracted from eventual reward. Make sure it's easy to toggle / scale over time!
3. Find/implement hex-convolutions if that's a thing. I feel it might be.

### Game logic
1. Time limit for games
2. Fix win condition: seems a stalling tactic is to just not move out of the home, thus forcing draw.
3. Correct number of starting pieces! (currently 1 row too little)

### Orchestration
1. Figure out how to parallelize, ideally some  `x * workers + 1 * trainer` setup.
2. Create a `docker-compose` rig that runs training.
3. Create a `docker-compose` rig that runs an the interface.
4. Add `redis` if needed for passing around data.

### Interface
1. Build some kind of graphical interface.
2. Use the interface to show games that have been played during training.
3. Make it so that a human can play against an agent on the backend.

## REFACTORING thoughts
1. Once a real nn is in place, the `reward` folder can probably be dropped.
2. `Runtime` might be redundant soon. Atleast when the webapp takes form.
5. If we don't use both mask and reverse mask (for nn training) we get rid of one of them
6. If we do keep the `disallowed_states` thing; figure out a nicer way to implement it!
