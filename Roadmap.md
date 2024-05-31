## Tentative road-map:

### 1. Interface
1. Extend interface to allow play vs AI
2. Extend interface to view tournament games.

### 2. Tournaments
1. Find a cleaner way of doing it.
2. Split `TournamentRuntime` into two classes 
3. There's a bug where the `NNService` gets called on a deallocated channel, right after a tournament has concluded (according to the counter). It needs to be fixed so we can know the tournament is correct.
4. Incorporate win-rate information into training flow.

### 3. Board bug
1. A rare bug exists where a `Board` fails to deserialize, causing workers to crash. Track down the cause and fix it!

### 3. Optimize Docker images
1. Move the engine-backend out of the python code.
2. Figure out how to make uv install only dependencies.
3. Move the venv over to the final stage which gets the remainder of the code and installs it.

## REFACTORING thoughts
1. If we don't use both mask and reverse mask (for nn training) we get rid of one of them.
2. Now that `Move` is as central as `Board`, an `Agent`'s `choose_move` should surely output a `Move` and not an index for one..?