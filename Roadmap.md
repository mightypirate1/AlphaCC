## Tentative road-map:
The items in each list is very roughly in order of importance.

### Optimize Docker images
1. Move the engine-backend out of the python code.
2. Figure out how to make uv install only dependencies.
3. Have a build-stage that prepares the venv with deps and engine installed.
4. Move the venv over to the final stage which gets the remainder of the code and installs it.

### Rust-rollouts
- Build a rollout-machine in rust using jit-compiled models
- If faster, make that the default

### Training
1. Lot's of weirdness currently since the tensorboard logging is done very unsystematically. Since sending data is currently not at all a bottleneck, we can probably send more data immediately, and have a single SummaryWriter in the `trainer_thread` that is passed to objects as needed that would make this much clearer. Full rewrite?

### Interface
1. Build some kind of graphical interface.
2. Use the interface to show games that have been played during training.
3. Make it so that a human can play against an agent on the backend.

## REFACTORING thoughts
1. `Runtime` might be redundant soon. Atleast when the webapp takes form.
2. If we don't use both mask and reverse mask (for nn training) we get rid of one of them
3. The division of responsibilities between Agent (in particular MCTSAgent ofc) and `worker_thread` is not ideal. E.g. how the trajectory is formed and manipulated on `on_game_end` is bad.
4. Now that `Move` is as central as `Board`, an `Agent`'s `choose_move` should surely output a `Move` and not an index for one..?