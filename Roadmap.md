## Tentative road-map:

### 1. Training improvements.
1. More sophisticated sampling of internal nodes.

### 2. Interface
1. Extend interface to view tournament games.
2. More game-modes; e.g. player-vs-player, ai-vs-ai etc
3. Game forking; when stepping back to a previous move, one can fork into a separate game (we could track this whole tree!)

### 3. Tournaments
1. Find a cleaner way of doing it.
2. Split `TournamentRuntime` into two classes 
3. Incorporate win-rate information into training flow.


## REFACTORING thoughts
1. If we don't use both mask and reverse mask (for nn training) we get rid of one of them.
