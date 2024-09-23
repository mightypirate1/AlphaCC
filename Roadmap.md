## Tentative road-map:

### 1. Containerization
1. Make the whole rig run with a single `docker compose up ...`

### 1. Interface
1. Extend interface to view tournament games.
2. More game-modes; e.g. player-vs-player, ai-vs-ai etc
3. Game forking; when stepping back to a previous move, one can fork into a separate game (we could track this whole tree!)

### 2. Tournaments
1. Find a cleaner way of doing it.
2. Split `TournamentRuntime` into two classes 
3. Incorporate win-rate information into training flow.

### 3. Board bug
1. A rare bug exists where a `Board` fails to deserialize, causing workers to crash. Track down the cause and fix it!

## REFACTORING thoughts
1. If we don't use both mask and reverse mask (for nn training) we get rid of one of them.
