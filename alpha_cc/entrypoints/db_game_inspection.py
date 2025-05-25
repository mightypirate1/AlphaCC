import click

from alpha_cc.config import Environment
from alpha_cc.db import GamesDB


@click.command("alpha-cc-db-game-inspection")
@click.option("--redis-url", type=str, default=Environment.redis_host_main)
@click.option("--game-id", type=str, default=None)
def main(redis_url: str, game_id: str | None) -> None:
    db = GamesDB(host=redis_url)
    if game_id is None:
        game_id_keys = db.list_entries()
        for game_id in game_id_keys:
            dbgamestate = db.get_state(game_id)
            click.echo(f"Game ID: {game_id}: {len(dbgamestate._moves)} moves")
        return
    state = db.get_state(game_id)
    for board in state.boards:
        board.render()
