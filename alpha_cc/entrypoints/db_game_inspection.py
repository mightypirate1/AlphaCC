import click

from alpha_cc.config import Environment
from alpha_cc.db import GamesDB


@click.command("alpha-cc-db-game-inspection")
@click.option("--redis-url", type=str, default=Environment.redis_host_main)
@click.option("--game-id", type=str, default=None)
def main(redis_url: str, game_id: str | None) -> None:
    db = GamesDB(host=redis_url)
    if game_id is None:
        game_ids = db.list_entries()
        click.echo(f"Game IDs: {game_ids}")
        return
    state = db.get_state(game_id)
    for board in state.boards:
        board.render()
