import click
import plotext as plt

from alpha_cc.training import ParamSchedule


@click.command("alpha-cc-param-schedule-eval")
@click.argument("schedule_str", type=str)
@click.option("--n", type=int, default=100)
def main(schedule_str: str, n: int) -> None:
    param_schedule = ParamSchedule.from_str(schedule_str)

    t_values = list(range(n))
    y_values = [param_schedule.as_float(t) for t in t_values]
    plt.clf()
    plt.plot(t_values, y_values, color=["red"])
    plt.title(f"Parameter Schedule: {schedule_str}")
    plt.xlabel("Time step (t)")
    plt.ylabel("Parameter value")
    plt.show()
