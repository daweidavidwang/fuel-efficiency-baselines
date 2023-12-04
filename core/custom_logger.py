"""Example of using RLlib's debug callbacks.

Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

from typing import Dict
import argparse
import numpy as np

import ray
from ray import tune
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class CustomLoggerCallback(DefaultCallbacks):
    def on_episode_start(
            self,
            *,
            worker,
            base_env,
            policies,
            episode,
            env_index = None,
            **kwargs,
        ):
        pass

    def on_episode_step(
            self,
            *,
            worker,
            base_env,
            policies = None,
            episode,
            env_index= None,
            **kwargs,
        ):
        pass

    def on_episode_end(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        env_index = None,
        **kwargs,
    ):
        episode.custom_metrics["total_fuel"] = worker.env.veh.total_fuel
        episode.custom_metrics["saved_fuel"] = (worker.env.veh_const.total_fuel - worker.env.veh.total_fuel)/worker.env.veh_const.total_fuel
        # episode.custom_metrics["saved_fuel"] = (worker.env.const_benchmark_veh.total_fuel - worker.env.veh.total_fuel)/worker.env.const_benchmark_veh.total_fuel
        # episode.custom_metrics["fuel_efficency_index"] = worker.env.veh.total_fuel/(worker.env.veh.position-worker.env.start_location)
        # episode.custom_metrics["avg_speed"] = (worker.env.veh.position-worker.env.start_location)/worker.env.veh.total_running_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=2000)
    args = parser.parse_args()

    ray.init()
    trials = tune.run(
        "PG",
        stop={
            "training_iteration": args.num_iters,
        },
        config={
            "env": "CartPole-v0",
            "callbacks": MyCallbacks,
        },
        return_trials=True)

    # verify custom metrics for integration tests
    custom_metrics = trials[0].last_result["custom_metrics"]
    print(custom_metrics)
    assert "pole_angle_mean" in custom_metrics
    assert "pole_angle_min" in custom_metrics
    assert "pole_angle_max" in custom_metrics
    assert "num_batches_mean" in custom_metrics
    assert "callback_ok" in trials[0].last_result