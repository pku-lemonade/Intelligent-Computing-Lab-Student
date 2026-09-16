from __future__ import annotations

from microllm.distributed.af_runtime import AFEvent, AFStage, AFTask, AFTrace


class AFPipeline:
    def schedule(self, tasks: list[AFTask]) -> AFTrace:
        by_id = {task.task_id: task for task in tasks}
        if len(by_id) != len(tasks):
            raise ValueError("task ids must be unique")
        finish_times: dict[str, float] = {}
        resource_available = {"attention": 0.0, "ffn": 0.0, "link": 0.0}
        events: list[AFEvent] = []

        for task in sorted(tasks, key=lambda item: (item.layer_id, item.flow_id, _stage_order(item.stage))):
            missing = [dependency for dependency in task.depends_on if dependency not in finish_times]
            if missing:
                raise ValueError(f"task {task.task_id} has unresolved dependencies: {missing}")
            dependencies = [finish_times[dependency] for dependency in task.depends_on]
            if task.stage == AFStage.ATTENTION and task.layer_id > 0:
                previous = f"{task.flow_id}:layer{task.layer_id - 1}:to_attention"
                if previous not in finish_times:
                    raise ValueError(f"task {task.task_id} requires {previous}")
                dependencies.append(finish_times[previous])
            resource = _resource(task.stage)
            start = max([resource_available[resource], *dependencies])
            end = start + task.duration_s
            resource_available[resource] = end
            finish_times[task.task_id] = end
            events.append(AFEvent(task.task_id, task.flow_id, task.layer_id, task.stage, resource, start, end))

        completion = max(finish_times.values(), default=0.0)
        return AFTrace(events=tuple(events), completion_s=completion)


def _stage_order(stage: AFStage) -> int:
    return {
        AFStage.ATTENTION: 0,
        AFStage.TO_FFN: 1,
        AFStage.FFN: 2,
        AFStage.TO_ATTENTION: 3,
    }[stage]


def _resource(stage: AFStage) -> str:
    if stage == AFStage.ATTENTION:
        return "attention"
    if stage == AFStage.FFN:
        return "ffn"
    return "link"
