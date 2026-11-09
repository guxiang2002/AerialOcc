import time

from pytorch_lightning.callbacks import Callback


def _format_duration(seconds):
    seconds = max(int(seconds), 0)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or parts:
        parts.append(f"{hours}h")
    if minutes > 0 or parts:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


class TrainingTimeMonitor(Callback):
    """Print elapsed time and ETA during training."""

    def __init__(self, log_every_n_steps=100):
        super().__init__()
        self.log_every_n_steps = max(int(log_every_n_steps), 1)
        self.start_time = None
        self.start_step = 0

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.start_step = int(trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return

        global_step = trainer.global_step
        max_steps = trainer.max_steps

        if global_step <= 0:
            return

        session_steps = max(global_step - self.start_step, 0)
        if session_steps <= 0:
            return

        should_log = (
            session_steps == 1
            or global_step % self.log_every_n_steps == 0
            or (max_steps is not None and max_steps > 0 and global_step >= max_steps)
        )
        if not should_log:
            return

        elapsed = time.time() - self.start_time if self.start_time is not None else 0.0
        avg_step_time = elapsed / session_steps

        remaining_steps = 0
        progress = 0.0
        eta_seconds = 0.0
        if max_steps is not None and max_steps > 0:
            remaining_steps = max(max_steps - global_step, 0)
            progress = min(global_step / max_steps, 1.0)
            eta_seconds = remaining_steps * avg_step_time

        epoch = trainer.current_epoch + 1
        progress_msg = f"{progress * 100:.2f}%" if max_steps is not None and max_steps > 0 else "N/A"
        print(
            "\n[TimeMonitor] "
            f"epoch={epoch} "
            f"step={global_step}"
            + (f"/{max_steps}" if max_steps is not None and max_steps > 0 else "")
            + f" progress={progress_msg} "
            f"elapsed={_format_duration(elapsed)} "
            f"eta={_format_duration(eta_seconds)} "
            f"avg_step={avg_step_time:.3f}s"
        )
