from pathlib import Path

from miles.backends.training_utils.checkpoint.tracker import (
    CHECKPOINT_TRACKER_FILENAME,
    lora_adapter_checkpoint_dir,
    read_checkpoint_tracker_iteration,
    tracked_lora_adapter_dir,
    write_checkpoint_tracker_iteration,
)


def _write_tracker(directory: Path, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CHECKPOINT_TRACKER_FILENAME).write_text(text)
    return directory


class TestReadCheckpointTrackerIteration:
    def test_a_tracker_answers_the_iteration_it_names(self, tmp_path):
        """This is how a caller tells which iteration a checkpoint directory stands at."""
        assert read_checkpoint_tracker_iteration(_write_tracker(tmp_path / "run", "50\n")) == 50

    def test_a_directory_without_a_tracker_holds_no_checkpoint(self, tmp_path):
        """A --save directory exists from the moment the run starts, long before it holds a checkpoint."""
        (tmp_path / "run").mkdir()

        assert read_checkpoint_tracker_iteration(tmp_path / "run") is None

    def test_a_directory_that_does_not_exist_holds_no_checkpoint(self, tmp_path):
        """The first run of a job passes a --load path nothing has created yet."""
        assert read_checkpoint_tracker_iteration(tmp_path / "nope") is None

    def test_a_tracker_naming_no_number_holds_no_iteration(self, tmp_path):
        """Megatron writes 'release' there, which names no iteration a caller could snapshot beside."""
        assert read_checkpoint_tracker_iteration(_write_tracker(tmp_path / "run", "release\n")) is None

    def test_no_directory_at_all_holds_no_checkpoint(self):
        """--save is optional, and None must not reach the filesystem."""
        assert read_checkpoint_tracker_iteration(None) is None


class TestWriteCheckpointTrackerIteration:
    def test_the_written_iteration_is_the_one_read_back(self, tmp_path):
        write_checkpoint_tracker_iteration(tmp_path, 7)

        assert read_checkpoint_tracker_iteration(tmp_path) == 7

    def test_a_later_save_replaces_the_tracker_and_leaves_no_other_file(self, tmp_path):
        write_checkpoint_tracker_iteration(tmp_path, 7)

        write_checkpoint_tracker_iteration(tmp_path, 8)

        assert read_checkpoint_tracker_iteration(tmp_path) == 8
        assert [path.name for path in tmp_path.iterdir()] == [CHECKPOINT_TRACKER_FILENAME]


class TestTrackedLoraAdapterDir:
    def test_a_tracked_lora_save_answers_its_adapter_directory(self, tmp_path):
        lora_adapter_checkpoint_dir(tmp_path, 3).mkdir(parents=True)
        write_checkpoint_tracker_iteration(tmp_path, 3)

        assert tracked_lora_adapter_dir(tmp_path) == tmp_path / "iter_0000003" / "adapter"

    def test_a_tracked_megatron_checkpoint_holds_no_adapter_directory(self, tmp_path):
        """Megatron's own checkpoint holds the whole model, which its own loader restores."""
        (tmp_path / "iter_0000003").mkdir()
        write_checkpoint_tracker_iteration(tmp_path, 3)

        assert tracked_lora_adapter_dir(tmp_path) is None

    def test_only_the_tracked_iteration_counts(self, tmp_path):
        """An adapter directory the tracker does not name yet may still be half written."""
        lora_adapter_checkpoint_dir(tmp_path, 4).mkdir(parents=True)
        (tmp_path / "iter_0000003").mkdir()
        write_checkpoint_tracker_iteration(tmp_path, 3)

        assert tracked_lora_adapter_dir(tmp_path) is None

    def test_a_directory_without_a_tracker_holds_no_adapter_directory(self, tmp_path):
        lora_adapter_checkpoint_dir(tmp_path, 3).mkdir(parents=True)

        assert tracked_lora_adapter_dir(tmp_path) is None
        assert tracked_lora_adapter_dir(None) is None
