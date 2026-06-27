"""
@file test_cross_seed_discovery.py
@brief Tests for the seed_roots() cross-seed sub-root locator in _discovery.py.

Validates that seed_roots enumerates the seed_<N>/ children of an output root in
sorted order, and that a root with no seed nesting degrades to the singleton
[root] so the cross-seed aggregator pools "one seed" transparently (pool of one).
"""

from pathlib import Path
from typing import List

from scripts.evaluation._discovery import seed_roots


def _touch(path: Path) -> None:
    """
    @brief Create an empty file, making parent directories as needed.
    @param path: File path to create.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


class TestSeedRoots:
    """
    @class TestSeedRoots
    @brief Tests for _discovery.seed_roots.
    """

    def test_enumerates_seed_children_sorted(self, tmp_path: Path) -> None:
        """
        @brief Multiple seed_<N>/ dirs are returned, sorted, directories only.
        """
        for seed in ("seed_123", "seed_42", "seed_7"):
            (tmp_path / seed).mkdir()
        # A stray non-seed dir and a stray file must be ignored.
        (tmp_path / "all_seeds").mkdir()
        _touch(tmp_path / "seed_note.txt")

        roots: List[Path] = seed_roots(tmp_path)

        assert roots == sorted(roots)
        names = [p.name for p in roots]
        assert names == ["seed_123", "seed_42", "seed_7"]
        assert all(p.is_dir() for p in roots)

    def test_no_seed_nesting_degrades_to_singleton(self, tmp_path: Path) -> None:
        """
        @brief A root with no seed_*/ children returns [root] (pool of one).
        """
        # A plain results tree (one arm/leaf, no seed_<N> layer).
        _touch(
            tmp_path
            / "full_method"
            / "6_42_27062026-0000"
            / "without_wrapper"
            / "episode_records.csv"
        )

        roots = seed_roots(tmp_path)

        assert roots == [tmp_path]

    def test_empty_root_degrades_to_singleton(self, tmp_path: Path) -> None:
        """
        @brief An empty root still returns [root] rather than an empty list.
        """
        assert seed_roots(tmp_path) == [tmp_path]
