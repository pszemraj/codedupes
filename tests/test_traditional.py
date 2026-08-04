from __future__ import annotations

from pathlib import Path
from textwrap import dedent, indent

import pytest

from codedupes import traditional as traditional_module
from codedupes.traditional import (
    build_reference_graph,
    extract_identifiers,
    find_potentially_unused,
    run_traditional_analysis,
)
from tests.conftest import extract_units


def test_exact_duplicates_via_ast_hash(tmp_path: Path) -> None:
    source = dedent(
        """
        def foo(a, b):
            return a + b

        def bar(x, y):
            return x + y
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)

    exact, near = run_traditional_analysis(units, jaccard_threshold=0.85)

    assert len(exact) == 1
    assert len(near) == 0
    methods = {pair.method for pair in exact}
    assert methods == {"ast_hash"}


def test_same_scope_redefinitions_keep_distinct_pair_identities(tmp_path: Path) -> None:
    source = dedent(
        """
        def same(x):
            first = x + 1
            return first

        def same(y):
            second = y + 1
            return second

        def same(z):
            third = z + 1
            return third
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)

    exact, _near = run_traditional_analysis(units)

    assert len({unit.uid for unit in units}) == 3
    assert {tuple(sorted((pair.unit_a.lineno, pair.unit_b.lineno))) for pair in exact} == {
        (1, 5),
        (1, 9),
        (5, 9),
    }


def test_near_duplicates_threshold_boundary(tmp_path: Path) -> None:
    source = dedent(
        """
        def first(a, b):
            return a + b + a

        def second(a, c):
            return a + c + c

        def third(a, b):
            return b + 2
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)

    exact_low, near_low = run_traditional_analysis(units, jaccard_threshold=0.3)
    _exact_high, near_high = run_traditional_analysis(units, jaccard_threshold=0.95)

    assert len(near_low) >= 1
    assert len(near_high) == 0
    assert len(exact_low) == 0


def test_identifier_extraction_ignores_python_builtins() -> None:
    identifiers = extract_identifiers("def count(values):\n    return len(list(values))\n")

    assert identifiers == {"count", "values"}


def test_alias_aware_reference_graph(tmp_path: Path) -> None:
    source = dedent(
        """
        def helper(value):
            return value

        alias = helper

        def caller(value):
            return alias(value)

        def dead():
            return 0
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=False)
    build_reference_graph(units)

    unused = find_potentially_unused(units, strict_unused=True)
    names = {unit.name for unit in unused}

    assert "helper" not in names
    assert "caller" in names
    assert "dead" in names


def test_public_function_is_skipped_by_default(tmp_path: Path) -> None:
    source = dedent(
        """
        def public_function():
            return 1

        def _private_function():
            return 2

        def _unused_private():
            return _private_function() + public_function()
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    unused = find_potentially_unused(units, strict_unused=False)

    names = {unit.name for unit in unused}
    assert "public_function" not in names
    assert "_private_function" in names


def test_noqa_and_main_block_mark_as_used(tmp_path: Path) -> None:
    source = dedent(
        """
        def ignored_unused():  # noqa: codedupes
            return 42

        def used_by_main():
            return 7

        if __name__ == "__main__":
            used_by_main()
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)
    names = {unit.name for unit in unused}

    assert "ignored_unused" not in names
    assert "used_by_main" not in names


def test_pyproject_entry_points_mark_as_used(tmp_path: Path) -> None:
    source = dedent(
        """
        def cli_entry():
            return 1

        def helper():
            return 2
        """
    ).strip()
    (tmp_path / "pyproject.toml").write_text(
        dedent(
            """
            [project]
            name = "sample"
            scripts = { sample-cli = "sample_module:cli_entry" }
            """
        ).strip()
    )
    project = tmp_path / "src"
    project.mkdir()
    (project / "__init__.py").write_text("")
    (project / "sample_module.py").write_text(source)
    extractor_file = project / "sample_module.py"

    from codedupes.extractor import CodeExtractor

    units = list(CodeExtractor(project).extract_from_file(extractor_file))
    assert len(units) == 2
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)
    names = {unit.name for unit in unused}
    assert "cli_entry" not in names
    assert "helper" in names


def test_main_block_references_are_resolved_once_per_file(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        def first():
            return 1

        def second():
            return 2

        if __name__ == "__main__":
            first()
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    extractions: list[tuple[Path, str]] = []
    resolutions: list[str] = []

    def fake_extract_main_block_references(
        path: Path,
        module_name: str,
    ) -> tuple[set[str], set[str], set[str]]:
        extractions.append((path, module_name))
        return {"first"}, set(), set()

    def fake_resolve_reference_targets(reference: str, _aliases: dict[str, str]) -> set[str]:
        resolutions.append(reference)
        return {reference}

    monkeypatch.setattr(
        traditional_module,
        "_extract_main_block_references",
        fake_extract_main_block_references,
    )
    monkeypatch.setattr(
        traditional_module,
        "_resolve_reference_targets",
        fake_resolve_reference_targets,
    )

    build_reference_graph(units)

    assert len(extractions) == 1
    assert extractions[0][1] == "sample"
    assert resolutions == ["first"]


def test_reference_graph_keeps_nested_global_and_nonlocal_targets_distinct(
    tmp_path: Path,
) -> None:
    source = dedent(
        """
        def target():
            return "module"

        def late():
            return "module"

        def outer():
            def target():
                return "enclosing"

            def use_nonlocal():
                nonlocal target
                return target()

            def use_global():
                global target
                return target()

            def use_late_binding():
                return late()

            def late():
                return "nested"

            return use_nonlocal, use_global, use_late_binding
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    by_name = {unit.qualified_name: unit for unit in units}

    build_reference_graph(units)

    module_target = by_name["sample.target"]
    enclosing_target = by_name["sample.outer.target"]
    global_caller = by_name["sample.outer.use_global"]
    nonlocal_caller = by_name["sample.outer.use_nonlocal"]
    module_late = by_name["sample.late"]
    nested_late = by_name["sample.outer.late"]
    late_caller = by_name["sample.outer.use_late_binding"]

    assert global_caller.uid in module_target.references
    assert global_caller.uid not in enclosing_target.references
    assert nonlocal_caller.uid in enclosing_target.references
    assert nonlocal_caller.uid not in module_target.references
    assert late_caller.uid in nested_late.references
    assert late_caller.uid not in module_late.references


@pytest.mark.parametrize(
    "unreachable_binding",
    [
        "_dead = 2",
        "from elsewhere import _dead",
        "_dead: int",
        "for _dead in ():\n    pass",
        "with context() as _dead:\n    pass",
        "try:\n    pass\nexcept Exception as _dead:\n    pass",
        "(_dead := 2)",
        "match value:\n    case _dead:\n        pass",
        "del _dead",
        "_dead += 1",
        "[_dead := 2 for _ in []]",
    ],
)
def test_reference_graph_honors_unreachable_compile_time_slots(
    tmp_path: Path,
    unreachable_binding: str,
) -> None:
    source = (
        'def _dead():\n    return "module"\n\n'
        "def caller():\n"
        "    _dead()\n"
        "    return\n"
        f"{indent(unreachable_binding, '    ')}\n"
    )
    units = extract_units(tmp_path, source, include_private=True)
    by_name = {unit.qualified_name: unit for unit in units}

    build_reference_graph(units)
    unused = {unit.qualified_name for unit in find_potentially_unused(units, strict_unused=True)}

    assert by_name["sample.caller"].uid not in by_name["sample._dead"].references
    assert "sample._dead" in unused


def test_unreachable_binding_respects_global_declaration(tmp_path: Path) -> None:
    source = dedent(
        """
        def _target():
            return "module"

        def caller():
            global _target
            _target()
            return
            _target = 2
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    by_name = {unit.qualified_name: unit for unit in units}

    build_reference_graph(units)

    assert by_name["sample.caller"].uid in by_name["sample._target"].references


def test_reference_graph_qualifies_module_and_function_local_import_aliases(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text(
        "def work():\n    return 'a'\n\n"
        "def unrelated():\n    return 'unrelated'\n\n"
        "class Remote:\n    def _run(self):\n        return 'remote'\n"
    )
    (tmp_path / "b.py").write_text("def work():\n    return 'b'\n")
    (tmp_path / "x.py").write_text("class Other:\n    def _run(self):\n        return 'other'\n")
    (tmp_path / "consumer.py").write_text(
        dedent(
            """
            from a import work as module_alias
            from a import work as stale_alias
            stale_alias = 42
            from a import work as with_alias
            with nullcontext(42) as with_alias:
                pass
            from a import work as empty_comp_alias
            [empty_comp_alias := 42 for _ in []]
            from a import work as starred_comp_alias
            [starred_comp_alias := 42 for _ in [*items]]
            from a import work as selected
            from a import work as immediate_alias

            class Immediate:
                value = immediate_alias()

            @immediate_alias
            def immediate_decorated():
                pass

            def immediate_default(value=immediate_alias()):
                return value

            from b import work as immediate_alias

            while True:
                try:
                    from a import work as finally_alias
                    risky()
                    from b import work as finally_alias
                finally:
                    break

            class a:
                def work(self):
                    return "consumer a"

            class b:
                def work(self):
                    return "consumer b"

            if choose_a:
                from a import work as module_conditional_alias
            else:
                from b import work as module_conditional_alias

            def use_module_conditional_alias():
                return module_conditional_alias()

            def use_module_alias():
                return module_alias()

            def use_stale_alias():
                return stale_alias

            def use_with_alias():
                return with_alias

            def use_empty_comp_alias():
                return empty_comp_alias()

            def use_starred_comp_alias():
                return starred_comp_alias()

            def use_mixed_alias(flag):
                global selected
                if flag:
                    from b import work as selected
                return selected()

            def use_finally_alias():
                return finally_alias()

            def use_break_alias():
                for _ in [0]:
                    from a import work as break_alias
                    break
                    from b import work as break_alias
                return break_alias()

            def use_while_break_alias():
                while True:
                    from a import work as while_alias
                    break
                    from b import work as while_alias
                return while_alias()

            def use_nested_continue_alias():
                for _ in [0]:
                    if True:
                        from a import work as nested_alias
                        continue
                        from b import work as nested_alias
                return nested_alias()

            def use_multi_iteration_alias(values):
                from b import work as loop_alias
                for _ in values:
                    loop_alias()
                    from a import work as loop_alias

            def use_mixed_receiver(flag, obj):
                if flag:
                    from a import Remote as receiver
                else:
                    receiver = obj
                return receiver._run()

            def use_late_immediate_alias():
                return immediate_alias()

            def use_missing_name():
                return unrelated()

            def use_known_receiver():
                from a import Remote
                return Remote._run

            def use_local_alias():
                from b import work as local_alias

                return local_alias()

            def use_conditional_alias(flag):
                if flag:
                    from a import work as conditional_alias
                else:
                    from b import work as conditional_alias

                return conditional_alias()
            """
        ).strip()
        + "\n"
    )
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "a.py").write_text("def work():\n    return 'pkg a'\n")
    (package / "consumer.py").write_text(
        "from .a import work as relative_alias\n\n"
        "def use_relative_alias():\n    return relative_alias()\n"
    )
    from codedupes.extractor import CodeExtractor

    units = CodeExtractor(tmp_path, include_private=True).extract_all()
    by_name = {unit.qualified_name: unit for unit in units}

    build_reference_graph(units)

    module_caller = by_name["consumer.use_module_alias"]
    stale_caller = by_name["consumer.use_stale_alias"]
    with_caller = by_name["consumer.use_with_alias"]
    empty_comp_caller = by_name["consumer.use_empty_comp_alias"]
    starred_comp_caller = by_name["consumer.use_starred_comp_alias"]
    mixed_caller = by_name["consumer.use_mixed_alias"]
    finally_caller = by_name["consumer.use_finally_alias"]
    break_caller = by_name["consumer.use_break_alias"]
    while_break_caller = by_name["consumer.use_while_break_alias"]
    nested_continue_caller = by_name["consumer.use_nested_continue_alias"]
    multi_iteration_caller = by_name["consumer.use_multi_iteration_alias"]
    mixed_receiver_caller = by_name["consumer.use_mixed_receiver"]
    late_immediate_caller = by_name["consumer.use_late_immediate_alias"]
    known_receiver_caller = by_name["consumer.use_known_receiver"]
    module_conditional_caller = by_name["consumer.use_module_conditional_alias"]
    local_caller = by_name["consumer.use_local_alias"]
    conditional_caller = by_name["consumer.use_conditional_alias"]
    relative_caller = by_name["pkg.consumer.use_relative_alias"]
    a_work = by_name["a.work"]
    a_unrelated = by_name["a.unrelated"]
    b_work = by_name["b.work"]
    package_a_work = by_name["pkg.a.work"]
    local_a_work = by_name["consumer.a.work"]
    local_b_work = by_name["consumer.b.work"]
    remote_method = by_name["a.Remote._run"]
    unrelated_method = by_name["x.Other._run"]
    assert module_caller.uid in a_work.references
    assert module_caller.uid not in b_work.references
    assert module_caller.uid not in local_a_work.references
    assert stale_caller.uid not in a_work.references
    assert with_caller.uid not in a_work.references
    assert empty_comp_caller.uid in a_work.references
    assert starred_comp_caller.uid in a_work.references
    assert mixed_caller.uid in a_work.references
    assert mixed_caller.uid in b_work.references
    assert finally_caller.uid in a_work.references
    assert finally_caller.uid in b_work.references
    assert break_caller.uid in a_work.references
    assert break_caller.uid not in b_work.references
    assert while_break_caller.uid in a_work.references
    assert while_break_caller.uid not in b_work.references
    assert nested_continue_caller.uid in a_work.references
    assert nested_continue_caller.uid not in b_work.references
    assert multi_iteration_caller.uid in a_work.references
    assert multi_iteration_caller.uid in b_work.references
    assert mixed_receiver_caller.uid in remote_method.references
    assert mixed_receiver_caller.uid in unrelated_method.references
    assert by_name["consumer.Immediate"].uid in a_work.references
    assert by_name["consumer.immediate_decorated"].uid in a_work.references
    assert by_name["consumer.immediate_default"].uid in a_work.references
    assert late_immediate_caller.uid in b_work.references
    assert late_immediate_caller.uid not in a_work.references
    assert not a_unrelated.references
    assert known_receiver_caller.uid in remote_method.references
    assert known_receiver_caller.uid not in unrelated_method.references
    assert module_conditional_caller.uid in a_work.references
    assert module_conditional_caller.uid in b_work.references
    assert local_caller.uid in b_work.references
    assert local_caller.uid not in a_work.references
    assert local_caller.uid not in local_b_work.references
    assert conditional_caller.uid in a_work.references
    assert conditional_caller.uid in b_work.references
    assert relative_caller.uid in package_a_work.references
    assert relative_caller.uid not in a_work.references


def test_reference_graph_tracks_abrupt_control_flow_aliases(tmp_path: Path) -> None:
    """Abrupt transfers must preserve only bindings that can reach later loads."""
    (tmp_path / "a.py").write_text("def work():\n    return 'a'\n")
    (tmp_path / "b.py").write_text("def work():\n    return 'b'\n")
    (tmp_path / "consumer.py").write_text(
        dedent(
            """
            def through_finally():
                for _ in [0]:
                    from a import work as selected
                    try:
                        break
                    finally:
                        from b import work as selected
                return selected()

            def break_in_finally():
                for _ in [0]:
                    try:
                        pass
                    finally:
                        from a import work as selected
                        break
                        from b import work as selected
                return selected()

            def return_in_with(manager):
                with manager:
                    from a import work as selected
                    return selected()
                    from b import work as selected

            def suppressing_with(flag, manager):
                if flag:
                    with manager:
                        try:
                            raise RuntimeError
                        finally:
                            pass
                    from b import work as selected
                else:
                    from a import work as selected
                return selected()

            def break_skips_else():
                for _ in [0]:
                    from a import work as selected
                    break
                else:
                    from b import work as selected
                return selected()

            def return_stops_suite():
                from a import work as selected
                return selected()
                from b import work as selected
            """
        ).strip()
        + "\n"
    )
    from codedupes.extractor import CodeExtractor

    units = CodeExtractor(tmp_path, include_private=True).extract_all()
    by_name = {unit.qualified_name: unit for unit in units}
    build_reference_graph(units)

    a_work = by_name["a.work"]
    b_work = by_name["b.work"]
    for caller_name in (
        "consumer.break_in_finally",
        "consumer.return_in_with",
        "consumer.break_skips_else",
        "consumer.return_stops_suite",
    ):
        caller = by_name[caller_name]
        assert caller.uid in a_work.references
        assert caller.uid not in b_work.references
    finally_caller = by_name["consumer.through_finally"]
    assert finally_caller.uid not in a_work.references
    assert finally_caller.uid in b_work.references
    suppressing_caller = by_name["consumer.suppressing_with"]
    assert suppressing_caller.uid in a_work.references
    assert suppressing_caller.uid in b_work.references


def test_reference_graph_tracks_program_point_and_repeated_aliases(tmp_path: Path) -> None:
    """Immediate, repeated, and module-qualified references retain target provenance."""
    (tmp_path / "a.py").write_text(
        "def work():\n    return 'a'\n\nclass Remote:\n    def run(self):\n        return 'a'\n"
    )
    (tmp_path / "b.py").write_text(
        "def work():\n    return 'b'\n\nclass Other:\n    def run(self):\n        return 'b'\n"
    )
    (tmp_path / "consumer.py").write_text(
        dedent(
            """
            from a import work
            decorator = work

            @decorator
            def decorated():
                pass

            decorator = 0
            while True:
                from a import work as while_decorator
                break
                from b import work as while_decorator

            @while_decorator
            def while_decorated():
                pass

            import a
            from b import Other

            def module_attribute():
                return a.Remote.run

            opaque_module = Other()

            def opaque_module_attribute():
                return opaque_module.run()

            def repeated_condition(flag):
                from b import work as selected
                while flag and selected():
                    from a import work as selected

            from a import work as after_loop
            for _ in values:
                pass
            else:
                from b import work as after_loop

            def use_after_loop():
                return after_loop()

            from a import work as filtered
            from b import work as replacement
            [None for _ in [0] if (filtered := replacement) for __ in []]

            def use_filtered():
                return filtered()

            from a import work as chained_filter
            [None for _ in [0] if False if (chained_filter := replacement)]

            def use_chained_filter():
                return chained_filter()

            from a import work as left
            from b import work as right
            [((tmp := left), (left := right), (right := tmp)) for _ in [0, 1]]

            def use_left():
                return left()

            def use_right():
                return right()
            """
        ).strip()
        + "\n"
    )
    from codedupes.extractor import CodeExtractor

    units = CodeExtractor(tmp_path, include_private=True).extract_all()
    by_name = {unit.qualified_name: unit for unit in units}
    build_reference_graph(units)

    a_work = by_name["a.work"]
    b_work = by_name["b.work"]
    for caller_name in ("consumer.decorated", "consumer.while_decorated"):
        caller = by_name[caller_name]
        assert caller.uid in a_work.references
        assert caller.uid not in b_work.references
    condition_caller = by_name["consumer.repeated_condition"]
    assert condition_caller.uid in a_work.references
    assert condition_caller.uid in b_work.references
    for caller_name in ("consumer.use_after_loop", "consumer.use_filtered"):
        caller = by_name[caller_name]
        assert caller.uid not in a_work.references
        assert caller.uid in b_work.references
    chained_filter_caller = by_name["consumer.use_chained_filter"]
    assert chained_filter_caller.uid in a_work.references
    assert chained_filter_caller.uid not in b_work.references
    left_caller = by_name["consumer.use_left"]
    right_caller = by_name["consumer.use_right"]
    assert left_caller.uid in a_work.references
    assert left_caller.uid not in b_work.references
    assert right_caller.uid not in a_work.references
    assert right_caller.uid in b_work.references
    module_attribute_caller = by_name["consumer.module_attribute"]
    assert module_attribute_caller.uid in by_name["a.Remote.run"].references
    assert module_attribute_caller.uid not in by_name["b.Other.run"].references
    opaque_attribute_caller = by_name["consumer.opaque_module_attribute"]
    assert opaque_attribute_caller.uid in by_name["b.Other.run"].references


def test_reference_graph_respects_class_lookup_order(tmp_path: Path) -> None:
    source = dedent(
        """
        def hook():
            return "module"

        class Early:
            before = hook()

            def hook(self):
                return "early"

        class Late:
            def hook(self):
                return "late"

            after = hook

            def caller(self):
                return hook()

        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    by_name = {unit.qualified_name: unit for unit in units}

    build_reference_graph(units)

    module_hook = by_name["sample.hook"]
    early_class = by_name["sample.Early"]
    early_hook = by_name["sample.Early.hook"]
    late_class = by_name["sample.Late"]
    late_hook = by_name["sample.Late.hook"]
    bare_method_caller = by_name["sample.Late.caller"]

    assert early_class.uid in module_hook.references
    assert early_class.uid not in early_hook.references
    assert late_class.uid in late_hook.references
    assert bare_method_caller.uid in module_hook.references
    assert bare_method_caller.uid not in late_hook.references


def test_entrypoint_roots_do_not_match_same_named_units_in_other_modules(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text(
        dedent(
            """
            def main():
                return "a main"

            def run():
                return "a run"

            if __name__ == "__main__":
                run()
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "b.py").write_text(
        dedent(
            """
            def main():
                return "b main"

            def run():
                return "b run"
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "pyproject.toml").write_text(
        dedent(
            """
            [project]
            name = "sample"
            scripts = { sample-cli = "a:main" }
            """
        ).strip()
        + "\n"
    )
    src_package = tmp_path / "src" / "pkg"
    src_package.mkdir(parents=True)
    (src_package / "__init__.py").write_text("")
    (src_package / "cli.py").write_text("def main():\n    return 'src main'\n")
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        pyproject_path.read_text().replace(
            'scripts = { sample-cli = "a:main" }',
            'scripts = { sample-cli = "a:main", src-cli = "pkg.cli:main" }',
        )
    )
    from codedupes.extractor import CodeExtractor

    units = CodeExtractor(tmp_path, include_private=True).extract_all()
    by_name = {unit.qualified_name: unit for unit in units}

    build_reference_graph(units, project_root=tmp_path)

    assert by_name["a.main"].references == {"project.entrypoint"}
    assert by_name["a.run"].references == {f"__main__::{tmp_path / 'a.py'}"}
    assert by_name["b.main"].references == set()
    assert by_name["b.run"].references == set()
    assert by_name["src.pkg.cli.main"].references == {"project.entrypoint"}

    root_package = tmp_path / "rootpkg"
    root_package.mkdir()
    (root_package / "__init__.py").write_text("def main():\n    return 'root package'\n")
    pyproject_path.write_text(
        pyproject_path.read_text().replace(
            'src-cli = "pkg.cli:main"',
            'src-cli = "pkg.cli:main", root-cli = "rootpkg:main"',
        )
    )
    root_units = CodeExtractor(root_package, include_private=True).extract_all()
    root_main = next(unit for unit in root_units if unit.qualified_name == "main")

    build_reference_graph(root_units, project_root=tmp_path)

    assert root_main.import_module_name == "rootpkg"
    assert root_main.references == {"project.entrypoint"}


def test_non_call_references_count_as_usage(tmp_path: Path) -> None:
    """Callback-style, property, and annotation references must mark units as used."""
    source = dedent(
        '''
        class Marker:
            """Annotation-only class."""

        class Config:
            """Holds a property accessed without a call."""

            @property
            def cached_value(self):
                return 1

        class _Marker:
            pass

        class _Holder:
            token = 1

        def validate(value):
            return value

        def register(callback):
            return callback

        def annotate(value: Marker) -> Marker:
            return value

        def wire():
            register(callback=validate)
            return Config().cached_value

        def same_spelled_local():
            return "module definition"

        def use_local_value():
            same_spelled_local = 2
            return same_spelled_local

        def inspect(value):
            match value:
                case _Marker():
                    return True
            return False

        def mutate_holder():
            _Holder.token = 2
            del _Holder.token

        def orphan():
            return None
        '''
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    build_reference_graph(units)
    unused_names = {unit.name for unit in find_potentially_unused(units, strict_unused=True)}

    # validate is only a keyword-argument reference, cached_value only a
    # property access, Marker only an annotation — none are calls.
    assert "validate" not in unused_names
    assert "cached_value" not in unused_names
    assert "Marker" not in unused_names
    assert "_Marker" not in unused_names
    assert "_Holder" not in unused_names
    # A local load must not suppress an unrelated same-spelled definition.
    assert "same_spelled_local" in unused_names
    # A genuinely unreferenced unit is still flagged.
    assert "orphan" in unused_names


def test_filtered_nested_definition_bodies_still_contribute_references(tmp_path: Path) -> None:
    source = dedent(
        """
        def target():
            return 1

        def outer():
            def _hidden():
                return target()

            return _hidden
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=False)
    by_name = {unit.qualified_name: unit for unit in units}

    build_reference_graph(units)

    assert "sample.outer._hidden" not in by_name
    assert by_name["sample.outer"].uid in by_name["sample.target"].references


def test_attribute_references_do_not_fall_through_to_module_functions(tmp_path: Path) -> None:
    source = dedent(
        """
        def _same_name():
            return "unused module function"

        class Box:
            def _same_name(self):
                return "live method"

            def caller(self):
                return self._same_name()
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    by_name = {unit.qualified_name: unit for unit in units}

    build_reference_graph(units)

    caller_uid = by_name["sample.Box.caller"].uid
    assert caller_uid in by_name["sample.Box._same_name"].references
    assert caller_uid not in by_name["sample._same_name"].references
    unused = {unit.qualified_name for unit in find_potentially_unused(units, strict_unused=True)}
    assert "sample.Box._same_name" not in unused
    assert "sample._same_name" in unused


def test_unused_analysis_skips_only_proven_ast_visitor_hooks(tmp_path: Path) -> None:
    source = dedent(
        """
        from ast import NodeTransformer as AstTransformer
        from framework import NodeVisitor

        class Visitor(AstTransformer):
            def visit_Name(self, node):
                return node

            def unused_helper(self):
                return 1

        class ImportedWorker(NodeVisitor):
            def visit_Name(self, node):
                return node

        class NodeVisitor:
            pass

        class LocalWorker(NodeVisitor):
            def visit_Name(self, node):
                return node

        class Ordinary:
            def visit_Name(self, node):
                return node
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)

    unused = find_potentially_unused(units, strict_unused=True)
    qualified_names = {unit.qualified_name for unit in unused}

    assert "sample.Visitor.visit_Name" not in qualified_names
    assert "sample.Visitor.unused_helper" in qualified_names
    assert "sample.LocalWorker.visit_Name" in qualified_names
    assert "sample.ImportedWorker.visit_Name" in qualified_names
    assert "sample.Ordinary.visit_Name" in qualified_names
