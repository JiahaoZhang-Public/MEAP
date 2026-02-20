from argparse import Namespace

import pytest
import scripts.release.release as release_script


def test_validate_version_requires_v_prefix():
    release_script._validate_version("v1.0.0-rc1")
    with pytest.raises(ValueError):
        release_script._validate_version("1.0.0-rc1")


def test_notes_renders_template(tmp_path):
    template = tmp_path / "template.md"
    template.write_text("Release {{VERSION}} on {{DATE_UTC}} tag={{TAG}}\n", encoding="utf-8")
    output = tmp_path / "notes.md"
    args = Namespace(version="v1.0.0-rc1", template=str(template), output=str(output))

    release_script.cmd_notes(args)

    text = output.read_text(encoding="utf-8")
    assert "v1.0.0-rc1" in text
    assert "tag=v1.0.0-rc1" in text
    assert "{{VERSION}}" not in text


def test_set_version_updates_pyproject(monkeypatch, tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "mm-eap"\nversion = "1.0.0.dev1"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(release_script, "PYPROJECT_PATH", pyproject)
    args = Namespace(version="1.0.0rc1")

    release_script.cmd_set_version(args)

    updated = pyproject.read_text(encoding="utf-8")
    assert 'version = "1.0.0rc1"' in updated
