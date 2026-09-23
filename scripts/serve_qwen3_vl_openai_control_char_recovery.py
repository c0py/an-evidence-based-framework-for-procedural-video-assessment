#!/usr/bin/env python3
"""Representation-only Qwen server recovery for raw JSON control characters.

LM Format Enforcer constrains token sequences, but one tokenizer token can
still decode to an unescaped ASCII control character inside a JSON string.
This wrapper changes only that invalid surface representation to the exact
JSON escape sequence before the original server performs ``json.loads`` and
JSON-Schema validation.  Parsing the escaped text recovers the same Unicode
string value; no key, enum, number, boolean, or semantic decision is changed.
"""

from __future__ import annotations

import json as standard_json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def escape_raw_control_characters_in_json_strings(text: str) -> str:
    """Escape raw U+0000--U+001F only while inside a JSON string."""
    output: list[str] = []
    in_string = False
    escaped = False
    for character in text:
        if not in_string:
            output.append(character)
            if character == '"':
                in_string = True
            continue
        if escaped:
            output.append(character)
            escaped = False
        elif character == "\\":
            output.append(character)
            escaped = True
        elif character == '"':
            output.append(character)
            in_string = False
        elif ord(character) < 0x20:
            # json.dumps returns one quoted JSON string.  Removing its quotes
            # yields the canonical escape spelling whose parsed value is the
            # exact original control character.
            output.append(standard_json.dumps(character)[1:-1])
        else:
            output.append(character)
    return "".join(output)


class _RepresentationOnlyJsonProxy:
    JSONDecodeError = standard_json.JSONDecodeError

    @staticmethod
    def loads(value: str | bytes | bytearray, *args: Any, **kwargs: Any) -> Any:
        if isinstance(value, str):
            value = escape_raw_control_characters_in_json_strings(value)
        try:
            return standard_json.loads(value, *args, **kwargs)
        except standard_json.JSONDecodeError as error:
            diagnostic_path = os.environ.get("JSON_RECOVERY_DIAGNOSTIC_PATH")
            if diagnostic_path and isinstance(value, str):
                path = Path(diagnostic_path)
                raw_path = path.with_suffix(".raw.txt")
                if path.exists() or raw_path.exists():
                    raise RuntimeError(
                        f"Refusing to overwrite JSON recovery diagnostic: {path}"
                    ) from error
                raw_path.write_text(value, encoding="utf-8")
                path.write_text(standard_json.dumps({
                    "schema_version": "json_recovery_parse_diagnostic_v1",
                    "character_count": len(value),
                    "utf8_size_bytes": len(value.encode("utf-8")),
                    "sha256": __import__("hashlib").sha256(
                        value.encode("utf-8")
                    ).hexdigest(),
                    "parse_error": str(error),
                    "error_position": error.pos,
                    "prefix": value[:240],
                    "suffix": value[-240:],
                    "raw_response_path": str(raw_path.resolve()),
                    "labels_accessed": False,
                }, indent=2) + "\n", encoding="utf-8")
            raise

    @staticmethod
    def dumps(value: Any, *args: Any, **kwargs: Any) -> str:
        return standard_json.dumps(value, *args, **kwargs)


def main() -> None:
    from scripts import serve_qwen3_vl_openai as base_server

    # The imported server looks up its module-global ``json`` object at call
    # time.  Replacing only that reference preserves every other server path.
    base_server.json = _RepresentationOnlyJsonProxy
    base_server.main()


if __name__ == "__main__":
    main()
