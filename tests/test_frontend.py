from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from hear.paths import AppPaths
from tests.helpers import make_test_app


def test_static_frontend_modules_served(tmp_path: Path) -> None:
    test_app = make_test_app(tmp_path)
    with TestClient(test_app) as client:
        index_resp = client.get("/")
        assert index_resp.status_code == 200
        assert "/static/app.css" in index_resp.text
        assert "/static/app.js" in index_resp.text

        for asset in (
            "/static/app.css",
            "/static/state.js",
            "/static/api.js",
            "/static/telemetry.js",
            "/static/audio.js",
            "/static/drawers.js",
            "/static/app.js",
        ):
            resp = client.get(asset)
            assert resp.status_code == 200, f"Failed to serve {asset}"


def test_static_frontend_dom_ids_and_imports_integrity() -> None:
    static_dir = AppPaths.from_env().static_dir
    html = (static_dir / "index.html").read_text(encoding="utf-8")
    html_ids = set(re.findall(r'\bid="([^"]+)"', html))

    exports_by_file: dict[Path, set[str]] = {}
    for js_file in sorted(static_dir.glob("*.js")):
        src = js_file.read_text(encoding="utf-8")
        exported = set(
            re.findall(
                r"\bexport\s+(?:async\s+)?(?:function|const|let|var|class)\s+([A-Za-z_$][\w$]*)",
                src,
            )
        )
        for block in re.findall(r"\bexport\s*\{([^}]+)\}", src):
            for item in block.split(","):
                cleaned = item.strip()
                if not cleaned:
                    continue
                parts = re.split(r"\s+as\s+", cleaned)
                exported.add(parts[-1].strip())
        exports_by_file[js_file.resolve()] = exported

    for js_file in sorted(static_dir.glob("*.js")):
        src = js_file.read_text(encoding="utf-8")
        for rel_import in re.findall(r'from\s+["\'](\.[^"\']+)["\']', src):
            target = (js_file.parent / rel_import).resolve()
            assert target.is_file(), f"Broken relative JS import '{rel_import}' in {js_file.name}"
        for names_block, rel_import in re.findall(
            r'import\s*\{([^}]+)\}\s*from\s*["\'](\.[^"\']+)["\']',
            src,
        ):
            target = (js_file.parent / rel_import).resolve()
            target_exports = exports_by_file.get(target, set())
            for item in names_block.split(","):
                cleaned = item.strip()
                if not cleaned:
                    continue
                imported_name = re.split(r"\s+as\s+", cleaned)[0].strip()
                assert imported_name in target_exports, (
                    f"Missing export '{imported_name}' in {target.name} "
                    f"(imported by {js_file.name})"
                )
        for dom_id in re.findall(r'getElementById\(\s*["\']([^"\']+)["\']\s*\)', src):
            assert dom_id in html_ids, (
                f"Missing DOM id='{dom_id}' in index.html (referenced by {js_file.name})"
            )
