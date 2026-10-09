"""
Unified Test Runner for M. Abdullah ML Projects Portfolio.
Executes test suites for each subproject in its isolated directory context
and reports a comprehensive summary.
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_SUITES = [
    ("fastapi-ml-deployment", PROJECT_ROOT / "fastapi-ml-deployment"),
    ("ml-pipeline", PROJECT_ROOT / "ml-pipeline"),
    ("retail-sales-forecaster", PROJECT_ROOT / "retail-sales-forecaster"),
    ("resume-matcher", PROJECT_ROOT / "resume-matcher"),
    ("nlp-fine-tuning-api", PROJECT_ROOT / "nlp-fine-tuning-api"),
]


def run_tests():
    python_bin = sys.executable
    print("=" * 70)
    print("RUNNING ALL ML PORTFOLIO TEST SUITES - M. ABDULLAH")
    print("=" * 70)

    results = {}
    all_passed = True

    for name, project_dir in TEST_SUITES:
        tests_dir = project_dir / "tests"
        if not tests_dir.exists():
            continue

        print(f"\n[SUITE] Executing tests for: {name} (cwd: {project_dir})...")
        cmd = [python_bin, "-m", "pytest", "tests", "-v", "--no-header"]
        proc = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True)

        passed = (proc.returncode == 0)
        results[name] = {
            "passed": passed,
            "stdout": proc.stdout,
            "stderr": proc.stderr
        }

        if passed:
            # Count passed tests
            count_line = [line for line in proc.stdout.splitlines() if "passed" in line]
            summary = count_line[-1] if count_line else "PASSED"
            print(f"  -> SUCCESS: {summary.strip()}")
        else:
            all_passed = False
            print(f"  -> FAILED (exit code {proc.returncode})")
            print(proc.stdout)
            if proc.stderr:
                print(proc.stderr)

    print("\n" + "=" * 70)
    print("TEST EXECUTION SUMMARY")
    print("=" * 70)
    for name, res in results.items():
        status = "PASSED [OK]" if res["passed"] else "FAILED [X]"
        print(f" - {name.ljust(30)}: {status}")
    print("=" * 70)

    if all_passed:
        print("ALL SUBPROJECT TEST SUITES PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("SOME TEST SUITES FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
