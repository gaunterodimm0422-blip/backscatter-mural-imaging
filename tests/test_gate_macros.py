import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "gate"


class GateMacroTests(unittest.TestCase):
    def test_all_macro_aliases_are_supplied_by_runner(self):
        aliases = set()
        for path in GATE.glob("*.mac"):
            aliases.update(re.findall(r"\{([A-Z0-9_]+)\}", path.read_text(encoding="utf-8")))
        runner = (GATE / "run_baseline.sh").read_text(encoding="utf-8")
        supplied = set(re.findall(r"\[([A-Z0-9_]+),", runner))
        self.assertEqual(aliases, supplied)

    def test_batch_macro_ends_with_exit(self):
        lines = [
            line.strip()
            for line in (GATE / "main_baseline.mac").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(lines[-1], "exit")

    def test_truth_and_phantom_sd_are_not_accidentally_removed(self):
        phantom = (GATE / "phantom_concrete.mac").read_text(encoding="utf-8")
        truth = (GATE / "truth_concrete.mac").read_text(encoding="utf-8")
        main = (GATE / "main_baseline.mac").read_text(encoding="utf-8")
        self.assertIn("/gate/concrete/attachPhantomSD", phantom)
        self.assertIn("PhaseSpaceActor TruthConcrete", truth)
        self.assertIn("storeSecondaries true", truth)
        self.assertIn("materials/BackscatterMaterials.db", main)
        self.assertNotIn("/opt/gate/GateMaterials.db", main)
        database = GATE / "materials" / "BackscatterMaterials.db"
        self.assertTrue(database.exists())
        contents = database.read_text(encoding="utf-8")
        for material in ("Vacuum:", "Concrete:", "Water:", "Iron:", "GAGG:"):
            self.assertIn(material, contents)

    def test_standard_target_runner_covers_all_phantoms(self):
        runner = (GATE / "run_standard.sh").read_text(encoding="utf-8")
        for name in (
            "thin_layer",
            "rod",
            "double_rod",
            "point",
            "double_point",
            "edge",
        ):
            self.assertIn(f"{name})", runner)
        target_truth = (GATE / "truth_target.mac").read_text(encoding="utf-8")
        self.assertIn("/gate/actor/TruthTarget/attachTo target", target_truth)
        for path in GATE.glob("phantom_*.mac"):
            if path.name in {
                "phantom_concrete.mac",
                "phantom_concrete_void.mac",
            }:
                continue
            contents = path.read_text(encoding="utf-8")
            self.assertIn("/gate/target/attachPhantomSD", contents)

    def test_concrete_void_is_a_nested_vacuum_cylinder(self):
        phantom = (GATE / "phantom_concrete_void.mac").read_text(
            encoding="utf-8"
        )
        runner = (GATE / "run_concrete_void.sh").read_text(encoding="utf-8")
        self.assertIn("/gate/concrete/daughters/insert cylinder", phantom)
        self.assertIn("/gate/void/setMaterial Vacuum", phantom)
        self.assertIn("/gate/concrete/attachPhantomSD", phantom)
        self.assertIn('PHANTOM_MAC="phantom_concrete_void.mac"', runner)

    def test_formal_campaign_batches_are_auditable(self):
        campaign = (GATE / "run_campaign_batch.sh").read_text(
            encoding="utf-8"
        )
        for batch in (
            "rods)",
            "double_rods_a)",
            "double_rods_b)",
            "edge)",
            "materials)",
            "edge_material)",
            "void_pilot)",
        ):
            self.assertIn(batch, campaign)
        self.assertIn("SKIP_EXPORT=1", campaign)
        self.assertIn('RUN_TAG="${RUN_TAG:-}"', campaign)
        self.assertIn('RUN_SUFFIX="_${RUN_TAG}"', campaign)
        self.assertIn('ROD_PRIMARIES="${ROD_PRIMARIES:-5000000}"', campaign)
        self.assertIn(
            'DOUBLE_ROD_PRIMARIES="${DOUBLE_ROD_PRIMARIES:-5000000}"',
            campaign,
        )
        self.assertIn(
            'VOID_PRIMARIES="${VOID_PRIMARIES:-2000000}"',
            campaign,
        )
        self.assertIn("Refusing to overwrite an existing run", campaign)
        self.assertIn("(exit ${exit_code})", campaign)
        self.assertIn('N_PRIMARIES="$DOUBLE_ROD_PRIMARIES"', campaign)
        self.assertIn('N_PRIMARIES="$VOID_PRIMARIES"', campaign)
        self.assertIn("RANDOM_SEED=24061875", campaign)

        watcher = (GATE / "watch_rods.sh").read_text(encoding="utf-8")
        self.assertIn("Active GATE processes:", watcher)
        self.assertIn("Completed: ${complete_count}/3", watcher)
        self.assertIn("5,000,000-event completion marker", watcher)

        campaign_watcher = (GATE / "watch_campaign.sh").read_text(
            encoding="utf-8"
        )
        for batch in (
            "void_pilot)",
            "double_rods_a)",
            "double_rods_b)",
            "edge)",
            "materials)",
            "edge_material)",
        ):
            self.assertIn(batch, campaign_watcher)
        self.assertIn("expected-event completion marker", campaign_watcher)

    def test_large_run_wrapper_uses_local_disk_then_exports(self):
        wrapper = (GATE / "run_local_export.sh").read_text(encoding="utf-8")
        worker = (GATE / "run_local_worker.sh").read_text(encoding="utf-8")
        self.assertIn("/home/vgate/backscatter_runs", wrapper)
        self.assertIn('bash "${WORK_GATE}/run_local_worker.sh"', wrapper)
        self.assertIn('cp -a "${WORK_GATE}/output/${RUN_ID}"', worker)
        self.assertIn("flock 9", worker)
        self.assertIn("Export directory already exists", wrapper)

    def test_blind_qualification_keeps_answers_out_of_public_bundle(self):
        runner = (
            GATE / "run_blind_void_qualification.sh"
        ).read_text(encoding="utf-8")
        watcher = (
            GATE / "watch_blind_void_qualification.sh"
        ).read_text(encoding="utf-8")
        reveal = (
            GATE / "reveal_blind_void_qualification.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("secrets.randbits", runner)
        self.assertIn("truth_key.json", runner)
        self.assertIn("truth_salt.txt", runner)
        self.assertIn(
            '(private_dir / "truth_commitment.sha256")',
            runner,
        )
        self.assertIn("salt + key_path.read_bytes()", runner)
        self.assertIn("MAX_PARALLEL > 3", runner)
        self.assertIn("SKIP_EXPORT=1", runner)
        self.assertIn("hits.root", runner)
        self.assertIn("public_manifest.json", runner)
        self.assertNotIn(
            'cp "${private_result}/run_parameters.txt"',
            runner,
        )
        self.assertNotIn(
            'cp "${private_result}/hits_truth_concrete.root"',
            runner,
        )
        self.assertIn("SCORE_BUNDLE_SHA256", reveal)
        self.assertIn(
            'COMMITMENT_PATH="${PRIVATE_CAMPAIGN}/truth_commitment.sha256"',
            reveal,
        )
        self.assertIn("Truth-key commitment verification failed", reveal)
        self.assertIn(
            "intentionally does not display private class or position",
            watcher,
        )
        self.assertNotIn("pgrep -af", watcher)


if __name__ == "__main__":
    unittest.main()
