import tempfile, unittest
from pathlib import Path
from psi_source_table_importer import import_source_table

class ImporterTests(unittest.TestCase):
    def test_wide_packets_keep_source_fields_and_null_timestamp(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sample.csv"
            p.write_text("Test,Air Flow (cm/s),Outcome\nA,20,yes\nB,,none\n", encoding="utf-8")
            packets, numeric, meta = import_source_table(p, "PSI-TEST")
            self.assertEqual(len(packets), 2)
            self.assertIsNone(packets[0]["timestamp_utc"])
            self.assertEqual(packets[0]["source_fields"]["Air Flow (cm/s)"], "20")
            self.assertEqual(packets[1]["field_status"]["Air Flow (cm/s)"], "missing")
            self.assertEqual(numeric[0]["si_value"], 0.2)
            self.assertEqual(numeric[0]["si_unit"], "m/s")

    def test_range_is_not_collapsed_to_scalar(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sample.csv"
            p.write_text("Range\n21.5 - 21.7\n", encoding="utf-8")
            packets, numeric, meta = import_source_table(p, "PSI-TEST")
            self.assertEqual(len(packets), 1)
            self.assertEqual(numeric, [])

    def test_flex_burning_rate_unit_override(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "flex.csv"
            p.write_text("Burning rate; mm\n0.4\n", encoding="utf-8")
            packets, numeric, meta = import_source_table(p, "PSI-69")
            self.assertAlmostEqual(numeric[0]["si_value"], 0.4e-6)
            self.assertEqual(numeric[0]["si_unit"], "m²/s")

if __name__ == "__main__":
    unittest.main()
