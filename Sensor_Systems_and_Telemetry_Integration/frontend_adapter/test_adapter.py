import json
import os
import tempfile
import unittest
from telemetry_frontend_adapter import to_frontend_payload, latest_value, write_frontend_json

class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.data = {"source": "NASA sample", "packets": [
            {"packet_id": "r1", "timestamp_utc": "2026-01-01T00:00:00Z", "experiment": "E1",
             "readings": {"temperature": {"value": 301.2, "unit": "K", "name": "Temperature"},
                          "oxygen": {"value": None, "unit": "%", "name": "Oxygen"}}},
            {"packet_id": "r2", "timestamp_utc": None, "experiment": "E2",
             "readings": {"temperature": {"value": 302.1, "unit": "K", "name": "Temperature"}}}
        ]}

    def test_packet_counts_and_no_fake_time(self):
        p = to_frontend_payload(self.data)
        self.assertEqual(p["packet_count"], 2)
        self.assertEqual(p["timestamped_packet_count"], 1)
        self.assertEqual(p["latest"]["timestamp_utc"], "2026-01-01T00:00:00Z")
        self.assertIsNone(p["last_record"]["timestamp_utc"])
        self.assertEqual(p["last_record"]["time_basis"], "untimed_record")

    def test_no_timestamp_means_no_latest_measurement(self):
        p = to_frontend_payload({"packets": [{"packet_id": "p1", "timestamp_utc": None,
            "readings": {"temp": 3}, "units": {"temp": "K"}}]})
        self.assertIsNone(p["latest"])
        self.assertIsNotNone(p["last_record"])
        self.assertEqual(latest_value(p, "temp"), (None, None))

    def test_latest_value(self):
        p = to_frontend_payload(self.data)
        self.assertEqual(latest_value(p, "temperature"), (301.2, "K"))

    def test_preserves_source_metadata(self):
        data = {"packets": [{"packet_id": "p1", "timestamp_utc": None,
            "source_file": "NASA.csv", "source_row_metadata": {"test_date": "2009-01-01"},
            "reading_metadata": {"FLEX::o2_pct": {"value_type": "reported", "quality_flags": "ok"}},
            "readings": {"FLEX::o2_pct": 21.0}, "units": {"FLEX::o2_pct": "vol%"},
            "variable_names": {"FLEX::o2_pct": "o2_pct"},
            "measurement_classes": {"FLEX::o2_pct": "reported_experimental_variable"}}]}
        payload = to_frontend_payload(data)
        packet = payload["packets"][0]
        self.assertEqual(packet["source_file"], "NASA.csv")
        self.assertEqual(packet["source_row_metadata"]["test_date"], "2009-01-01")
        self.assertEqual(packet["readings"]["FLEX::o2_pct"]["source_metadata"]["quality_flags"], "ok")

    def test_json_dump(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.json")
            write_frontend_json(self.data, path)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["packet_count"], 2)


    def test_same_r_channel_packets_are_supported(self):
        from telemetry_frontend_adapter import to_frontend_payload
        packet = {
            "packet_id": "same-r-1",
            "timestamp_utc": "2010-07-31T12:25:16Z",
            "timestamp_source": "source_epoch_s_unix_seconds",
            "time_basis": "source_epoch_s_unix_utc",
            "source_epoch_s": 1280492716.0,
            "elapsed_test_time_s": 79.0,
            "time_string": "2010/211/12:25:16",
            "test_id": 58,
            "material": "Teflon",
            "label_verified": True,
            "channels": {"iss_obs": 3.4, "ptrak": None},
            "quality_flags": ["ptrak_minus_999_sentinel"]
        }
        payload = to_frontend_payload({"packets": [packet]})
        p = payload["packets"][0]
        self.assertEqual(payload["timestamped_packet_count"], 1)
        self.assertEqual(p["source_epoch_s"], 1280492716.0)
        self.assertEqual(p["readings"]["iss_obs"]["unit"], "V")
        self.assertIsNone(p["readings"]["ptrak"]["value"])
        self.assertEqual(p["readings"]["ptrak"]["source_metadata"]["quality_flags"],
                         ["ptrak_minus_999_sentinel"])
        self.assertIsNone(p["readings"]["iss_obs"]["source_metadata"])
        self.assertTrue(p["label_verified"])

    def test_saffire_registry_metadata_is_applied(self):
        packet = {
            "packet_id": "psi100-1",
            "timestamp_utc": "2017-06-05T13:15:00.155Z",
            "timestamp_source": "CygnusTimestamp_epoch_milliseconds",
            "source_sheet": "RawData",
            "channels": {"SAFFIRE3::pcm_air_temp_degc": 20.6},
            "quality_flags": []
        }
        registry = {"variables": [{
            "variable_id": "SAFFIRE3::pcm_air_temp_degc",
            "display_name": "PCM Air Temp (degC)",
            "unit": "degC",
            "measurement_class": "telemetry_channel",
            "source_column": "PCM Air Temp (degC)",
            "instrument_identity_status": "header-level name only"
        }]}
        payload = to_frontend_payload({"channel_registry": registry, "packets": [packet]})
        reading = payload["packets"][0]["readings"]["SAFFIRE3::pcm_air_temp_degc"]
        self.assertEqual(reading["unit"], "degC")
        self.assertEqual(reading["source_column"], "PCM Air Temp (degC)")

    def test_jsonl_loader_reads_packets(self):
        from telemetry_frontend_adapter import load_and_adapt_jsonl
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "packets.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"packet_id":"a","timestamp_utc":None,"channels":{"x":1}})+"\n")
            payload = load_and_adapt_jsonl(path)
            self.assertEqual(payload["packet_count"], 1)
            self.assertEqual(payload["packets"][0]["readings"]["x"]["value"], 1)

if __name__ == "__main__":
    unittest.main()
