import json, tempfile, unittest
from pathlib import Path
import pandas as pd
from src.telemetry import load_registry, validate_observations, build_wide_packets, export_packets_json, read_source_table, import_table
ROOT=Path(__file__).resolve().parents[1]

class TelemetryTests(unittest.TestCase):
 def test_registry_is_not_sensor_claim(self):
  reg=load_registry(ROOT/'config/variable_registry.json')
  self.assertTrue(reg)
  self.assertTrue(all(v['is_instrument_sensor'] is False for v in reg.values()))
 def test_exported_observations_validate(self):
  reg=load_registry(ROOT/'config/variable_registry.json')
  rows=json.loads((ROOT/'data/processed/telemetry_observations.json').read_text())
  result=validate_observations(rows,reg)
  self.assertTrue(result['ok'],result['errors'][:5])
 def test_timestamp_not_fabricated(self):
  rows=json.loads((ROOT/'data/processed/telemetry_observations.json').read_text())
  self.assertTrue(all(r['timestamp_utc'] is None and r['elapsed_time_s'] is None for r in rows))
 def test_wide_packet_groups_variables(self):
  obs=[
   {"observation_id":"a|1|temp","source_row_key":"file.csv:1","source_file":"file.csv","experiment_id":"EXP1","psi_id":"PSI1","run_id":"R1","timestamp_utc":"2026-01-01T00:00:00Z","variable_id":"TEMP","variable_name":"temp","value":20,"unit":"C","measurement_class":"reported_experimental_variable"},
   {"observation_id":"a|2|o2","source_row_key":"file.csv:2","source_file":"file.csv","experiment_id":"EXP1","psi_id":"PSI1","run_id":"R1","timestamp_utc":"2026-01-01T00:00:00Z","variable_id":"O2","variable_name":"o2","value":21,"unit":"vol%","measurement_class":"reported_experimental_variable"}
  ]
  envelope,df=build_wide_packets(obs)
  self.assertEqual(envelope["packet_count"],1)
  self.assertEqual(envelope["packets"][0]["readings"],{"TEMP":20.0,"O2":21.0})
  self.assertEqual(df.loc[0,"TEMP"],20)
  self.assertEqual(df.loc[0,"O2"],21)
  self.assertIn("reading_metadata",envelope["packets"][0])
 def test_json_importer_accepts_records_envelope(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"sample.json"
   p.write_text(json.dumps({"observations":[{"psi_id":"PSI-X","value":1}]}))
   df=read_source_table(p)
   self.assertEqual(df.loc[0,"psi_id"],"PSI-X")
 def test_bass_multistep_flow_is_not_misread_as_one_number(self):
  observations, errors = import_table(ROOT/'data/processed/source_BASS_PSI-26_test_log_v1.0.csv', ROOT/'config/variable_registry.json')
  self.assertEqual(errors, [])
  airflow = [r for r in observations if r['variable_id']=='BASS::air_flow_settings_cm_s']
  self.assertEqual(len(airflow), 46)
  self.assertNotEqual(airflow[0]['value'], 105.0)
  self.assertIn('air_flow_settings_cm_s', airflow[0]['source_row_metadata'])
 def test_row_metadata_follows_source_column_order(self):
  # Field order must not depend on set iteration (PYTHONHASHSEED), so rebuilds are byte-identical.
  src=ROOT/'data/processed/source_BASS_PSI-26_test_log_v1.0.csv'
  cols=list(read_source_table(src).columns)
  observations, _ = import_table(src, ROOT/'config/variable_registry.json')
  keys=list(observations[0]['source_row_metadata'])
  self.assertEqual(keys, [c for c in cols if c in keys])
 def test_source_corrections_fully_applied(self):
  import sys; sys.path.insert(0,str(ROOT/'src'))
  from apply_source_corrections import apply
  rep=apply(check_only=True)
  self.assertEqual(rep['pending'],[])
  self.assertGreater(rep['already_correct'],0)
 def test_saffire_fabric_and_silicone_thickness_is_submillimetre(self):
  rows=json.loads((ROOT/'data/processed/telemetry_observations.json').read_text())
  thick=[r for r in rows if r['variable_id'] in ('SAFFIRE::thickness_mm','MICROGRAVITY::characteristic_size_mm')
         and str(r.get('experiment_id','')).startswith(('Saffire','SAFFIRE')) and 'PMMA' not in str(r.get('fuel_or_material'))]
  self.assertTrue(thick)
  self.assertTrue(all(r['value']<2 for r in thick),[(r['record_id'],r['value']) for r in thick if r['value']>=2])
 def test_vol_percent_oxygen_is_percentage_scale(self):
  rows=json.loads((ROOT/'data/processed/telemetry_observations.json').read_text())
  o2=[r for r in rows if r['unit']=='vol%' and 'o2' in r['variable_id'].lower() and 'co2' not in r['variable_id'].lower()]
  self.assertTrue(o2)
  self.assertTrue(all(1<r['value']<=100 for r in o2))
 def test_same_r_channel_inventory_covers_model_inputs(self):
  import csv as _csv
  inv=list(_csv.DictReader(open(ROOT/'ml_lead_handoff/SAME_R_CHANNEL_INVENTORY.csv',encoding='utf-8')))
  adapter=json.loads((ROOT/'ml_lead_handoff/ML_LEAD_SAME_R_ADAPTER.json').read_text(encoding='utf-8'))
  self.assertEqual(len(inv),98)
  mapped={r['source_column']:r['ml_model_input_key'] for r in inv if r['ml_model_input_key']}
  self.assertEqual(mapped,adapter['model_channels'])
 def test_json_dump_is_packet_envelope(self):
  env,df=build_wide_packets([])
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"packets.json"; export_packets_json(env,p)
   self.assertEqual(json.loads(p.read_text())["format"],"wide_packet_per_timestamp")

if __name__=='__main__': unittest.main()
