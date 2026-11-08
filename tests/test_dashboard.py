import http.client
import json
from pathlib import Path
import threading
import tempfile
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from AutoPilot import Detector, MAX_UPLOAD, dispatch_tool, make_server
from SurroundCameraBenchmark import build_commands, POSITIONS


class FakeDetector:
    weights = 'test-model.pt'
    device = 'cpu'

    def predict(self, data, confidence):
        if data != b'jpeg':
            raise ValueError('Invalid image')
        return {'detections': [], 'inference_ms': 1, 'width': 320, 'height': 240}


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server('127.0.0.1', 0, FakeDetector(), 'test-key')
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def post(self, body=b'jpeg', query='', **headers):
        return self.request('POST', '/api/detect' + query, body, {
            'Authorization': 'Bearer test-key', 'Content-Type': 'image/jpeg', **headers,
        })

    def test_shell_and_manifest_assets_are_served(self):
        status, headers, body = self.request('GET', '/')
        self.assertEqual(status, 200)
        self.assertIn(b'AutoPilot', body)
        self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])
        _, _, manifest = self.request('GET', '/manifest.webmanifest')
        for icon in json.loads(manifest)['icons']:
            status, headers, body = self.request('GET', icon['src'])
            self.assertEqual(status, 200)
            decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(f'{decoded.shape[1]}x{decoded.shape[0]}', icon['sizes'])

    def test_api_requires_access_key(self):
        for method, path in [('GET', '/api/health'), ('POST', '/api/detect')]:
            with self.subTest(method=method):
                self.assertEqual(self.request(method, path)[0], 401)
        self.assertEqual(self.post(Authorization='Bearer wrong')[0], 401)

    def test_successful_detection_is_not_cached(self):
        status, headers, body = self.post()
        self.assertEqual(status, 200)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(json.loads(body)['width'], 320)

    def test_bad_input_and_nonfinite_thresholds_are_rejected(self):
        self.assertEqual(self.post(b'bad')[0], 400)
        self.assertEqual(self.post(**{'Content-Type': 'text/plain'})[0], 415)
        for confidence in ('NaN', 'inf', '-1', '1.1', 'hello'):
            with self.subTest(confidence=confidence):
                self.assertEqual(self.post(query=f'?confidence={confidence}')[0], 400)
        self.assertEqual(self.post(b'')[0], 413)
        self.assertEqual(self.post(**{'Content-Length': str(MAX_UPLOAD + 1)})[0], 413)

    def test_busy_and_model_failure_return_actionable_errors(self):
        with patch.object(FakeDetector, 'predict', side_effect=BlockingIOError('busy')):
            self.assertEqual(self.post()[0], 429)
        with patch.object(FakeDetector, 'predict', side_effect=RuntimeError('model missing')):
            self.assertEqual(self.post()[0], 503)

    def test_no_arbitrary_file_access(self):
        for path in ('/../README.md', '/%2e%2e/README.md', '/AutoPilot.py'):
            with self.subTest(path=path):
                self.assertEqual(self.request('GET', path)[0], 404)

    def test_decoder_rejects_invalid_and_oversize_images_before_model_load(self):
        detector = Detector('not-a-real-model.pt', 'cpu')
        with self.assertRaises(ValueError):
            detector.predict(b'garbage', .35)
        _, image = cv2.imencode('.jpg', np.zeros((10, 2049, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            detector.predict(image.tobytes(), .35)
        self.assertIsNone(detector.model)


class PackagedCliTests(unittest.TestCase):
    def test_dispatch_preserves_tool_arguments_and_adds_default_weights(self):
        with patch('AutoPilot.runpy.run_module') as run, patch.object(sys, 'argv', []):
            self.assertTrue(dispatch_tool(['--tool', 'risk', '--source', '0']))
            run.assert_called_once_with('RiskVideoBenchmark', run_name='__main__')
            self.assertEqual(sys.argv[1:3], ['--source', '0'])
            self.assertEqual(sys.argv[-2], '--weights')
        with patch('AutoPilot.runpy.run_module'), patch.object(sys, 'argv', []):
            dispatch_tool(['--tool', 'video', '--weights=custom.pt'])
            self.assertEqual(sys.argv[1:], ['--weights=custom.pt'])
        self.assertFalse(dispatch_tool(['--port', '8765']))
        with self.assertRaises(SystemExit):
            dispatch_tool(['--tool', 'unknown'])

    def test_frozen_surround_workers_relaunch_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'cameras.json'
            config.write_text(json.dumps({position: {'source': i} for i, position in enumerate(POSITIONS)}))
            with patch.object(sys, 'frozen', True, create=True):
                commands = build_commands(config, Path(directory), [])
            self.assertEqual(len(commands), 5)
            for command in commands:
                self.assertEqual(command[:3], [sys.executable, '--tool', 'risk'])


if __name__ == '__main__':
    unittest.main()
