import unittest
from unittest.mock import patch
from main import app
from fastapi.testclient import TestClient


class TestProcessEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_process_endpoint(self):
        payload = {
            "resume": "John Doe",
            "description": "Software Engineer",
            "title": "Resume"
        }

        # Mock the Slayer.process() method
        with patch('main.Slayer.process') as mock_process:
            mock_process.return_value = "processed resume"

            response = self.client.post("/process/", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual("processed resume", response.json())

            # Assert that the Slayer.process() method was called with the correct arguments
            mock_process.assert_called_once()
