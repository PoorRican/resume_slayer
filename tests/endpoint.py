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
            json = response.json()
            self.assertTrue('data' in json.keys())
            self.assertEqual({"data": "<p>processed resume</p>"}, response.json())

            # Assert that the Slayer.process() method was called with the correct arguments
            mock_process.assert_called_once()

    def test_html_output(self):
        with open("../job_desc.md") as f:
            description = f.read()
        with open("../resume.md") as f:
            resume = f.read()

        payload = {
            "resume": resume,
            "description": description,
            "title": "django engineer"
        }

        # Mock the Slayer.process() method
        with patch('main.Slayer.process') as mock_process:
            mock_process.return_value = resume

            response = self.client.post("/process/", json=payload)
            self.assertEqual(response.status_code, 200)

            # lazy way to assert text is html
            text = response.json()['data']
            self.assertEqual("<", text[0])
            self.assertEqual(">", text[-1])
