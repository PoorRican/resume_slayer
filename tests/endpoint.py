import unittest
from unittest.mock import patch
from main import app
from fastapi.testclient import TestClient


class TestProcessEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_process_endpoint(self):
        # Define the request payload
        payload = {
            "resume": "John Doe",
            "description": "Software Engineer",
            "title": "Resume"
        }

        # Mock the Slayer.process() method
        with patch('main.Slayer.process') as mock_process:
            mock_process.return_value = "processed resume"

            # Send a POST request to the /process/ endpoint
            response = self.client.post("/process/", json=payload)

            # Assert that the response status code is 200
            self.assertEqual(response.status_code, 200)

            # Assert that the response contains the expected data
            self.assertEqual(response.json(), {"data": "processed resume"})

            # Assert that the Slayer.process() method was called with the correct arguments
            mock_process.assert_called_once()
