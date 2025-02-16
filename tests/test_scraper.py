import unittest
import os
from scripts.data_scraper import scrape_twitter_data


class TestScraper(unittest.TestCase):
    def setUp(self):
        # Define file paths
        self.raw_data_dir = "data/raw"
        self.raw_data_file = os.path.join(self.raw_data_dir, "raw_data.csv")

        # Ensure raw data directory exists
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def test_scraping(self):
        # Perform scraping
        scrape_twitter_data(keyword="#example", max_tweets=10)

        # Assert the raw data file is created
        self.assertTrue(os.path.exists(self.raw_data_file), "Raw data file should be created after scraping")

        # Assert the file is not empty
        with open(self.raw_data_file, "r") as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 1, "Raw data file should not be empty")

        # Assert the file contains a header and at least one tweet
        self.assertIn("tweet,likes", lines[0].strip(), "File should contain the correct header")

    def tearDown(self):
        # Cleanup: Remove the raw data file after testing
        if os.path.exists(self.raw_data_file):
            os.remove(self.raw_data_file)


if __name__ == "__main__":
    unittest.main()
