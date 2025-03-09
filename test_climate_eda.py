import unittest
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import re

class TestClimateEDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the notebook execution and extract code & markdown content."""
        # Load the notebook
        with open('climate_eda.ipynb', 'r', encoding='utf-8') as f:
            cls.notebook = nbformat.read(f, as_version=4)

        # Extract all code from notebook cells before execution
        cls.code_cells = [cell for cell in cls.notebook.cells if cell['cell_type'] == 'code']
        cls.markdown_cells = [cell for cell in cls.notebook.cells if cell['cell_type'] == 'markdown']
        
        cls.all_code = '\n'.join(cell['source'] for cell in cls.code_cells)
        cls.all_markdown = '\n'.join(cell['source'] for cell in cls.markdown_cells)

        # Execute the notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(cls.notebook, {'metadata': {'path': '.'}})

    def setUp(self):
        """Ensure all_code is available for each test case."""
        self.all_code = self.__class__.all_code
        self.all_markdown = self.__class__.all_markdown

    def test_bivariate_analysis(self):
        """Test for bivariate analysis visualizations."""
        bivariate_vis_patterns = [
            r"scatter\(", r"regplot\(", r"lineplot\(", r"barplot\(", r"heatmap\(", r"corr\("
        ]
        found_bivariate_vis = any(re.search(pattern, self.all_code) for pattern in bivariate_vis_patterns)
        self.assertTrue(found_bivariate_vis, "No evidence of bivariate visualizations")

    def calculate_grade(self):
        """Calculate the grade based on passing tests."""
        test_methods = [method for method in dir(self) if method.startswith('test_')]
        total_tests = len(test_methods)

        # Run tests and count passed ones
        passed_tests = 0
        for test in test_methods:
            test_method = getattr(self, test)
            try:
                test_method()
                passed_tests += 1
            except AssertionError:
                pass

        return round((passed_tests / total_tests) * 100)

if __name__ == '__main__':
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestClimateEDA)
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)

    # Calculate and print the grade
    test_case = TestClimateEDA()
    grade = test_case.calculate_grade()
    print(f"\nFinal Grade: {grade}/100")
