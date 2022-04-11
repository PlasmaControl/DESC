rm -f flake8.output

echo "Running Flake8 using settings defined in flake8.ini..."
flake8 --config flake8rc --output-file flake8_errors.output ../desc/ ../tests/

echo "Generating Flake8 report..."
flake8 --config flake8rc -qqq --statistics --output-file flake8_summary.output ../desc/ ../tests/

echo "Running Pylint on settings defined in pylintrc..."
pylint ../desc  ../tests > "pylint.output"
echo "Done!"