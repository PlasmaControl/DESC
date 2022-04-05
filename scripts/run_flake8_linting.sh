rm -f flake8.output

echo "Running Flake8 using settings defined in flake8.ini..."
flake8 --config flake8rc --output-file flake8_errors.output ../desc/ ../tests/

echo "Generating Flake8 report..."
flake8 --config flake8.rc -qqq --statistics --output-file flake8_summary.output ../desc/ ../tests/