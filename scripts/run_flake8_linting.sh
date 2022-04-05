rm -f flake8_errors.output flake8_summary.output
echo "Generating Flake8 Error Summary"
flake8 --config flake8.ini -qqq --output-file flake8_summary.output --statistics ../desc/ ../tests/
echo "Generating Flake8 Full Report"
flake8 --config flake8.ini --output-file flake8_errors.output ../desc/ ../tests/
echo "Number of formatting errors found:"
flake8 --config flake8.ini -qqq --count ../desc/ ../tests/ || flake_return_code=$?
echo "FLAKE_RETURN_CODE =$flake_return_code"