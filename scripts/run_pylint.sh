rm -f pylint.output
echo "Running Pylint on settings defined in pylintrc..."
pylint ../desc  > "pylint.output"
echo "Done!"